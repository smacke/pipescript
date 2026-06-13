from __future__ import annotations

import ast
from typing import TYPE_CHECKING, Any, Callable, cast

from pyccolo.stmt_mapper import StatementMapper

from pipescript.utils import get_user_ns

if TYPE_CHECKING:
    from pipescript.tracers.macro_tracer import ArgReplacer, MacroTracer

    CallableNodeTransformer = Callable[[ast.expr], ast.expr]


class DynamicMacroArgSubstitutor(ast.NodeTransformer):
    def __init__(
        self,
        arg_node_id_to_placeholder_name: dict[int, str],
        ordered_arg_names: list[str],
        arg_node_subst_exprs: list[ast.expr],
        dynamic_macros: dict[str, DynamicMacro],
        method_dynamic_macros: dict[str, DynamicMacro],
    ) -> None:
        self.arg_node_id_to_placeholder_name = arg_node_id_to_placeholder_name
        self.ordered_arg_names = ordered_arg_names
        self.arg_node_subst_exprs = arg_node_subst_exprs
        self.dynamic_macros = dynamic_macros
        self.method_dynamic_macros = method_dynamic_macros

    def visit_Subscript(self, node: ast.Subscript) -> ast.AST:
        if not isinstance(node.value, ast.Name):
            return self.generic_visit(node)
        dynamic_macro = self.dynamic_macros.get(
            node.value.id, self.method_dynamic_macros.get(node.value.id)
        )
        if not isinstance(dynamic_macro, TemplateDynamicMacro):
            return self.generic_visit(node)
        expanded = dynamic_macro.expand(node.slice)
        return expanded

    def visit_Name(self, node: ast.Name) -> ast.AST:
        arg_name = self.arg_node_id_to_placeholder_name.get(id(node))
        if arg_name is None:
            return super().generic_visit(node)
        return self.arg_node_subst_exprs[self.ordered_arg_names.index(arg_name)]


class DynamicMacro:
    def __init__(self, is_method: bool) -> None:
        self._is_method = is_method

    @property
    def is_method(self) -> bool:
        return self._is_method

    def expand(self, args: ast.expr) -> ast.expr | Any:
        raise NotImplementedError()

    @classmethod
    def create(
        cls, node_slice: ast.expr, tracer: MacroTracer, is_method: bool
    ) -> DynamicMacro:
        if (
            isinstance(node_slice, ast.Tuple)
            and len(node_slice.elts) > 1
            and all(isinstance(elt, ast.Name) for elt in node_slice.elts[1:])
        ):
            macro_template = node_slice.elts[0]
            ordered_arg_names = [cast(ast.Name, elt).id for elt in node_slice.elts[1:]]
        else:
            macro_template = node_slice
            ordered_arg_names = []
        transformer: CallableNodeTransformer | None = None
        if isinstance(macro_template, ast.Name) and len(ordered_arg_names) == 0:
            user_ns = get_user_ns()
            if user_ns is not None and macro_template.id in user_ns:
                transformer = user_ns[macro_template.id]
        if transformer is None or not callable(transformer):
            return TemplateDynamicMacro(
                macro_template, ordered_arg_names, tracer, is_method
            )
        else:
            return TransformerDynamicMacro(transformer, is_method)


class TemplateDynamicMacro(DynamicMacro):
    def __init__(
        self,
        template: ast.expr,
        ordered_arg_names: list[str],
        tracer: MacroTracer,
        is_method: bool,
    ) -> None:
        super().__init__(is_method)
        self.template = template
        self.ordered_arg_names = ordered_arg_names
        self.tracer = tracer
        # Durable snapshot of the template's augmentation marks (placeholders,
        # pipe operators, nested-macro markers). Latch now, while we are still in
        # the defining cell's expansion and the marks are fresh -- a macro may
        # not be *expanded* until many cells later, by which point process-wide
        # mark churn (``reset_bookkeeping``) could have wiped them. See
        # `_sync_template_marks`.
        self._template_marks: list[tuple[ast.AST, Any]] | None = None
        self._sync_template_marks()

    def _sync_template_marks(self) -> None:
        """Keep the template's augmentation marks alive across expansions.

        ``expand`` (and the re-parse of what it produces) relies on pyccolo's
        process-wide ``augmented_node_ids_by_spec`` still marking the template's
        nodes -- its ``$$`` placeholders, its ``|>`` pipe operators, its nested
        ``map[...]``/``do[...]`` macro markers, etc. But that mapping is global
        and gets wiped wholesale by ``reset_bookkeeping`` (e.g. between cells in
        a test harness) -- after which a macro defined in an earlier cell expands
        with no marks and silently mis-expands. We hold the template node objects
        (valid as long as the template lives) together with the specs that marked
        them, latched the first time the marks are present, and re-establish them
        before every expansion."""
        import pyccolo as pyc

        by_spec = pyc.BaseTracer.augmented_node_ids_by_spec
        if self._template_marks is None:
            captured: list[tuple[ast.AST, Any]] = []
            for node in ast.walk(self.template):
                for spec in pyc.BaseTracer.get_augmentations(id(node)):
                    captured.append((node, spec))
            if captured:
                self._template_marks = captured
        if self._template_marks:
            for node, spec in self._template_marks:
                by_spec[spec].add(id(node))

    @property
    def arg_replacer(self) -> ArgReplacer:
        return self.tracer.arg_replacer

    @property
    def dynamic_macros(self) -> dict[str, DynamicMacro]:
        return self.tracer.dynamic_macros

    @property
    def dynamic_method_macros(self) -> dict[str, DynamicMacro]:
        return self.tracer.dynamic_method_macros

    def expand(self, args: ast.expr) -> ast.expr:
        self._sync_template_marks()
        template_copy: ast.expr = StatementMapper.bookkeeping_propagating_copy(
            self.template
        )
        with self.arg_replacer.macro_visit_context():
            self.arg_replacer(template_copy)
        arg_node_id_to_placeholder_name = dict(
            self.arg_replacer.arg_node_id_to_placeholder_name
        )
        ordered_arg_names = list(self.ordered_arg_names)
        for arg_name in arg_node_id_to_placeholder_name.values():
            if arg_name not in ordered_arg_names:
                ordered_arg_names.append(arg_name)
        expanded_args: list[ast.expr] = [args]
        if len(ordered_arg_names) > 1:
            if not isinstance(args, (ast.List, ast.Tuple)):
                raise ValueError(
                    "Need multiple args but unable to expand singleton subscript arg"
                )
            expanded_args = args.elts
        if len(ordered_arg_names) != len(expanded_args):
            raise ValueError(
                "Wrong number of arguments to macro: expected %d but got %d"
                % (len(ordered_arg_names), len(expanded_args))
            )
        substitutor = DynamicMacroArgSubstitutor(
            arg_node_id_to_placeholder_name,
            ordered_arg_names,
            expanded_args,
            self.dynamic_macros,
            self.dynamic_method_macros,
        )
        return substitutor.visit(template_copy)


class TransformerDynamicMacro(DynamicMacro):
    def __init__(
        self,
        transformer: CallableNodeTransformer | type[CallableNodeTransformer],
        is_method: bool,
    ) -> None:
        super().__init__(is_method)
        self.transformer = transformer

    def expand(self, args: ast.expr) -> ast.expr:
        transformer = (
            self.transformer()
            if isinstance(self.transformer, type)
            else self.transformer
        )
        return transformer(args)
