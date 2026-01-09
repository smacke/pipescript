from __future__ import annotations

import ast
from typing import Any, Callable

import pyccolo as pyc

from pipescript.utils import has_augmentations


def parent_is_or_boolop(node_id: int) -> bool:
    parent = pyc.BaseTracer.containing_ast_by_id.get(node_id)
    return isinstance(parent, ast.BoolOp) and isinstance(parent.op, ast.Or)


def should_instrument_for_spec(
    spec: pyc.AugmentationSpec | set[pyc.AugmentationSpec] | str,
    attr: str | None = None,
) -> Callable[[ast.AST | int], bool]:
    if isinstance(spec, str):
        spec = getattr(OptionalChainingTracer, spec)
    assert not isinstance(spec, str)
    return lambda node: has_augmentations(getattr(node, attr or "", node), spec)


def should_instrument_for_spec_on_parent(
    spec: pyc.AugmentationSpec | set[pyc.AugmentationSpec] | str,
    attr: str | None = None,
) -> Callable[[ast.AST | int], bool]:
    if isinstance(spec, str):
        spec = getattr(OptionalChainingTracer, spec)
    assert not isinstance(spec, str)

    def should_instrument(node_or_id: ast.AST | int) -> bool:
        node_id = node_or_id if isinstance(node_or_id, int) else id(node_or_id)
        parent = pyc.BaseTracer.containing_ast_by_id.get(node_id)
        if parent is None:
            return False
        return has_augmentations(getattr(parent, attr or "", parent), spec)

    return should_instrument


class NullishInstrumentationChainChecker(ast.NodeVisitor):
    def __init__(self) -> None:
        self._contains_nullish_instrumentation = False

    def __call__(self, node_or_id: ast.AST | int) -> bool:
        node = (
            node_or_id
            if isinstance(node_or_id, ast.AST)
            else pyc.BaseTracer.ast_node_by_id.get(node_or_id)
        )
        if node is None:
            return False
        self._contains_nullish_instrumentation = False
        self.visit(node)
        return self._contains_nullish_instrumentation

    def check_parent(self, node_or_id: ast.AST | int) -> bool:
        node_id = node_or_id if isinstance(node_or_id, int) else id(node_or_id)
        parent = pyc.BaseTracer.containing_ast_by_id.get(node_id)
        if parent is None:
            return False
        else:
            return self(parent)

    def visit_Attribute(self, node: ast.Attribute) -> None:
        if has_augmentations(
            node,
            {
                OptionalChainingTracer.attr_optional_chaining_spec,
                OptionalChainingTracer.attr_permissive_chaining_spec,
            },
        ):
            self._contains_nullish_instrumentation = True
            return
        self.visit(node.value)

    def visit_Subscript(self, node: ast.Subscript) -> None:
        if has_augmentations(
            node,
            {
                OptionalChainingTracer.subscript_optional_chaining_spec,
                OptionalChainingTracer.subscript_permissive_chaining_spec,
                OptionalChainingTracer.subscript_permissive_and_optional_chaining_spec,
            },
        ):
            self._contains_nullish_instrumentation = True
            return
        self.visit(node.value)

    def visit_Call(self, node: ast.Call) -> None:
        if has_augmentations(node, OptionalChainingTracer.call_optional_chaining_spec):
            self._contains_nullish_instrumentation = True
            return
        self.visit(node.func)


class OptionalChainingTracer(pyc.BaseTracer):
    global_guards_enabled = False
    allow_reentrant_events = True

    class ResolvesToNone:
        def __init__(self, eventually: bool) -> None:
            self.__eventually = eventually

        def __getattr__(self, _item: str):
            if self.__eventually:
                return self
            else:
                return None

        def __getitem__(self, _item: str):
            if self.__eventually:
                return self
            else:
                return None

        def __call__(self, *_, **__):
            return self

    resolves_to_none_eventually = ResolvesToNone(eventually=True)
    resolves_to_none_immediately = ResolvesToNone(eventually=False)

    call_optional_chaining_spec = pyc.AugmentationSpec(
        aug_type=pyc.AugmentationType.call, token="?.(", replacement="("
    )

    subscript_permissive_and_optional_chaining_spec = pyc.AugmentationSpec(
        aug_type=pyc.AugmentationType.dot_suffix, token="?.?[", replacement="["
    )

    subscript_optional_chaining_spec = pyc.AugmentationSpec(
        aug_type=pyc.AugmentationType.dot_suffix, token="?.[", replacement="["
    )

    subscript_permissive_chaining_spec = pyc.AugmentationSpec(
        aug_type=pyc.AugmentationType.dot_suffix, token=".?[", replacement="["
    )

    attr_optional_chaining_spec = pyc.AugmentationSpec(
        aug_type=pyc.AugmentationType.dot_suffix, token="?.", replacement="."
    )

    attr_permissive_chaining_spec = pyc.AugmentationSpec(
        aug_type=pyc.AugmentationType.dot_suffix, token=".?", replacement="."
    )

    nullish_coalescing_spec = pyc.AugmentationSpec(
        aug_type=pyc.AugmentationType.boolop, token="??", replacement=" or "
    )

    chain_checker = NullishInstrumentationChainChecker()

    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self._saved_ret_expr = None
        self.lexical_call_stack: pyc.TraceStack = self.make_stack()
        with self.lexical_call_stack.register_stack_state():
            # TODO: pop this the right number of times if an exception occurs
            self.cur_call_is_none_resolver: bool = False
        self.lexical_nullish_stack: pyc.TraceStack = self.make_stack()
        with self.lexical_nullish_stack.register_stack_state():
            self.cur_boolop_has_nullish_coalescer = False
            self.coalesced_value: Any | None = None

    def clear_stacks(self):
        # will be registered as a post_run_cell event
        while len(self.lexical_nullish_stack) > 0:
            self.lexical_nullish_stack.pop()

    @pyc.register_handler(
        pyc.before_subscript_load,
        when=should_instrument_for_spec(
            {
                subscript_optional_chaining_spec,
                subscript_permissive_chaining_spec,
                subscript_permissive_and_optional_chaining_spec,
            }
        ),
        reentrant=True,
    )
    def handle_before_subscript_with_optional_chaining_spec(
        self, obj, node: ast.Subscript, *_, attr_or_subscript: object, **__
    ):
        __hide_pyccolo_frame__ = True  # noqa: F841
        this_node_augmentations = self.get_augmentations(id(node))
        if (
            this_node_augmentations
            & {
                self.subscript_optional_chaining_spec,
                self.subscript_permissive_and_optional_chaining_spec,
            }
            and obj is None
        ):
            return self.resolves_to_none_eventually
        elif (
            this_node_augmentations
            & {
                self.subscript_permissive_chaining_spec,
                self.subscript_permissive_and_optional_chaining_spec,
            }
            and hasattr(obj, "__contains__")
            and attr_or_subscript not in obj
        ):
            return self.resolves_to_none_immediately
        else:
            return obj

    @pyc.register_handler(
        pyc.before_attribute_load,
        when=should_instrument_for_spec(
            {attr_optional_chaining_spec, attr_permissive_chaining_spec}
        ),
        reentrant=True,
    )
    def handle_before_attr_with_optional_chaining_spec(
        self, obj, node: ast.Attribute, *_, **__
    ):
        __hide_pyccolo_frame__ = True  # noqa: F841
        this_node_augmentations = self.get_augmentations(id(node))
        if self.attr_optional_chaining_spec in this_node_augmentations and obj is None:
            return self.resolves_to_none_eventually
        elif (
            self.attr_permissive_chaining_spec in this_node_augmentations
            and not hasattr(obj, node.attr)
        ):
            return self.resolves_to_none_immediately
        else:
            # this should not happen
            return obj

    @pyc.register_handler(
        pyc.before_call,
        when=chain_checker,
        reentrant=True,
    )
    def handle_before_call_with_call_optional_chaining_spec(self, func, *_, **__):
        if func is None:
            func = self.resolves_to_none_eventually
        with self.lexical_call_stack.push():
            self.cur_call_is_none_resolver = func is self.resolves_to_none_eventually
        return func

    @pyc.register_raw_handler(
        pyc.before_argument,
        when=chain_checker.check_parent,
        reentrant=True,
    )
    def handle_before_arg_of_func_with_call_optional_chaining_spec(
        self, arg_lambda, *_, **__
    ):
        if self.cur_call_is_none_resolver:
            return lambda: None
        else:
            return arg_lambda

    @pyc.register_raw_handler(
        pyc.after_call,
        when=chain_checker,
        reentrant=True,
    )
    def handle_after_call_with_call_optional_chaining_spec(self, *_, **__):
        self.lexical_call_stack.pop()

    @pyc.register_raw_handler(
        pyc.after_load_complex_symbol, when=chain_checker, reentrant=True
    )
    def handle_after_chain_with_optional_specs(self, ret, *_, **__):
        if isinstance(ret, self.ResolvesToNone):
            return pyc.Null
        else:
            return ret

    @pyc.register_handler(
        pyc.before_boolop,
        when=lambda node: isinstance(node.op, ast.Or)
        and should_instrument_for_spec("nullish_coalescing_spec", attr="values")(node),
        reentrant=True,
    )
    def before_or_boolop_nullish_coalesce(self, ret, node: ast.BoolOp, *_, **__):
        with self.lexical_nullish_stack.push():
            self.cur_boolop_has_nullish_coalescer = any(
                self.nullish_coalescing_spec in self.get_augmentations(id(val))
                for val in node.values
            )
            self.coalesced_value = None
        return ret

    @pyc.register_handler(
        pyc.after_boolop,
        when=lambda node: isinstance(node.op, ast.Or)
        and should_instrument_for_spec("nullish_coalescing_spec", attr="values")(node),
        reentrant=True,
    )
    def after_or_boolop_nullish_coalesce(self, *_, **__):
        self.lexical_nullish_stack.pop()

    def _maybe_compute_nullish_coalesced_value(self, ret, node_id: int) -> None:
        __hide_pyccolo_frame__ = True  # noqa: F841
        if self.coalesced_value is not None:
            return
        val = ret()
        if self.nullish_coalescing_spec in self.get_augmentations(node_id):
            self.coalesced_value = None if val is None else val
        else:
            self.coalesced_value = val or None

    @pyc.register_raw_handler(
        pyc.before_boolop_arg,
        when=lambda node_id: parent_is_or_boolop(node_id)
        and should_instrument_for_spec_on_parent(
            "nullish_coalescing_spec", attr="values"
        )(node_id),
        reentrant=True,
    )
    def coalesce_boolop_values(self, ret, node_id: int, *_, is_last: bool, **__):
        __hide_pyccolo_frame__ = True  # noqa: F841
        if self.cur_boolop_has_nullish_coalescer:
            if is_last:
                if self.coalesced_value is None:
                    return ret
                else:
                    return lambda: self.coalesced_value
            else:
                self._maybe_compute_nullish_coalesced_value(ret, node_id)
                return lambda: None
        else:
            return ret
