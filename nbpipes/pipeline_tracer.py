"""
Implementation of a pipeline operators using Pyccolo -- including a purely
functional `|>` operator, an assigning operator `|>>`, placeholders, and more.
"""

from __future__ import annotations

import ast
import builtins
import functools
from types import FrameType
from typing import Any, Callable, Sequence, cast

import pyccolo as pyc
from pyccolo.examples.optional_chaining import OptionalChainer
from pyccolo.stmt_mapper import StatementMapper
from pyccolo.trace_events import TraceEvent

from nbpipes.placeholders import PlaceholderReplacer, SingletonArgCounterMixin
from nbpipes.traceback_patch import frame_to_node_mapping, patch_find_node_ipython
from nbpipes.utils import allow_pipelines_in_loops_and_calls, get_user_ns, null, peek


def node_is_bitor_op(
    node: ast.AST | None,
    allowlisted_augmentations: set[pyc.AugmentationSpec] | None = None,
) -> bool:
    if not isinstance(node, ast.BinOp) or not isinstance(node.op, ast.BitOr):
        return False
    augs = PipelineTracer.get_augmentations(id(node))
    if allowlisted_augmentations is None:
        return bool(augs)
    else:
        return bool(augs & allowlisted_augmentations)


def parent_is_bitor_op(
    node_or_id: ast.expr | int,
    allowlisted_augmentations: set[pyc.AugmentationSpec] | None = None,
) -> bool:
    node_id = node_or_id if isinstance(node_or_id, int) else id(node_or_id)
    parent = pyc.BaseTracer.containing_ast_by_id.get(node_id)
    return node_is_bitor_op(parent, allowlisted_augmentations=allowlisted_augmentations)


def is_outer_or_allowlisted(node_or_id: ast.AST | int) -> bool:
    node_id = node_or_id if isinstance(node_or_id, int) else id(node_or_id)
    if pyc.is_outer_stmt(node_id):
        return True
    containing_stmt = pyc.BaseTracer.containing_stmt_by_id.get(node_id)
    parent_stmt = pyc.BaseTracer.parent_stmt_by_id.get(
        node_id if containing_stmt is None else id(containing_stmt)
    )
    while parent_stmt is not None:
        if isinstance(parent_stmt, ast.With):
            context_expr = parent_stmt.items[0].context_expr
            if (
                isinstance(context_expr, ast.Call)
                and isinstance(context_expr.func, ast.Name)
                and context_expr.func.id == allow_pipelines_in_loops_and_calls.__name__
            ):
                return True
        elif isinstance(parent_stmt, (ast.AsyncFunctionDef, ast.FunctionDef)):
            for deco in parent_stmt.decorator_list:
                if isinstance(deco, ast.Name):
                    actual_deco = deco
                elif isinstance(deco, ast.Call) and isinstance(deco.func, ast.Name):
                    actual_deco = deco.func
                else:
                    continue
                if actual_deco.id == allow_pipelines_in_loops_and_calls.__name__:
                    return True
        parent_stmt = pyc.BaseTracer.parent_stmt_by_id.get(id(parent_stmt))
    return False


def is_partial_call(node: ast.Call) -> bool:
    return PipelineTracer.partial_call_spec in PipelineTracer.get_augmentations(
        id(node)
    )


def partial_call_currier(func: Callable[..., Any]) -> Callable[..., Any]:
    def make_curried_caller(*curried_args, **curried_kwargs):
        @functools.wraps(func)
        def wrapped_with_curried(*args, **kwargs):
            return func(*curried_args, *args, **curried_kwargs, **kwargs)

        return wrapped_with_curried

    return make_curried_caller


_skip_binop_args_lambda = lambda: None  # noqa: E731


class PipelineTracer(pyc.BaseTracer):

    allow_reentrant_events = True
    global_guards_enabled = False

    pipeline_dict_op_spec = pyc.AugmentationSpec(
        aug_type=pyc.AugmentationType.binop, token="**|>", replacement="|"
    )

    pipeline_tuple_op_spec = pyc.AugmentationSpec(
        aug_type=pyc.AugmentationType.binop, token="*|>", replacement="|"
    )

    pipeline_op_assign_spec = pyc.AugmentationSpec(
        aug_type=pyc.AugmentationType.binop, token="|>>", replacement="|"
    )

    pipeline_op_spec = pyc.AugmentationSpec(
        aug_type=pyc.AugmentationType.binop, token="|>", replacement="|"
    )

    value_first_left_partial_apply_dict_op_spec = pyc.AugmentationSpec(
        aug_type=pyc.AugmentationType.binop, token="**$>", replacement="|"
    )

    value_first_left_partial_apply_tuple_op_spec = pyc.AugmentationSpec(
        aug_type=pyc.AugmentationType.binop, token="*$>", replacement="|"
    )

    value_first_left_partial_apply_op_spec = pyc.AugmentationSpec(
        aug_type=pyc.AugmentationType.binop, token="$>", replacement="|"
    )

    function_first_left_partial_apply_dict_op_spec = pyc.AugmentationSpec(
        aug_type=pyc.AugmentationType.binop, token="<$**", replacement="|"
    )

    function_first_left_partial_apply_tuple_op_spec = pyc.AugmentationSpec(
        aug_type=pyc.AugmentationType.binop, token="<$*", replacement="|"
    )

    function_first_left_partial_apply_op_spec = pyc.AugmentationSpec(
        aug_type=pyc.AugmentationType.binop, token="<$", replacement="|"
    )

    apply_dict_op_spec = pyc.AugmentationSpec(
        aug_type=pyc.AugmentationType.binop, token="<|**", replacement="|"
    )

    apply_tuple_op_spec = pyc.AugmentationSpec(
        aug_type=pyc.AugmentationType.binop, token="<|*", replacement="|"
    )

    apply_op_spec = pyc.AugmentationSpec(
        aug_type=pyc.AugmentationType.binop, token="<|", replacement="|"
    )

    nullpipe_dict_op_spec = pyc.AugmentationSpec(
        aug_type=pyc.AugmentationType.binop, token="**?>", replacement="|"
    )

    nullpipe_tuple_op_spec = pyc.AugmentationSpec(
        aug_type=pyc.AugmentationType.binop, token="*?>", replacement="|"
    )

    nullpipe_op_spec = pyc.AugmentationSpec(
        aug_type=pyc.AugmentationType.binop, token="?>", replacement="|"
    )

    null_apply_dict_op_spec = pyc.AugmentationSpec(
        aug_type=pyc.AugmentationType.binop, token="<?**", replacement="|"
    )

    null_apply_tuple_op_spec = pyc.AugmentationSpec(
        aug_type=pyc.AugmentationType.binop, token="<?*", replacement="|"
    )

    null_apply_op_spec = pyc.AugmentationSpec(
        aug_type=pyc.AugmentationType.binop, token="<?", replacement="|"
    )

    left_compose_dict_op_spec = pyc.AugmentationSpec(
        aug_type=pyc.AugmentationType.binop, token="**.>", replacement="|"
    )

    left_compose_tuple_op_spec = pyc.AugmentationSpec(
        aug_type=pyc.AugmentationType.binop, token="*.>", replacement="|"
    )

    left_compose_op_spec = pyc.AugmentationSpec(
        aug_type=pyc.AugmentationType.binop, token=".>", replacement="|"
    )

    alt_compose_dict_op_spec = pyc.AugmentationSpec(
        aug_type=pyc.AugmentationType.binop, token="<.**", replacement="|"
    )

    alt_compose_tuple_op_spec = pyc.AugmentationSpec(
        aug_type=pyc.AugmentationType.binop, token="<.*", replacement="|"
    )

    alt_compose_op_spec = pyc.AugmentationSpec(
        aug_type=pyc.AugmentationType.binop, token="<.", replacement="|"
    )

    compose_dict_op_spec = pyc.AugmentationSpec(
        aug_type=pyc.AugmentationType.binop, token=".** ", replacement="| "
    )

    compose_tuple_op_spec = pyc.AugmentationSpec(
        aug_type=pyc.AugmentationType.binop, token=".* ", replacement="| "
    )

    compose_op_spec = pyc.AugmentationSpec(
        aug_type=pyc.AugmentationType.binop, token=". ", replacement="| "
    )

    partial_call_spec = pyc.AugmentationSpec(
        aug_type=pyc.AugmentationType.call, token="$(", replacement="("
    )

    arg_placeholder_spec = pyc.AugmentationSpec(
        aug_type=pyc.AugmentationType.dot_prefix,
        token="$",
        replacement="_",
    )

    placeholder_replacer = PlaceholderReplacer(arg_placeholder_spec)

    extra_builtins = [allow_pipelines_in_loops_and_calls, null, peek]

    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.binop_arg_nodes_to_skip: set[int] = set()
        self.binop_nodes_to_eval: set[int] = set()
        self.lexical_chain_stack: pyc.TraceStack = self.make_stack()
        with self.register_additional_ast_bookkeeping():
            self.placeholder_arg_position_cache: dict[int, list[str]] = {}
        self.exc_to_propagate: Exception | None = None
        with self.lexical_chain_stack.register_stack_state():
            self.cur_chain_placeholder_lambda: Callable[..., Any] | None = None
        patch_find_node_ipython()
        user_ns = get_user_ns()
        for extra_builtin in self.extra_builtins:
            extra_builtin_name = extra_builtin.__name__
            if hasattr(builtins, extra_builtin_name):
                continue
            setattr(builtins, extra_builtin_name, extra_builtin)
            if user_ns is not None:
                user_ns[extra_builtin_name] = extra_builtin

    @pyc.register_handler(pyc.before_call, when=is_partial_call, reentrant=True)
    def curry_partial_calls(self, ret, node: ast.Call, *_, **__):
        if self.partial_call_spec in self.get_augmentations(id(node)):
            return partial_call_currier(ret)
        else:
            return ret

    @pyc.register_handler(pyc.before_load_complex_symbol, when=is_outer_or_allowlisted)
    def handle_chain_placeholder_rewrites(
        self, ret, node: ast.expr, frame: FrameType, *_, **__
    ):
        from nbpipes.macro_tracer import MacroTracer

        with self.lexical_chain_stack.push():
            self.cur_chain_placeholder_lambda = None
        if not self.placeholder_replacer.search(
            node, allow_top_level=True, check_all_calls=False
        ):
            return ret
        elif (
            MacroTracer.initialized()
            and id(node) in MacroTracer.instance().placeholder_inference_skip_nodes
        ):
            # TODO: this is kinda hacky. Normally we would skip this handler because the
            #   placeholder replacer would short circuit when it sees a macro
            #   boundary, but in this case, we're already past the macro boundary.
            #   Ideally we would have a more robust way to tell that we're already
            #   inside a macro instead of relying on this allowlist mechanism.
            return ret
        __hide_pyccolo_frame__ = True
        frame_to_node_mapping[frame.f_code.co_filename, frame.f_lineno] = node
        node_copy = StatementMapper.bookkeeping_propagating_copy(node)
        assert isinstance(node_copy, ast.expr)
        orig_ctr = self.placeholder_replacer.arg_ctr
        lambda_body_parent_call = None
        lambda_body = node_copy
        while (
            isinstance(lambda_body, ast.Call)
            and isinstance(lambda_body.func, ast.Call)
            and not self.placeholder_replacer.search(
                cast(Sequence[ast.AST], lambda_body.args + lambda_body.keywords),
                allow_top_level=True,
                check_all_calls=False,
            )
        ):
            lambda_body_parent_call = lambda_body
            lambda_body = lambda_body.func
        placeholder_names = self.placeholder_replacer.rewrite(
            lambda_body, allow_top_level=True, check_all_calls=False
        )
        ast_lambda = SingletonArgCounterMixin.create_placeholder_lambda(
            placeholder_names, orig_ctr, lambda_body, frame.f_globals
        )
        ast_lambda.body = lambda_body
        if lambda_body_parent_call is None:
            node_to_eval: ast.expr = ast_lambda
        else:
            lambda_body_parent_call.func = ast_lambda
            node_to_eval = node_copy
        self.cur_chain_placeholder_lambda = lambda: __hide_pyccolo_frame__ and pyc.eval(
            node_to_eval, frame.f_globals, frame.f_locals
        )
        return lambda: OptionalChainer.resolves_to_none_eventually

    @pyc.register_raw_handler(pyc.before_argument, when=is_outer_or_allowlisted)
    def handle_before_arg(self, ret: object, *_, **__):
        if self.cur_chain_placeholder_lambda:
            return lambda: None
        else:
            return ret

    @pyc.register_raw_handler(
        pyc.after_load_complex_symbol, when=is_outer_or_allowlisted
    )
    def handle_after_placeholder_chain(self, ret, *_, **__):
        __hide_pyccolo_frame__ = True
        override_ret = self.cur_chain_placeholder_lambda
        self.lexical_chain_stack.pop()
        if override_ret is None:
            return ret
        try:
            return __hide_pyccolo_frame__ and override_ret()
        except Exception as e:
            self.exc_to_propagate = e
            raise e from None

    def should_propagate_handler_exception(
        self, _evt: TraceEvent, exc: Exception
    ) -> bool:
        if exc is self.exc_to_propagate:
            self.exc_to_propagate = None
            return True
        return False

    @pyc.register_raw_handler(
        (pyc.before_left_binop_arg, pyc.before_right_binop_arg),
        when=lambda node: parent_is_bitor_op(node) and is_outer_or_allowlisted(node),
    )
    def maybe_skip_binop_arg(self, ret: object, node_id: int, *_, **__):
        if node_id in self.binop_arg_nodes_to_skip:
            self.binop_arg_nodes_to_skip.remove(node_id)
            return _skip_binop_args_lambda
        else:
            return ret

    def reorder_placeholder_names_for_prior_positions(
        self, node: ast.expr, placeholder_names: list[str]
    ) -> list[str]:
        prev_placeholders = self.placeholder_arg_position_cache.get(id(node))
        if prev_placeholders is None:
            return placeholder_names
        index_by_name = {
            name: (
                prev_placeholders.index(name)
                if name in prev_placeholders
                else float("inf")
            )
            for name in placeholder_names
        }
        return sorted(placeholder_names, key=lambda name: index_by_name[name])

    def transform_pipeline_placeholders(
        self,
        node: ast.expr,
        parent: ast.BinOp,
        frame_globals: dict[str, Any],
        allow_top_level: bool,
        full_node: ast.expr | None = None,
        associate_lhs: bool = False,
    ) -> ast.Lambda:
        orig_ctr = self.placeholder_replacer.arg_ctr
        placeholder_names = self.placeholder_replacer.rewrite(
            node, allow_top_level=allow_top_level, check_all_calls=True
        )
        if associate_lhs:
            node_to_associate = node
        else:
            node_to_associate = parent
        placeholder_names_to_associate = [
            name for name in placeholder_names if not name[1].isdigit()
        ]
        if placeholder_names_to_associate:
            self.placeholder_arg_position_cache[id(node_to_associate)] = [
                name for name in placeholder_names if not name[1].isdigit()
            ]
        if not associate_lhs:
            placeholder_names = self.reorder_placeholder_names_for_prior_positions(
                parent.left, placeholder_names
            )
        return SingletonArgCounterMixin.create_placeholder_lambda(
            placeholder_names, orig_ctr, full_node or node, frame_globals
        )

    @pyc.register_handler(
        pyc.before_right_binop_arg,
        when=lambda node: parent_is_bitor_op(node) and is_outer_or_allowlisted(node),
    )
    def transform_pipeline_rhs_placeholders(
        self, ret: object, node: ast.expr, frame: FrameType, *_, **__
    ):
        if ret is _skip_binop_args_lambda:
            return ret
        parent: ast.BinOp = self.containing_ast_by_id.get(id(node))  # type: ignore[assignment]
        if not self.get_augmentations(id(parent)):
            return ret
        __hide_pyccolo_frame__ = True
        frame_to_node_mapping[frame.f_code.co_filename, frame.f_lineno] = node
        allow_top_level = not isinstance(node, ast.BinOp) or not self.get_augmentations(
            id(node)
        )
        if not self.placeholder_replacer.search(
            node, allow_top_level=allow_top_level, check_all_calls=True
        ):
            return ret
        transformed = StatementMapper.bookkeeping_propagating_copy(node)
        ast_lambda = self.transform_pipeline_placeholders(
            transformed, parent, frame.f_globals, allow_top_level=allow_top_level
        )
        ast_lambda.body = transformed
        evaluated_lambda = pyc.eval(ast_lambda, frame.f_globals, frame.f_locals)
        return lambda: __hide_pyccolo_frame__ and evaluated_lambda

    @classmethod
    def search_left_descendant_placeholder(cls, node: ast.BinOp) -> int:
        num_traversals = 0
        parent: ast.BinOp
        while True:
            if not (
                cls.get_augmentations(id(node))
                & {
                    cls.pipeline_op_spec,
                    cls.pipeline_tuple_op_spec,
                    cls.pipeline_dict_op_spec,
                    cls.pipeline_op_assign_spec,
                    cls.nullpipe_op_spec,
                    cls.nullpipe_tuple_op_spec,
                    cls.nullpipe_dict_op_spec,
                    cls.value_first_left_partial_apply_op_spec,
                    cls.value_first_left_partial_apply_tuple_op_spec,
                    cls.value_first_left_partial_apply_dict_op_spec,
                    cls.left_compose_op_spec,
                    cls.left_compose_tuple_op_spec,
                    cls.left_compose_dict_op_spec,
                }
            ):
                return -1
            parent = node
            node = node.left  # type: ignore[assignment]
            num_traversals += 1
            if not node_is_bitor_op(node):
                break
        if node_is_bitor_op(
            parent,
            allowlisted_augmentations={
                cls.left_compose_op_spec,
                cls.left_compose_tuple_op_spec,
                cls.left_compose_dict_op_spec,
            },
        ):
            # don't create lambdas if the leftmost pipeline op is a composition operator
            return -1
        if cls.placeholder_replacer.search(
            node, allow_top_level=False, check_all_calls=True
        ):
            return num_traversals
        else:
            return -1

    @pyc.register_handler(
        pyc.before_binop,
        when=lambda node: node_is_bitor_op(node) and is_outer_or_allowlisted(node),
    )
    def transform_pipeline_lhs_placeholders(
        self, ret: object, node: ast.BinOp, frame: FrameType, *_, **__
    ):
        num_left_traversals_to_lhs_placeholder_node = (
            self.search_left_descendant_placeholder(node)
        )
        if num_left_traversals_to_lhs_placeholder_node < 0:
            return ret
        __hide_pyccolo_frame__ = True
        frame_to_node_mapping[frame.f_code.co_filename, frame.f_lineno] = node
        self.binop_arg_nodes_to_skip.add(id(node.left))
        self.binop_arg_nodes_to_skip.add(id(node.right))
        self.binop_nodes_to_eval.add(id(node))
        left_arg = transformed = StatementMapper.bookkeeping_propagating_copy(node)
        for _i in range(num_left_traversals_to_lhs_placeholder_node):
            left_arg = left_arg.left  # type: ignore[assignment]
        ast_lambda = self.transform_pipeline_placeholders(
            left_arg,
            node,
            frame.f_globals,
            allow_top_level=False,
            full_node=transformed,
            associate_lhs=True,
        )
        ast_lambda.body = transformed
        evaluated_lambda = pyc.eval(ast_lambda, frame.f_globals, frame.f_locals)
        return lambda *_, **__: __hide_pyccolo_frame__ and evaluated_lambda

    pipeline_null = object()

    @pyc.register_handler(
        pyc.before_binop,
        when=lambda node: node_is_bitor_op(node) and is_outer_or_allowlisted(node),
    )
    def transform_pipeline_apply_ops(
        self, ret: object, node: ast.BinOp, frame: FrameType, *_, **__
    ):
        if id(node) in self.binop_nodes_to_eval:
            self.binop_nodes_to_eval.remove(id(node))
            return ret
        __hide_pyccolo_frame__ = True
        pipeline_null = self.pipeline_null
        frame_to_node_mapping[frame.f_code.co_filename, frame.f_lineno] = node.left
        this_node_augmentations = self.get_augmentations(id(node))
        if self.pipeline_op_spec in this_node_augmentations:
            return lambda x, y: (
                __hide_pyccolo_frame__ and pipeline_null if x is pipeline_null else y(x)
            )
        elif self.pipeline_tuple_op_spec in this_node_augmentations:
            return lambda x, y: (
                __hide_pyccolo_frame__ and pipeline_null
                if x is pipeline_null
                else y(*x)
            )
        elif self.pipeline_dict_op_spec in this_node_augmentations:
            return lambda x, y: (
                __hide_pyccolo_frame__ and pipeline_null
                if x is pipeline_null
                else y(**x)
            )
        elif self.pipeline_op_assign_spec in this_node_augmentations:
            rhs: ast.Name = node.right  # type: ignore
            if not isinstance(rhs, ast.Name):
                raise ValueError(
                    "unable to assign to RHS of type %s" % type(node.right)
                )
            # eagerly assign it so that we don't get a name error
            frame.f_globals[rhs.id] = None

            def assign_globals(val):
                frame.f_globals[rhs.id] = val
                return __hide_pyccolo_frame__ and val

            return lambda x, y: (
                __hide_pyccolo_frame__ and pipeline_null
                if x is pipeline_null
                else assign_globals(x)
            )
        elif self.nullpipe_op_spec in this_node_augmentations:
            return lambda x, y: (
                __hide_pyccolo_frame__ and pipeline_null
                if x in (None, pipeline_null)
                else y(x)
            )
        elif self.nullpipe_tuple_op_spec in this_node_augmentations:
            return lambda x, y: (
                __hide_pyccolo_frame__ and pipeline_null
                if x in (None, pipeline_null)
                else y(*x)
            )
        elif self.nullpipe_dict_op_spec in this_node_augmentations:
            return lambda x, y: (
                __hide_pyccolo_frame__ and pipeline_null
                if x in (None, pipeline_null)
                else y(**x)
            )
        elif self.value_first_left_partial_apply_op_spec in this_node_augmentations:
            return lambda x, y: (
                __hide_pyccolo_frame__ and pipeline_null
                if x is pipeline_null
                else (
                    lambda *args, **kwargs: __hide_pyccolo_frame__
                    and y(x, *args, **kwargs)
                )
            )
        elif (
            self.value_first_left_partial_apply_tuple_op_spec in this_node_augmentations
        ):
            return lambda x, y: (
                __hide_pyccolo_frame__ and pipeline_null
                if x is pipeline_null
                else (
                    lambda *args, **kwargs: __hide_pyccolo_frame__
                    and y(*x, *args, **kwargs)
                )
            )
        elif (
            self.value_first_left_partial_apply_dict_op_spec in this_node_augmentations
        ):
            return lambda x, y: (
                __hide_pyccolo_frame__ and pipeline_null
                if x is pipeline_null
                else (
                    lambda *args, **kwargs: __hide_pyccolo_frame__
                    and y(*args, **x, **kwargs)
                )
            )
        elif self.function_first_left_partial_apply_op_spec in this_node_augmentations:
            return lambda x, y: (
                __hide_pyccolo_frame__ and pipeline_null
                if y is pipeline_null
                else (
                    lambda *args, **kwargs: (
                        __hide_pyccolo_frame__ and x(y, *args, **kwargs)
                    )
                )
            )
        elif (
            self.function_first_left_partial_apply_tuple_op_spec
            in this_node_augmentations
        ):
            return lambda x, y: (
                __hide_pyccolo_frame__ and pipeline_null
                if y is pipeline_null
                else (
                    lambda *args, **kwargs: __hide_pyccolo_frame__
                    and x(*y, *args, **kwargs)
                )
            )
        elif (
            self.function_first_left_partial_apply_dict_op_spec
            in this_node_augmentations
        ):
            return lambda x, y: (
                __hide_pyccolo_frame__ and pipeline_null
                if y is pipeline_null
                else (
                    lambda *args, **kwargs: __hide_pyccolo_frame__
                    and x(*args, **y, **kwargs)
                )
            )
        elif self.apply_op_spec in this_node_augmentations:
            return lambda x, y: (
                __hide_pyccolo_frame__ and pipeline_null if y is pipeline_null else x(y)
            )
        elif self.apply_tuple_op_spec in this_node_augmentations:
            return lambda x, y: (
                __hide_pyccolo_frame__ and pipeline_null
                if y is pipeline_null
                else x(*y)
            )
        elif self.apply_dict_op_spec in this_node_augmentations:
            return lambda x, y: (
                __hide_pyccolo_frame__ and pipeline_null
                if y is pipeline_null
                else x(**y)
            )
        elif self.null_apply_op_spec in this_node_augmentations:
            return lambda x, y: (
                __hide_pyccolo_frame__ and pipeline_null
                if y in (None, pipeline_null)
                else x(y)
            )
        elif self.null_apply_tuple_op_spec in this_node_augmentations:
            return lambda x, y: (
                __hide_pyccolo_frame__ and pipeline_null
                if y in (None, pipeline_null)
                else x(*y)
            )
        elif self.null_apply_dict_op_spec in this_node_augmentations:
            return lambda x, y: (
                __hide_pyccolo_frame__ and pipeline_null
                if y in (None, pipeline_null)
                else x(**y)
            )
        else:
            return ret

    @pyc.register_handler(
        pyc.after_binop,
        when=lambda node: node_is_bitor_op(node)
        and not parent_is_bitor_op(node)
        and is_outer_or_allowlisted(node),
    )
    def coalesce_pipeline_null(self, ret, *_, **__):
        if ret is self.pipeline_null:
            return pyc.Null
        else:
            return ret

    @pyc.register_handler(
        pyc.before_binop,
        when=lambda node: node_is_bitor_op(node) and is_outer_or_allowlisted(node),
    )
    def transform_pipeline_compose_ops(
        self, ret: object, node: ast.BinOp, frame: FrameType, *_, **__
    ):
        if id(node) in self.binop_nodes_to_eval:
            self.binop_nodes_to_eval.remove(id(node))
            return ret
        __hide_pyccolo_frame__ = True
        frame_to_node_mapping[frame.f_code.co_filename, frame.f_lineno] = node.left
        this_node_augmentations = self.get_augmentations(id(node))
        if {self.compose_op_spec, self.alt_compose_op_spec} & this_node_augmentations:

            def __pipeline_compose(f, g):
                def __composed(*args, **kwargs):
                    return __hide_pyccolo_frame__ and f(g(*args, **kwargs))

                return __composed

            return __pipeline_compose
        elif {
            self.compose_tuple_op_spec,
            self.alt_compose_tuple_op_spec,
        } & this_node_augmentations:

            def __pipeline_tuple_compose(f, g):
                def __tuple_composed(*args, **kwargs):
                    return f(*g(*args, **kwargs))

                return __tuple_composed

            return __pipeline_tuple_compose
        elif {
            self.compose_dict_op_spec,
            self.alt_compose_dict_op_spec,
        } & this_node_augmentations:

            def __pipeline_dict_compose(f, g):
                def __tuple_composed(*args, **kwargs):
                    return f(**g(*args, **kwargs))

                return __tuple_composed

            return __pipeline_dict_compose
        elif self.left_compose_op_spec in this_node_augmentations:

            def __left_pipeline_compose(f, g):
                def __composed(*args, **kwargs):
                    return __hide_pyccolo_frame__ and g(f(*args, **kwargs))

                return __composed

            return __left_pipeline_compose
        elif self.left_compose_tuple_op_spec in this_node_augmentations:

            def __left_pipeline_tuple_compose(f, g):
                def __tuple_composed(*args, **kwargs):
                    return g(*f(*args, **kwargs))

                return __tuple_composed

            return __left_pipeline_tuple_compose
        elif self.left_compose_dict_op_spec in this_node_augmentations:

            def __left_pipeline_dict_compose(f, g):
                def __tuple_composed(*args, **kwargs):
                    return g(**f(*args, **kwargs))

                return __tuple_composed

            return __left_pipeline_dict_compose
        else:
            return ret
