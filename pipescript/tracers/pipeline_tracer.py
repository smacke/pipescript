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
from pyccolo.stmt_mapper import StatementMapper
from pyccolo.trace_events import TraceEvent

import pipescript.api.utils
from pipescript.analysis.placeholders import (
    PlaceholderReplacer,
    SingletonArgCounterMixin,
)
from pipescript.api.macros import fork, parallel
from pipescript.api.utils import (
    _dynamic_lookup,
    collapse,
    lshift,
    null,
    peek,
    pop,
    push,
    replace,
    rshift,
    unnest,
)
from pipescript.constants import pipeline_null
from pipescript.patches.traceback_patch import (
    frame_to_node_mapping,
    patch_find_node_ipython,
)
from pipescript.tracers.optional_chaining_tracer import OptionalChainingTracer
from pipescript.utils import get_user_ns, has_augmentations


def node_is_pipeline_bitor_op(
    node: ast.AST | None,
    allowlisted_augmentations: set[pyc.AugmentationSpec] | None = None,
) -> bool:
    if not isinstance(node, ast.BinOp) or not isinstance(node.op, ast.BitOr):
        return False
    return has_augmentations(node, expected_augs=allowlisted_augmentations)


def parent_is_pipeline_bitor_op(node_or_id: ast.expr | int) -> bool:
    node_id = node_or_id if isinstance(node_or_id, int) else id(node_or_id)
    parent = pyc.BaseTracer.containing_ast_by_id.get(node_id)
    return node_is_pipeline_bitor_op(parent)


# TODO: this is not ideal. we are preventing the pipeline_null -> None coalescing behavior for
#   fork / parallel macros explicitly, which feels error prone. Is there a more robust way?
def parent_is_fork_macro(node_or_id: ast.expr | int) -> bool:
    node_id = node_or_id if isinstance(node_or_id, int) else id(node_or_id)
    parent = pyc.BaseTracer.containing_ast_by_id.get(node_id)
    while isinstance(parent, (ast.Lambda, ast.Tuple)):
        parent = pyc.BaseTracer.containing_ast_by_id.get(id(parent))
    if (
        isinstance(parent, ast.Subscript)
        and isinstance(parent.value, ast.Name)
        and parent.value.id in (fork.__name__, parallel.__name__)
    ):
        return True
    elif (
        isinstance(parent, ast.Call)
        and isinstance(parent.func, ast.Name)
        and parent.func.id in (fork.__name__, parallel.__name__)
    ):
        return True
    else:
        return False


def node_is_function_power_op(node: ast.AST | None) -> bool:
    if not isinstance(node, ast.BinOp) or not isinstance(node.op, ast.Pow):
        return False
    return has_augmentations(node, expected_augs=PipelineTracer.function_power_op_spec)


def is_partial_call(node: ast.Call) -> bool:
    return PipelineTracer.partial_call_spec in PipelineTracer.get_augmentations(
        id(node)
    )


def is_chain_with_placeholders(node_or_id: ast.AST | int) -> bool:
    node_id = node_or_id if isinstance(node_or_id, int) else id(node_or_id)
    node = pyc.BaseTracer.ast_node_by_id.get(node_id)
    if node is None:
        return False
    while True:
        containing_node = pyc.BaseTracer.containing_ast_by_id.get(node_id)
        if node is containing_node or not isinstance(
            containing_node, (ast.Call, ast.Attribute, ast.Subscript)
        ):
            break
        node = containing_node
    return PipelineTracer.placeholder_replacer.search(
        node, allow_top_level=True, check_all_calls=False
    )


def partial_call_currier(func: Callable[..., Any]) -> Callable[..., Any]:
    def make_curried_caller(*curried_args, **curried_kwargs):
        @functools.wraps(func)
        def wrapped_with_curried(*args, **kwargs):
            __hide_pyccolo_frame__ = True  # noqa: F841
            return func(*curried_args, *args, **curried_kwargs, **kwargs)

        return wrapped_with_curried

    return make_curried_caller


_skip_binop_args_lambda = lambda: None  # noqa: E731


class PipelineTracer(pyc.BaseTracer):

    allow_reentrant_events = True
    global_guards_enabled = False
    multiple_threads_allowed = True

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

    forward_compose_dict_op_spec = pyc.AugmentationSpec(
        aug_type=pyc.AugmentationType.binop, token="**.>", replacement="|"
    )

    forward_compose_tuple_op_spec = pyc.AugmentationSpec(
        aug_type=pyc.AugmentationType.binop, token="*.>", replacement="|"
    )

    forward_compose_op_spec = pyc.AugmentationSpec(
        aug_type=pyc.AugmentationType.binop, token=".>", replacement="|"
    )

    compose_dict_op_spec = pyc.AugmentationSpec(
        aug_type=pyc.AugmentationType.binop, token="<.**", replacement="|"
    )

    compose_tuple_op_spec = pyc.AugmentationSpec(
        aug_type=pyc.AugmentationType.binop, token="<.*", replacement="|"
    )

    compose_op_spec = pyc.AugmentationSpec(
        aug_type=pyc.AugmentationType.binop, token="<.", replacement="|"
    )

    function_power_op_spec = pyc.AugmentationSpec(
        aug_type=pyc.AugmentationType.binop, token=".**", replacement="**"
    )

    # just prevents partial call spec from taking effect when it shouldn't
    # TODO: to deal with this properly we need a regex version of partial_call_spec
    #   that matches tokens starting with a valid prefix
    _non_partial_call_op_spec = pyc.AugmentationSpec(
        aug_type=pyc.AugmentationType.dot_prefix, token=" $(", replacement=" ($)("
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

    extra_builtins = [
        _dynamic_lookup,
        collapse,
        lshift,
        null,
        peek,
        pop,
        push,
        replace,
        rshift,
        unnest,
    ]
    assert set(pipescript.api.utils.__all__) <= {eb.__name__ for eb in extra_builtins}

    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.binop_arg_nodes_to_skip: set[int] = set()
        self.binop_nodes_to_eval: set[int] = set()
        self.lexical_chain_stack: pyc.TraceStack = self.make_stack()
        self._overridden_builtins: list[str] = []
        with self.register_additional_ast_bookkeeping():
            self.placeholder_arg_position_cache: dict[int, list[str]] = {}
        self.exc_to_propagate: Exception | None = None
        with self.lexical_chain_stack.register_stack_state():
            # TODO: pop this the right number of times if an exception occurs
            self.cur_chain_placeholder_lambda: Callable[..., Any] | None = None
        patch_find_node_ipython()
        user_ns = get_user_ns()
        for extra_builtin in self.extra_builtins:
            extra_builtin_name = extra_builtin.__name__
            if hasattr(builtins, extra_builtin_name):
                continue
            self._overridden_builtins.append(extra_builtin_name)
            setattr(builtins, extra_builtin_name, extra_builtin)
            if user_ns is not None and extra_builtin_name:
                user_ns.setdefault(extra_builtin_name, extra_builtin)

    def clear_stacks(self):
        # will be registered as a post_run_cell event
        while len(self.lexical_chain_stack) > 0:
            self.lexical_chain_stack.pop()

    def reset(self) -> None:
        for extra_builtin_name in self._overridden_builtins:
            if hasattr(builtins, extra_builtin_name):
                delattr(builtins, extra_builtin_name)
        self._overridden_builtins.clear()
        super().reset()

    @pyc.register_handler(pyc.before_call, when=is_partial_call, reentrant=True)
    def curry_partial_calls(self, ret, *_, **__):
        return partial_call_currier(ret)

    @pyc.register_handler(
        pyc.before_load_complex_symbol, when=is_chain_with_placeholders, reentrant=True
    )
    def handle_chain_placeholder_rewrites(
        self, ret, node: ast.expr, frame: FrameType, *_, **__
    ):
        with self.lexical_chain_stack.push():
            self.cur_chain_placeholder_lambda = None
        if not self.placeholder_replacer.search(
            node, allow_top_level=True, check_all_calls=False
        ):
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
        ast_lambda, _extra_defaults, modified_lambda_body = (
            SingletonArgCounterMixin.create_placeholder_lambda(
                placeholder_names, orig_ctr, lambda_body, frame
            )
        )
        lambda_body = modified_lambda_body or lambda_body
        ast_lambda.body = lambda_body
        if lambda_body_parent_call is None:
            node_to_eval: ast.expr = ast_lambda
        else:
            lambda_body_parent_call.func = ast_lambda
            node_to_eval = node_copy
        self.cur_chain_placeholder_lambda = lambda: __hide_pyccolo_frame__ and pyc.eval(
            node_to_eval, frame.f_globals, frame.f_locals
        )
        return lambda: OptionalChainingTracer.resolves_to_none_eventually

    @pyc.register_raw_handler(
        pyc.before_argument, when=is_chain_with_placeholders, reentrant=True
    )
    def handle_before_arg(self, ret: object, *_, **__):
        if self.cur_chain_placeholder_lambda:
            return lambda: None
        else:
            return ret

    @pyc.register_raw_handler(
        pyc.after_load_complex_symbol,
        when=is_chain_with_placeholders,
        reentrant=True,
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
        when=parent_is_pipeline_bitor_op,
        reentrant=True,
    )
    def maybe_skip_binop_arg(self, ret: object, node_id: int, *_, **__):
        if node_id in self.binop_arg_nodes_to_skip:
            self.binop_arg_nodes_to_skip.remove(node_id)
            ret = _skip_binop_args_lambda
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
        frame: FrameType,
        allow_top_level: bool,
        full_node: ast.expr | None = None,
        associate_lhs: bool = False,
    ) -> tuple[ast.Lambda, ast.expr | None]:
        orig_ctr = self.placeholder_replacer.arg_ctr
        placeholder_names = self.placeholder_replacer.rewrite(
            node, allow_top_level=allow_top_level, check_all_calls=True
        )
        if associate_lhs:
            node_to_associate = node
        else:
            node_to_associate = parent
        placeholder_names_to_associate = [
            name
            for name in placeholder_names
            if len(name) == 1 or not name[1].isdigit()
        ]
        if placeholder_names_to_associate:
            self.placeholder_arg_position_cache[id(node_to_associate)] = (
                placeholder_names_to_associate
            )
        if not associate_lhs:
            placeholder_names = self.reorder_placeholder_names_for_prior_positions(
                parent.left, placeholder_names
            )
        ast_lambda, _extra_defaults, modified_lambda_body = (
            SingletonArgCounterMixin.create_placeholder_lambda(
                placeholder_names, orig_ctr, full_node or node, frame
            )
        )
        return ast_lambda, modified_lambda_body

    @pyc.register_handler(
        pyc.before_right_binop_arg,
        when=parent_is_pipeline_bitor_op,
        reentrant=True,
    )
    def transform_pipeline_rhs_placeholders(
        self, ret: object, node: ast.expr, frame: FrameType, *_, **__
    ):
        if ret is _skip_binop_args_lambda:
            return ret
        parent: ast.BinOp = self.containing_ast_by_id.get(id(node))  # type: ignore[assignment]
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
        ast_lambda, modified_lambda_body = self.transform_pipeline_placeholders(
            transformed, parent, frame, allow_top_level=allow_top_level
        )
        transformed = modified_lambda_body or transformed
        ast_lambda.body = transformed
        evaluated_lambda = pyc.eval(ast_lambda, frame.f_globals, frame.f_locals)
        return lambda: __hide_pyccolo_frame__ and evaluated_lambda

    @classmethod
    def search_left_descendant_placeholder(cls, node: ast.BinOp) -> int:
        num_traversals = 0
        parent: ast.BinOp
        while True:
            if not has_augmentations(
                node,
                {
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
                    cls.forward_compose_op_spec,
                    cls.forward_compose_tuple_op_spec,
                    cls.forward_compose_dict_op_spec,
                },
            ):
                return -1
            parent = node
            node = node.left  # type: ignore[assignment]
            num_traversals += 1
            if not node_is_pipeline_bitor_op(node):
                break
        if node_is_pipeline_bitor_op(
            parent,
            allowlisted_augmentations={
                cls.forward_compose_op_spec,
                cls.forward_compose_tuple_op_spec,
                cls.forward_compose_dict_op_spec,
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
        when=node_is_pipeline_bitor_op,
        reentrant=True,
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
        transformed: ast.expr
        left_arg = transformed = StatementMapper.bookkeeping_propagating_copy(node)
        for _i in range(num_left_traversals_to_lhs_placeholder_node):
            left_arg = left_arg.left  # type: ignore[assignment]
        ast_lambda, modified_lambda_body = self.transform_pipeline_placeholders(
            left_arg,
            node,
            frame,
            allow_top_level=False,
            full_node=transformed,
            associate_lhs=True,
        )
        transformed = modified_lambda_body or transformed
        ast_lambda.body = transformed
        evaluated_lambda = pyc.eval(ast_lambda, frame.f_globals, frame.f_locals)
        return lambda *_, **__: __hide_pyccolo_frame__ and evaluated_lambda

    @pyc.register_handler(
        pyc.before_binop,
        when=node_is_pipeline_bitor_op,
        reentrant=True,
    )
    def transform_pipeline_apply_ops(
        self, ret: object, node: ast.BinOp, frame: FrameType, *_, **__
    ):
        if id(node) in self.binop_nodes_to_eval:
            self.binop_nodes_to_eval.remove(id(node))
            return ret
        __hide_pyccolo_frame__ = True
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
        when=lambda node: node_is_pipeline_bitor_op(node)
        and not parent_is_pipeline_bitor_op(node)
        and not parent_is_fork_macro(node),
        reentrant=True,
    )
    def coalesce_pipeline_null(self, ret, *_, **__):
        if ret is pipeline_null:
            return pyc.Null
        else:
            return ret

    @pyc.register_handler(
        pyc.before_binop,
        when=node_is_pipeline_bitor_op,
        reentrant=True,
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
        if {self.compose_op_spec, self.compose_op_spec} & this_node_augmentations:

            def __pipeline_compose(f, g):
                def __composed(*args, **kwargs):
                    g_result = __hide_pyccolo_frame__ and g(*args, **kwargs)
                    return (
                        __hide_pyccolo_frame__ and pipeline_null
                        if g_result is pipeline_null
                        else f(g_result)
                    )

                return __composed

            return __pipeline_compose
        elif {
            self.compose_tuple_op_spec,
            self.compose_tuple_op_spec,
        } & this_node_augmentations:

            def __pipeline_tuple_compose(f, g):
                def __tuple_composed(*args, **kwargs):
                    g_result = __hide_pyccolo_frame__ and g(*args, **kwargs)
                    return (
                        __hide_pyccolo_frame__ and pipeline_null
                        if g_result is pipeline_null
                        else f(*g_result)
                    )

                return __tuple_composed

            return __pipeline_tuple_compose
        elif {
            self.compose_dict_op_spec,
            self.compose_dict_op_spec,
        } & this_node_augmentations:

            def __pipeline_dict_compose(f, g):
                def __tuple_composed(*args, **kwargs):
                    g_result = __hide_pyccolo_frame__ and g(*args, **kwargs)
                    return (
                        __hide_pyccolo_frame__ and pipeline_null
                        if g_result is pipeline_null
                        else f(**g_result)
                    )

                return __tuple_composed

            return __pipeline_dict_compose
        elif self.forward_compose_op_spec in this_node_augmentations:

            def __left_pipeline_compose(f, g):
                def __composed(*args, **kwargs):
                    f_result = __hide_pyccolo_frame__ and f(*args, **kwargs)
                    return (
                        __hide_pyccolo_frame__ and pipeline_null
                        if f_result is pipeline_null
                        else g(f_result)
                    )

                return __composed

            return __left_pipeline_compose
        elif self.forward_compose_tuple_op_spec in this_node_augmentations:

            def __left_pipeline_tuple_compose(f, g):
                def __tuple_composed(*args, **kwargs):
                    f_result = __hide_pyccolo_frame__ and f(*args, **kwargs)
                    return (
                        __hide_pyccolo_frame__ and pipeline_null
                        if f_result is pipeline_null
                        else g(*f_result)
                    )

                return __tuple_composed

            return __left_pipeline_tuple_compose
        elif self.forward_compose_dict_op_spec in this_node_augmentations:

            def __left_pipeline_dict_compose(f, g):
                def __tuple_composed(*args, **kwargs):
                    f_result = __hide_pyccolo_frame__ and f(*args, **kwargs)
                    return (
                        __hide_pyccolo_frame__ and pipeline_null
                        if f_result is pipeline_null
                        else g(**f_result)
                    )

                return __tuple_composed

            return __left_pipeline_dict_compose
        else:
            return ret

    @pyc.register_handler(
        pyc.before_binop,
        when=node_is_function_power_op,
        reentrant=True,
    )
    def exponentiate_functions(self, ret, *_, **__):
        __hide_pyccolo_frame__ = True

        @functools.cache
        def __power_compose(func, exponent):
            if exponent == 1:
                return func
            f1 = __power_compose(func, exponent // 2)
            f2 = __power_compose(func, (exponent + 1) // 2)

            return lambda v, *args: (
                __hide_pyccolo_frame__ and f1(f2(v))
                if len(args) == 0
                else f1(*f2(v, *args))
            )

        return lambda x, y: (
            __hide_pyccolo_frame__ and __power_compose(x, y)
            if callable(x)
            else ret(x, y)
        )
