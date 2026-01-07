from __future__ import annotations

import ast
import builtins
import itertools
from contextlib import contextmanager
from types import FrameType
from typing import Generator, Sequence, cast

import pyccolo as pyc
import pyccolo.fast as fast

from pipescript.analysis.extract_names import ExtractNames


class FreeVarTransformer(ast.NodeTransformer):
    frame_cache: dict[int, FrameType] = {}

    def __init__(self, freevars: set[str], frame: FrameType) -> None:
        self.freevars = freevars
        self.frame_id = id(frame)
        self.frame_cache[self.frame_id] = frame

    @fast.location_of_arg
    def visit_Name(self, node: ast.Name) -> ast.Name | ast.Call:
        if node.id not in self.freevars:
            return node
        from pipescript.api.utils import _dynamic_lookup

        return fast.Call(
            func=fast.Name(_dynamic_lookup.__name__, ast.Load()),
            args=[fast.Constant(value=node.id), fast.Constant(value=self.frame_id)],
            keywords=[],
        )


class SingletonArgCounterMixin:
    _arg_ctr = 0

    @property
    def arg_ctr(self) -> int:
        return self._arg_ctr

    @arg_ctr.setter
    def arg_ctr(self, new_arg_ctr: int) -> None:
        SingletonArgCounterMixin._arg_ctr = new_arg_ctr

    @classmethod
    def create_placeholder_lambda(
        cls,
        placeholder_names: list[str],
        orig_ctr: int,
        lambda_body: ast.expr,
        frame: FrameType,
        created_starred_arg: bool = False,
    ) -> tuple[ast.Lambda, set[str], ast.expr | None]:
        num_lambda_args = cls._arg_ctr - orig_ctr
        lambda_args = []
        extra_defaults = ExtractNames.extract_names(lambda_body) - set(
            placeholder_names
        )
        for arg_idx in range(orig_ctr, orig_ctr + num_lambda_args):
            arg = f"_{arg_idx}"
            lambda_args.append(arg)
            extra_defaults.discard(arg)
        lambda_args.extend(placeholder_names)
        if created_starred_arg:
            lambda_args.append(f"*_{cls._arg_ctr}")
            cls._arg_ctr += 1
        modified_lambda_body: ast.expr | None = None
        if frame.f_locals is not frame.f_globals:
            freevars = {
                arg
                for arg in extra_defaults
                if arg not in frame.f_globals
                and arg not in frame.f_locals
                and not hasattr(builtins, arg)
            }
            if len(freevars) > 0:
                modified_lambda_body = FreeVarTransformer(freevars, frame).visit(
                    lambda_body
                )
        extra_defaults = {arg for arg in extra_defaults if arg in frame.f_locals}
        lambda_arg_str = ", ".join(
            itertools.chain(lambda_args, (f"{arg}={arg}" for arg in extra_defaults))
        )
        return (
            cast(
                ast.Lambda,
                cast(
                    ast.Expr, ast.parse(f"lambda {lambda_arg_str}: None").body[0]
                ).value,
            ),
            extra_defaults,
            modified_lambda_body,
        )


# TODO: this analysis logic is doing double duty for chain transformations and pipeline step transformations,
#  leading to some confusing logic / flags, e.g. `check_all_calls` and `allow_top_level`. Probably the common
#  functionality should be extracted and two separate classes used for each type of transformation.
#  `allow_top_level` is still probably sound as a flag.
class PlaceholderReplacer(ast.NodeVisitor, SingletonArgCounterMixin):
    def __init__(self, arg_placeholder_spec: pyc.AugmentationSpec) -> None:
        self.mutate = False
        self.allow_top_level = False
        self.check_all_calls = False
        self.placeholder_names: dict[str, None] = {}
        self.arg_placeholder_spec = arg_placeholder_spec

    @contextmanager
    def disallow_top_level(self) -> Generator[None, None, None]:
        old_allow_top_level = self.allow_top_level
        try:
            self.allow_top_level = False
            yield
        finally:
            self.allow_top_level = old_allow_top_level

    def visit_Call(self, node: ast.Call) -> None:
        if self.check_all_calls:
            with self.disallow_top_level():
                self.generic_visit(node)
            return
        if not isinstance(node.func, ast.BinOp) or not pyc.BaseTracer.get_augmentations(
            id(node.func)
        ):
            # don't want to disallow top level yet -- if node.func is a call, still want to
            # be able to visit its args and keywords
            self.visit(node.func)
        if not self.allow_top_level:
            # defer visiting nested calls
            return
        with self.disallow_top_level():
            for arg in node.args:
                self.visit(arg)
            for kw in node.keywords:
                self.visit(kw.value)

    def visit_Subscript(self, node: ast.Subscript) -> None:
        from pipescript.tracers.macro_tracer import MacroTracer

        if isinstance(node.value, ast.Name) and node.value.id in MacroTracer.macros:
            # defer visiting nested quick lambdas
            return
        self.generic_visit(node)

    def visit_BinOp(self, node: ast.BinOp) -> None:
        if (
            not self.allow_top_level
            and isinstance(node.op, ast.BitOr)
            and pyc.BaseTracer.get_augmentations(id(node))
        ):
            # defer visiting nested pipeline ops
            return
        with self.disallow_top_level():
            self.generic_visit(node)

    def visit_Name(self, node: ast.Name) -> None:
        if (
            id(node)
            not in pyc.BaseTracer.augmented_node_ids_by_spec[self.arg_placeholder_spec]
        ):
            return
        assert node.id.startswith("_")
        arg_ctr = self.arg_ctr
        name = node.id
        if name == "_":
            self.arg_ctr += 1
        elif len(name) > 1 and name[0] == "_" and name[1].isalpha():
            name = name[1:]
            self.placeholder_names[name] = None
        if not self.mutate:
            return
        if name == "_":
            node.id = f"_{arg_ctr}"
        else:
            node.id = name
        pyc.BaseTracer.augmented_node_ids_by_spec[self.arg_placeholder_spec].discard(
            id(node)
        )

    def search(
        self,
        node: ast.AST | Sequence[ast.AST],
        allow_top_level: bool,
        check_all_calls: bool,
    ) -> bool:
        if isinstance(node, list):
            return any(
                self.search(
                    inner,
                    allow_top_level=allow_top_level,
                    check_all_calls=check_all_calls,
                )
                for inner in node
            )
        assert isinstance(node, ast.AST)
        orig_ctr = self.arg_ctr
        try:
            self.allow_top_level = allow_top_level
            self.check_all_calls = check_all_calls
            self.visit(node)
            found = self.arg_ctr > orig_ctr or len(self.placeholder_names) > 0
        finally:
            self.arg_ctr = orig_ctr
            self.placeholder_names.clear()
        return found

    def rewrite(
        self, node: ast.expr, allow_top_level: bool, check_all_calls: bool
    ) -> list[str]:
        old_mutate = self.mutate
        try:
            self.mutate = True
            self.allow_top_level = allow_top_level
            self.check_all_calls = check_all_calls
            self.visit(node)
            ret = self.placeholder_names
        finally:
            self.mutate = old_mutate
            self.placeholder_names = {}
        return list(ret.keys())
