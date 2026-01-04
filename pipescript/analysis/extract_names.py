from __future__ import annotations

import ast
from typing import TYPE_CHECKING

import pyccolo.fast as fast

if TYPE_CHECKING:
    from pipescript.tracers.pipeline_tracer import PipelineTracer


class ExtractNames(ast.NodeVisitor):
    def __init__(self, tracer: PipelineTracer) -> None:
        self.names: set[str] = set()
        self.tracer = tracer

    def visit_Lambda(self, node: ast.Lambda) -> None:
        before_names = set(self.names)
        self.generic_visit(node.body)
        for arg in fast.iter_arguments(node.args):
            if arg.arg not in before_names:
                self.names.discard(arg.arg)

    def visit_Name(self, node: ast.Name) -> None:
        if (
            node.id != "_"
            and id(node)
            not in self.tracer.augmented_node_ids_by_spec[
                self.tracer.arg_placeholder_spec
            ]
        ):
            self.names.add(node.id)

    def generic_visit_comprehension(
        self, node: ast.GeneratorExp | ast.DictComp | ast.ListComp | ast.SetComp
    ) -> None:
        before_names = set(self.names)
        self.generic_visit(node)
        after_names = self.names
        self.names = set()
        for gen in node.generators:
            self.visit(gen.target)
        # need to clear names referenced as targets since these
        # do not need to be passed externally to any lambdas
        for name in self.names:
            if name not in before_names:
                after_names.discard(name)
        self.names = after_names

    visit_GeneratorExp = visit_DictComp = visit_ListComp = visit_SetComp = (
        generic_visit_comprehension
    )

    @classmethod
    def extract_names(cls, node: ast.expr) -> set[str]:
        from pipescript.tracers.pipeline_tracer import PipelineTracer

        visitor = cls(PipelineTracer.instance())
        visitor.visit(node)
        return visitor.names
