# -*- coding: utf-8 -*-
"""Pytest configuration for the pipescript test suite.

Several tests activate pyccolo tracers (``with PipelineTracer:`` etc.). When a
module is first imported inside the scope of a tracer that instruments it,
pyccolo rewrites the module's AST; with bytecode caching on, that *instrumented*
bytecode -- which references pyccolo's per-session emit builtins -- can be
persisted to ``__pycache__``. A later plain import then loads that stale cached
bytecode with no tracer active and dies with a NameError on the emit builtin.

Disabling bytecode writes for the test process makes the suite incapable of
leaving such artifacts behind, no matter which tracer a test activates. No test
relies on bytecode caching, so this only costs a little recompilation.
"""
import sys

# Must run at import time (before pytest collects/imports any test module).
sys.dont_write_bytecode = True
