from __future__ import annotations

import os
import sys

import pyccolo as pyc
from pyccolo.__main__ import main as pyccolo_main

import pipescript.tracers


def main() -> int:
    script_path = os.path.abspath(sys.argv[1])
    tracer_refs = [
        f"{pipescript.tracers.__name__}.{tracer_name}"
        for tracer_name in pipescript.tracers.__all__
    ]
    for tracer_ref in tracer_refs:
        tracer = pyc.resolve_tracer(tracer_ref).instance()
        tracer._tracing_enabled_files.add(script_path)
    sys.argv.extend(["-t"] + tracer_refs)
    return pyccolo_main()


if __name__ == "__main__":
    sys.exit(main())
