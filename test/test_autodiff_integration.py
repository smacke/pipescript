"""
Integration test: pyccolo's reverse-mode autodiff example driven through
pipescript's ``|>`` pipe operator.

The pipe applies a function inside the PipelineTracer's own handler, not at an
instrumented call site, so a co-active ``before_call`` autodiff tracer cannot see
it. ``PipelineTracer.application_hooks`` is the supported way to participate: we
register a hook that, when the piped value is a ``Var``, resolves the function
the same way ``before_call`` would (numpy/math swap, or helper instrumentation).

Skipped unless numpy and the autodiff example (a recent pyccolo) are available.
"""

from __future__ import annotations

from typing import Generator

import pytest

np = pytest.importorskip("numpy")
autodiff = pytest.importorskip("pyccolo.examples.autodiff")

import pyccolo as pyc  # noqa: E402

from pipescript.tracers.pipeline_tracer import PipelineTracer  # noqa: E402

Var = autodiff.Var
AutodiffTracer = autodiff.AutodiffTracer
resolve_call = autodiff.resolve_call
value_and_grad = autodiff.value_and_grad


def _autodiff_hook(func: object, value: object) -> object:
    return resolve_call(func) if isinstance(value, Var) else func


@pytest.fixture
def autodiff_pipes() -> Generator[None, None, None]:
    PipelineTracer.application_hooks.append(_autodiff_hook)
    try:
        with AutodiffTracer.instance():
            with PipelineTracer:
                yield
    finally:
        PipelineTracer.application_hooks.remove(_autodiff_hook)


def relu(z):  # a user helper with no autodiff rule -> would need on-demand instrument
    return np.maximum(z, 0.0)


def _param_loss(model):  # module scope so value_and_grad can getsource + recompile
    return np.sum(model["w"] * model["w"]) + np.sum(model["b"])


def test_pipe_numpy_function_on_var(autodiff_pipes: None) -> None:
    ns = {"np": np}
    x = Var(np.array([1.0, 2.0, 3.0]))
    ns["x"] = x
    loss = pyc.eval("x |> np.exp |> np.sum", ns, ns)  # sum(exp(x))
    loss.backward()
    assert np.allclose(loss.value, np.sum(np.exp([1.0, 2.0, 3.0])))
    assert np.allclose(x.grad, np.exp([1.0, 2.0, 3.0]))  # d/dx sum(exp x) = exp x


def test_pipe_chain_matches_finite_difference(autodiff_pipes: None) -> None:
    # log(sum(maximum(x, 0))) via a pipe chain, gradient-checked.
    def f(arr):
        ns = {"np": np, "x": Var(arr)}
        out = pyc.eval("x |> np.square |> np.sum |> np.log", ns, ns)
        out.backward()
        return out.value, ns["x"].grad

    arr = np.array([0.5, -1.0, 2.0])
    val, grad = f(arr)
    h = 1e-6
    fd = np.array(
        [
            (np.log(np.sum((arr + h * e) ** 2)) - np.log(np.sum((arr - h * e) ** 2)))
            / (2 * h)
            for e in np.eye(len(arr))
        ]
    )
    assert np.allclose(grad, fd, atol=1e-5)


def test_plain_pipe_is_unaffected(autodiff_pipes: None) -> None:
    # When the piped value is not a Var, the hook is a no-op and numpy runs normally.
    result = pyc.eval("5.0 |> np.exp", {"np": np})
    assert not isinstance(result, Var)
    assert np.allclose(result, np.exp(5.0))


def test_pipe_through_user_helper(autodiff_pipes: None) -> None:
    # A user helper with no rule is instrumented on demand even though it is
    # applied while PipelineTracer is co-active (requires pyccolo to keep the
    # instrumenting tracer in the rewrite when it transiently disables itself).
    ns = {"np": np, "relu": relu}
    y = Var(np.array([-1.0, 2.0, -3.0, 4.0]))
    ns["y"] = y
    loss = pyc.eval("y |> relu |> np.sum", ns, ns)  # sum(relu(y))
    loss.backward()
    assert np.allclose(y.grad, (np.array([-1.0, 2.0, -3.0, 4.0]) > 0).astype(float))


def test_params_brace_surface_builds_param_pytree() -> None:
    # The literal ``params{ w = ...; b = ... }`` brace block, wired via pipescript's
    # namespace-block mechanism, harvests assignments into autodiff's Param pytree:
    # bare values are trainable, ``frozen(...)`` is held fixed, ``_`` names are
    # block-local temporaries, and the result trains like any other param tree.
    from pipescript.tracers.brace_block_tracer import BraceBlockTracer
    from pipescript.tracers.macro_tracer import MacroTracer
    from pipescript.tracers.optional_chaining_tracer import OptionalChainingTracer

    autodiff.register_pipescript_params_macro()
    rng = np.random.default_rng(0)
    ns = {"np": np, "rng": rng, "frozen": autodiff.frozen, "tied": autodiff.tied}
    # frozen[...] and tied[...] bracket sub-macros: tied[w] ties to sibling w by
    # name, reusing its init; `_scale` is a block-local temporary (excluded).
    src = (
        "params{\n"
        "  _scale = 0.1\n"
        "  w = _scale * rng.standard_normal((2, 3))\n"
        "  b = frozen[np.zeros(3)]\n"
        "  w_tied = tied[w]\n"
        "}"
    )
    try:
        with BraceBlockTracer:
            with PipelineTracer:
                with MacroTracer:
                    with OptionalChainingTracer:
                        model = pyc.eval(src, ns, ns)
    finally:
        MacroTracer.static_macros.pop("params", None)
        MacroTracer.namespace_block_macros.pop("params", None)
        __import__("builtins").__dict__.pop("params", None)

    Param = autodiff.Param
    assert sorted(model) == ["b", "w", "w_tied"]  # `_scale` temporary excluded
    assert isinstance(model["w"], Param) and model["w"].trainable
    assert isinstance(model["b"], Param) and not model["b"].trainable
    # tied[w] linked w_tied to w: same (non-None) tie key, shared init
    assert model["w"].tie is not None and model["w"].tie == model["w_tied"].tie
    assert np.allclose(model["w_tied"].value, model["w"].value)

    # the harvested pytree differentiates + updates: w gets a gradient, b is frozen
    _, (g,) = value_and_grad(_param_loss)(model)
    assert np.allclose(g["w"], 2 * model["w"].value)
    assert g["b"] is None
    stepped = autodiff.sgd_update(model, g, lr=0.1)
    assert np.allclose(stepped["b"].value, model["b"].value)  # frozen unchanged


def test_value_and_grad_over_pipe_lambda(autodiff_pipes: None) -> None:
    # A network defined as a pipe lambda has no recoverable source, so
    # value_and_grad runs it directly (it is already woven by the tracers) rather
    # than instrumenting it. Here the piped arg ($) is the parameter W.
    X = np.array([[1.0, 2.0], [-1.0, 0.5], [0.3, -2.0]])
    ns = {"np": np, "X": X}
    # loss(W) = sum(relu(X @ W))
    loss_pipe = pyc.eval("$ |> np.matmul(X, $) |> np.maximum(0.0, $) |> np.sum", ns, ns)

    w = np.array([[0.5], [-0.3]])
    value, (grad_w,) = value_and_grad(loss_pipe)(w)

    def loss_of(weights: np.ndarray) -> float:
        return float(np.sum(np.maximum(0.0, X @ weights)))

    assert np.allclose(value, loss_of(w))
    h = 1e-6
    fd = np.zeros_like(w)
    for i in range(w.shape[0]):
        for j in range(w.shape[1]):
            d = np.zeros_like(w)
            d[i, j] = h
            fd[i, j] = (loss_of(w + d) - loss_of(w - d)) / (2 * h)
    assert np.allclose(grad_w, fd, atol=1e-5)


def test_with_weights_pipe_forward_trains_and_infers_clean() -> None:
    # The headline: define the forward ONCE with unqualified weights via
    # ``with weights:`` (which injects Weight proxies), then get clean array
    # inference and tape-style training from the same definition -- no Var bound in.
    from pipescript.tracers.brace_block_tracer import BraceBlockTracer
    from pipescript.tracers.macro_tracer import MacroTracer
    from pipescript.tracers.optional_chaining_tracer import OptionalChainingTracer

    PipelineTracer.application_hooks.append(_autodiff_hook)
    g = globals()
    rng = np.random.default_rng(0)
    X = rng.standard_normal((6, 3))
    Y = np.eye(2)[rng.integers(0, 2, size=6)]
    setup = {
        "X": X,
        "Y": Y,
        "softmax": autodiff.softmax,
        "cross_entropy": autodiff.cross_entropy,
    }
    added = [k for k in setup if k not in g]  # only clean up what we introduce
    g.update({k: setup[k] for k in added})
    weights = autodiff.params(w=0.1 * rng.standard_normal((3, 2)), b=np.zeros(2))
    try:
        with BraceBlockTracer, PipelineTracer, MacroTracer, OptionalChainingTracer:
            with weights:  # inject proxies w, b into module globals
                model = pyc.eval("$ |> $ @ w + b |> softmax", g, g)
                preds = model(X)  # inference: arrays in, ndarray out, no tape
                g["model"] = model
                obj = pyc.eval("lambda: model(X) |> cross_entropy($, Y)", g, g)
                first = last = None
                for _ in range(40):
                    value, grads = weights.grad(obj)
                    first = float(value) if first is None else first
                    last = float(value)
                    weights.step(grads, 0.5)
        assert type(preds) is np.ndarray and np.allclose(preds.sum(1), 1.0)
        assert last < first and last < 0.1  # the same definition trained the weights
        assert "w" not in g and "b" not in g  # proxies cleaned up on exit
    finally:
        PipelineTracer.application_hooks.remove(_autodiff_hook)
        for k in added + ["model"]:
            g.pop(k, None)


def _fork_fd(A, B, x, h=1e-6):  # d/dx [sum(A@x) + sum(B@x)]
    def f(v):
        return float(np.sum(A @ v) + np.sum(B @ v))

    g = np.zeros_like(x)
    for i in range(len(x)):
        xp, xm = x.copy(), x.copy()
        xp[i] += h
        xm[i] -= h
        g[i] = (f(xp) - f(xm)) / (2 * h)
    return g


def test_fork_of_pipe_nets_differentiates() -> None:
    # fork[...] over branches that are pipescript-syntax nets (closing over A/B,
    # piping into np.sum) must differentiate -- regression for fork miscounting
    # captured free vars as placeholders. Needs the full macro stack, not just the
    # autodiff_pipes fixture (fork is a MacroTracer macro).
    from pipescript.tracers.brace_block_tracer import BraceBlockTracer
    from pipescript.tracers.macro_tracer import MacroTracer
    from pipescript.tracers.optional_chaining_tracer import OptionalChainingTracer

    A = np.random.default_rng(1).standard_normal((4, 3))
    B = np.random.default_rng(2).standard_normal((4, 3))
    ns = {"np": np, "A": A, "B": B}
    x = np.random.default_rng(3).standard_normal(3)
    PipelineTracer.application_hooks.append(_autodiff_hook)
    try:
        with BraceBlockTracer, PipelineTracer, MacroTracer, OptionalChainingTracer:
            loss = pyc.eval(
                "$ |> fork[A @ $ |> np.sum, B @ $ |> np.sum] |> sum", ns, ns
            )
            value, (gx,) = value_and_grad(loss)(x)
    finally:
        PipelineTracer.application_hooks.remove(_autodiff_hook)
    assert np.allclose(value, np.sum(A @ x) + np.sum(B @ x))
    assert np.allclose(gx, _fork_fd(A, B, x), atol=1e-5)


def _highway_ref(Wh, bh, Wt, bt, X):  # numpy reference for a highway layer
    H = np.maximum(X @ Wh + bh, 0.0)  # transform
    T = 1.0 / (1.0 + np.exp(-(X @ Wt + bt)))  # transform gate
    return H * T + X * (1.0 - T)  # carry the input by (1 - T)


def test_highway_network_via_fork() -> None:
    # A highway layer, y = H(x)*T(x) + x*(1 - T(x)), is a natural fork: send the
    # input through the identity (carry), the transform H, and the gate T, then
    # combine. Defined with pipescript syntax over ambient weights; trains end-to-end.
    #
    #     highway = $ |> fork[$, transform, gate] *|> ($x, $h, $t) \
    #                     *|> $h * $t + $x * (1 - $t)
    #
    # `fork[...]` yields (x, H, T); `*|> ($x, $h, $t)` names those three (this is the
    # one spot where binding is positional -- the naming order must match the branch
    # order), and the combine then refers to them by name, so it is order-independent.
    # The naming stage is just tuple repacking, transparent to the backward pass.
    from pipescript.tracers.brace_block_tracer import BraceBlockTracer
    from pipescript.tracers.macro_tracer import MacroTracer
    from pipescript.tracers.optional_chaining_tracer import OptionalChainingTracer

    PipelineTracer.application_hooks.append(_autodiff_hook)
    g = globals()
    rng = np.random.default_rng(0)
    d = 4
    X = rng.standard_normal((8, d))
    Y = np.tanh(X) + 0.5 * X  # a target to fit
    setup = {"X": X, "Y": Y, "relu": autodiff.relu, "sigmoid": autodiff.sigmoid}
    added = [k for k in setup if k not in g]
    g.update({k: setup[k] for k in added})
    weights = autodiff.params(
        Wh=0.3 * rng.standard_normal((d, d)),
        bh=np.zeros(d),
        Wt=0.3 * rng.standard_normal((d, d)),
        bt=np.zeros(d),
    )
    built: list[str] = []
    try:
        with BraceBlockTracer, PipelineTracer, MacroTracer, OptionalChainingTracer:
            with weights:  # inject Wh, bh, Wt, bt proxies
                g["transform"] = pyc.eval("$ |> $ @ Wh + bh |> relu", g, g)
                g["gate"] = pyc.eval("$ |> $ @ Wt + bt |> sigmoid", g, g)
                g["highway"] = pyc.eval(
                    "$ |> fork[$, transform, gate] "
                    "*|> ($x, $h, $t) *|> $h * $t + $x * (1 - $t)",
                    g,
                    g,
                )
                built = ["transform", "gate", "highway"]
                # the fork-built layer matches the numpy reference (architecture ok)
                y = g["highway"](X)
                ref = _highway_ref(*(weights[k].value for k in weights), X)
                assert type(y) is np.ndarray and np.allclose(y, ref)
                # ...and it trains: fit Y by MSE over the ambient weights
                mse = pyc.eval(
                    "lambda: highway(X) |> f{ ($ - Y) * ($ - Y) } |> np.mean", g, g
                )
                first = last = None
                for _ in range(150):
                    value, grads = weights.grad(mse)
                    first = float(value) if first is None else first
                    last = float(value)
                    weights.step(grads, 0.3)
        assert last < 0.5 * first  # learned a meaningful fit
    finally:
        PipelineTracer.application_hooks.remove(_autodiff_hook)
        for k in added + built:
            g.pop(k, None)
