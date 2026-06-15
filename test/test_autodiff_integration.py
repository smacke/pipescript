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
