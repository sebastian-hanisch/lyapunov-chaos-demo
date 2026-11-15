"""Plotly-Abbildungen. Alle Achsen sind gesperrt (fixedrange), damit Touch-Geräte beim Scrollen nicht am Diagramm hängen bleiben."""

import numpy as np
import plotly.graph_objects as go

import lyc_algorithm as A
import lyc_constants as C

FLOOR = 1e-300


def lock_axes(fig):
    fig.update_xaxes(fixedrange=True)
    fig.update_yaxes(fixedrange=True)
    return fig


def _base(fig, height=340):
    fig.update_layout(height=height, margin=dict(l=10, r=10, t=10, b=10), legend=dict(orientation="h", y=-0.22), plot_bgcolor="rgba(0,0,0,0)")
    return lock_axes(fig)


def _positive(values):
    v = np.asarray(values, dtype=float)
    return np.where(np.isfinite(v) & (v > 0), v, np.nan)


# ---------------------------------------------------------------- Experiment 1
def build_curve(c, r_now=None):
    lam = np.where(np.isfinite(c.lam), c.lam, np.nan)
    colors = np.where(lam > c.thr, C.COLOR_POS, np.where(lam < -c.thr, C.COLOR_NEG, C.COLOR_ZERO))
    fig = go.Figure()
    fig.add_trace(go.Scattergl(x=c.rs, y=lam, mode="markers", marker=dict(size=3.5, color=colors), name="λ_N(r)", hovertemplate="r = %{x:.4f}: λ = %{y:.4f}<extra></extra>"))
    fig.add_trace(go.Scatter(x=[c.rs[0], c.rs[-1]], y=[0, 0], mode="lines", line=dict(color=C.COLOR_GREY, width=1), name="λ = 0", hoverinfo="skip"))
    for x, label in ((C.ONSET, "3,5699456"), (C.WINDOW_R, "1 + √8")):
        if c.rs[0] <= x <= c.rs[-1]:
            fig.add_vline(x=x, line=dict(color="rgba(192,86,42,0.55)", width=1, dash="dot"))
            fig.add_annotation(x=x, y=1, yref="paper", text=label, showarrow=False, yanchor="bottom", font=dict(size=10, color="#c0562a"))
    if r_now is not None and c.rs[0] <= r_now <= c.rs[-1]:
        fig.add_vline(x=r_now, line=dict(color="black", width=1))
    fig.update_xaxes(title_text="r", range=[c.rs[0], c.rs[-1]])
    fig.update_yaxes(title_text="Lyapunov-Exponent λ_N")
    return _base(fig, 380)


# ---------------------------------------------------------------- Experiment 2
def build_distance(h):
    ks = np.arange(len(h.path))
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=ks, y=_positive(h.path), mode="lines+markers", line=dict(color=C.COLOR_MAIN, width=2), marker=dict(size=4), name="Abstand der beiden Bahnen (erster Start)",
                             hovertemplate="k = %{x}: %{y:.3e}<extra></extra>"))
    fig.add_trace(go.Scatter(x=[0, len(h.path) - 1], y=[h.tol, h.tol], mode="lines", line=dict(color=C.COLOR_REF, width=1, dash="dash"), name="Toleranz", hoverinfo="skip"))
    if h.lam > 0:
        y = h.d0 * np.exp(h.lam * ks)
        fig.add_trace(go.Scatter(x=ks, y=np.minimum(y, 1.0), mode="lines", line=dict(color=C.COLOR_GREY, width=1, dash="dot"), name="δ₀·e^(λk)", hoverinfo="skip"))
    fig.update_xaxes(title_text="Schritt k")
    fig.update_yaxes(title_text="Abstand (logarithmisch)", type="log")
    return _base(fig)


# ---------------------------------------------------------------- Experiment 3
def build_errors(e):
    ns = list(e.checks)
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=ns, y=[w.median_abs for w in e.rows4], mode="lines+markers", line=dict(color=C.COLOR_MAIN, width=3), name=f"r = 4 (Steigung {e.slope4:.2f})".replace(".", ","),
                             hovertemplate="N = %{x}: %{y:.2e}<extra></extra>"))
    fig.add_trace(go.Scatter(x=ns, y=[w.median_abs for w in e.rows39], mode="lines+markers", line=dict(color=C.COLOR_REF, width=3), name=f"r = 3,9 (Steigung {e.slope39:.2f})".replace(".", ","),
                             hovertemplate="N = %{x}: %{y:.2e}<extra></extra>"))
    n = np.array(ns, dtype=float)
    fig.add_trace(go.Scatter(x=ns, y=e.rows4[0].median_abs * (n[0] / n), mode="lines", line=dict(color=C.COLOR_GREY, width=1, dash="dot"), name="1/N", hoverinfo="skip"))
    fig.add_trace(go.Scatter(x=ns, y=e.rows39[0].median_abs * np.sqrt(n[0] / n), mode="lines", line=dict(color=C.COLOR_GREY, width=1, dash="dash"), name="1/√N", hoverinfo="skip"))
    fig.update_xaxes(title_text="Länge der Bahn N", type="log")
    fig.update_yaxes(title_text="Median |λ_N − λ|", type="log")
    return _base(fig)


# ---------------------------------------------------------------- Experiment 4
def build_conjugation(x0, steps=24):
    x = [x0]
    for _ in range(steps):
        x.append(4.0 * x[-1] * (1.0 - x[-1]))
    th = [float(A.to_theta(v)) for v in x]
    ks = np.arange(steps + 1)
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=ks, y=x, mode="lines+markers", line=dict(color=C.COLOR_MAIN, width=2), marker=dict(size=4), name="x_k (logistisch, r = 4)", hovertemplate="k = %{x}: %{y:.5f}<extra></extra>"))
    fig.add_trace(go.Scatter(x=ks, y=th, mode="lines+markers", line=dict(color=C.COLOR_REF, width=2), marker=dict(size=4), name="θ_k = (2/π)·arcsin √x_k",
                             hovertemplate="k = %{x}: %{y:.5f}<extra></extra>"))
    fig.update_xaxes(title_text="Schritt k")
    fig.update_yaxes(title_text="Wert", range=[0, 1])
    return _base(fig)


def build_tent():
    th = np.linspace(0, 1, 401)
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=th, y=A.tent(th), mode="lines", line=dict(color=C.COLOR_REF, width=3), name="T(θ) = 1 − |1 − 2θ| (Steigung ±2)"))
    fig.add_trace(go.Scatter(x=th, y=th, mode="lines", line=dict(color=C.COLOR_GREY, width=1, dash="dash"), name="y = θ", hoverinfo="skip"))
    cyc = [2 / 7, 4 / 7, 6 / 7]
    fig.add_trace(go.Scatter(x=cyc, y=[float(A.tent(t)) for t in cyc], mode="markers", marker=dict(size=10, color="black", symbol="diamond"), name="3-Zyklus 2/7 → 4/7 → 6/7"))
    fig.update_xaxes(title_text="θ", range=[0, 1])
    fig.update_yaxes(title_text="T(θ)", range=[0, 1])
    return _base(fig, 320)


# ---------------------------------------------------------------- Experiment 5
def build_two_methods(t):
    idx = np.arange(1, len(t.derivative) + 1)
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=idx, y=t.derivative, mode="markers", marker=dict(size=9, color=C.COLOR_MAIN), name="Ableitungsmethode", hovertemplate="Start %{x}: %{y:.5f}<extra></extra>"))
    ok = np.isfinite(t.two_orbit)
    fig.add_trace(go.Scatter(x=idx[ok], y=t.two_orbit[ok], mode="markers", marker=dict(size=9, color=C.COLOR_REF, symbol="x"), name="Zweibahn-Verfahren (endlich)", hovertemplate="Start %{x}: %{y:.5f}<extra></extra>"))
    if (~ok).any():
        lo = float(np.min(t.derivative))
        fig.add_trace(go.Scatter(x=idx[~ok], y=np.full((~ok).sum(), lo), mode="markers", marker=dict(size=9, color="black", symbol="circle-open"), name="nicht endlich", hoverinfo="skip"))
    fig.update_xaxes(title_text="Start Nr.")
    fig.update_yaxes(title_text="λ-Schätzung")
    return _base(fig)
