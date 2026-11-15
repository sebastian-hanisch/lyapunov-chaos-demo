"""Kennzahlen, Verdikt-Logik und Tabellen (DEMO-PLAYBOOK §12: jedes Verdikt ist eine reine Funktion (Art, Text); die App ruft st.<Art>(Text)).

Zahlen in Texten haben ein Dezimalkomma (`de`). Jede Zahl, die ein Verdikt nennt, kommt aus der Kennzahl-Struktur, die der Kern liefert."""

import json
import math
from dataclasses import dataclass
from pathlib import Path

import numpy as np

import lyc_algorithm as A
import lyc_constants as C

ROOT = Path(__file__).resolve().parent
LN10 = math.log(10.0)


def de(x: float, nd: int = 2) -> str:
    """Zahl mit Dezimalkomma und nd Nachkommastellen (Minus als „−“)."""
    return f"{x:.{nd}f}".replace(".", ",").replace("-", "−")


def de_sci(x: float, nd: int = 1) -> str:
    """Zahl in Zehnerpotenz-Schreibweise mit Dezimalkomma, z. B. 3,2·10⁻⁶."""
    if x == 0:
        return "0"
    mant, exp = f"{x:.{nd}e}".split("e")
    sup = str(int(exp)).replace("-", "⁻").translate(str.maketrans("0123456789", "⁰¹²³⁴⁵⁶⁷⁸⁹"))
    return f"{mant.replace('.', ',')}·10{sup}"


def load_frozen() -> dict:
    """Die vorgerechneten Messungen (data/messungen.json, erzeugt von tools/messung_schaetzfehler.py)."""
    return json.loads((ROOT / "data" / "messungen.json").read_text(encoding="utf-8"))


@dataclass(frozen=True)
class Settings:
    r: float = C.DEFAULT_R
    seed: int = C.DEFAULT_SEED
    e1_lo: float = C.E1_LO_DEFAULT
    e1_hi: float = C.E1_HI_DEFAULT
    e1_ncols: int = C.DEFAULT_E1_NCOLS
    e1_n: int = C.DEFAULT_E1_N
    e1_thr: float = C.DEFAULT_E1_THR
    e2_exp: int = C.DEFAULT_E2_EXP
    e2_tol: float = C.DEFAULT_E2_TOL
    e4_x0: float = C.DEFAULT_E4_X0
    e4_n: int = C.DEFAULT_E4_N
    e5_exp: int = C.DEFAULT_E5_EXP
    e5_n: int = C.DEFAULT_E5_N


# ============================================================ Experiment 1 =================================================================
@dataclass(frozen=True)
class CurveAnalysis:
    lo: float
    hi: float
    thr: float
    n: int
    rs: np.ndarray
    lam: np.ndarray
    counts: A.CurveCounts
    window: tuple[int, int] | None     # (Werte, davon λ < 0) im Bereich 3,8284 bis 3,857, None wenn der Bereich keinen Wert dort hat


def effective_range(lo: float, hi: float) -> tuple[float, float]:
    """Mindestbreite 0,005 (der obere Regler kann unter dem unteren stehen)."""
    return lo, max(hi, lo + 0.005)


def analyse_curve(lo: float, hi: float, n_cols: int, n: int, thr: float) -> CurveAnalysis:
    lo, hi = effective_range(lo, hi)
    rs = np.linspace(lo, hi, n_cols)
    lam = A.lambda_curve(rs, n)
    w = A.window_counts(rs, lam)
    return CurveAnalysis(lo, hi, thr, n, rs, lam, A.count_curve(rs, lam, thr), w if w[0] else None)


def verdict_curve(c: CurveAnalysis) -> tuple[str, str]:
    cc = c.counts
    parts = []
    if cc.n_above:
        parts.append(f"Von {cc.n_above} r-Werten oberhalb von {de(C.ONSET, 7)} haben {cc.pos} ({de(100 * cc.pos / cc.n_above, 1)} %) ein λ über {de(c.thr, 4)}, "
                     f"{cc.zero} ein |λ| höchstens {de(c.thr, 4)} und {cc.neg} ein λ unter −{de(c.thr, 4)} (Fenster mit stabilem Zyklus).")
    if cc.max_below is not None:
        parts.append(f"Unterhalb davon (r ≥ {de(c.rs[0], 3)}) ist das größte λ {de(cc.max_below, 4)}; {cc.n_below_pos} Werte liegen über der Schwelle.")
    if c.window is not None:
        parts.append(f"Im Bereich 3,8284 bis {de(C.WINDOW_END, 3)} haben {c.window[1]} von {c.window[0]} r-Werten ein λ unter 0.")
    if not parts:
        parts.append("Der Ausschnitt enthält keinen r-Wert.")
    parts.append(f"Das gilt für dieses Raster ({len(c.rs)} r-Werte, N = {c.n}) und diese Schwelle, nicht für „fast alle r“.")
    return "info", " ".join(parts)


# ============================================================ Experiment 2 =================================================================
@dataclass(frozen=True)
class HorizonAnalysis:
    r: float
    d0: float
    tol: float
    steps: np.ndarray                  # Horizont je Start (HORIZON_NMAX, wenn nicht erreicht)
    lam: float                         # Exponent: ln 2 bei r = 4, sonst Schätzung aus der Bahn
    lam_exact: bool
    formula: float | None
    path: np.ndarray                   # Abstand der beiden Bahnen des ersten Starts
    reached: int                       # Zahl der Starts mit erreichter Toleranz
    float_mean: float                  # mittlerer Horizont der Float64-Bahn gegen die Decimal-Bahn (gleiche Starts, gleiche Toleranz)
    extent: float | None = None        # Breite der Bahn (Maximum minus Minimum nach dem Einschwingen), gemessen an einer Bahn; nur gesetzt, wenn kein Start die Toleranz erreicht


def estimate_lambda(r: float, seed: int) -> float:
    """λ(r): ln 2 bei r = 4 (exakt), sonst Mittel über LAMBDA_STARTS Schätzungen (N = LAMBDA_N, 3000 Einschwingschritte)."""
    if r == 4.0:
        return A.LN2
    x0 = A.start_values(seed, C.LAMBDA_STARTS)
    return float(A.lyapunov_estimates(r, x0, (C.LAMBDA_N,), transient=3000)[C.LAMBDA_N].mean())


def analyse_horizon(r: float, exp: int, tol: float, seed: int) -> HorizonAnalysis:
    d0 = 10.0 ** -exp
    x0 = A.start_values(seed, C.E2_STARTS)
    steps = A.horizon_decimal(r, x0, d0, tol)
    lam = estimate_lambda(r, seed)
    reached = int(np.sum(steps < C.HORIZON_NMAX))
    extent = A.attractor_extent(r, float(x0[0])) if reached == 0 else None
    return HorizonAnalysis(r, d0, tol, steps, lam, r == 4.0, A.horizon_formula(lam, d0, tol), A.distance_path(r, float(x0[0]), d0, int(min(steps[0] + 3, C.HORIZON_NMAX))), reached,
                           float(A.horizon_float(r, x0, tol).mean()), extent)


def effective_initial_error(lam: float, tol: float, mean_steps: float) -> float | None:
    """Anfangsfehler, den die Formel n = ln(tol/δ₀)/λ zu einem mittleren Horizont gehörig machte: δ₀ = tol·e^(−λ·n). Nur für λ > 0; gerechnete Größe, nicht gemessen."""
    return tol * math.exp(-lam * mean_steps) if lam > 0 else None


def verdict_horizon(h: HorizonAnalysis) -> tuple[str, str]:
    if h.reached == 0 and h.lam <= 0:
        return "info", (f"Bei r = {de(h.r)} wächst der Abstand der beiden Bahnen nicht auf die Toleranz {de(h.tol, 2)} (λ ≈ {de(h.lam, 4)} ≤ 0): "
                        f"in keinem der {len(h.steps)} Starts innerhalb von {C.HORIZON_NMAX} Schritten. Es gibt keinen Vorhersagehorizont, kleine Fehler schrumpfen.")
    if h.reached == 0:
        head = (f"Bei r = {de(h.r)} (λ ≈ {de(h.lam, 4)} > 0, kleine Fehler wachsen also im Mittel) erreicht der Abstand der beiden Bahnen die Toleranz {de(h.tol, 2)} "
                f"in keinem der {len(h.steps)} Starts innerhalb von {C.HORIZON_NMAX} Schritten.")
        if h.extent is not None and h.tol >= h.extent:
            return "info", (head + f" Der Grund liegt nicht im Exponenten: Die Toleranz ist nicht kleiner als die Breite der Bahn (gemessen an einer Bahn: {de(h.extent, 2)}), "
                            "der Abstand zweier Bahnen kann sie deshalb nicht überschreiten. Eine kleinere Toleranz wählen.")
        tail = f" Die Formel ln(Toleranz/Anfangsfehler)/λ ergibt {de(h.formula, 0)} Schritte." if h.formula is not None else ""
        return "info", (head + tail + " Es gibt hier keinen Horizont innerhalb dieser Schrittzahl; ein größerer Anfangsfehler oder eine kleinere Toleranz verkürzt ihn.")
    mean = float(h.steps[h.steps < C.HORIZON_NMAX].mean())
    text = (f"Mittlerer Vorhersagehorizont über {h.reached} Starts: {de(mean, 1)} Schritte (kleinster {int(h.steps.min())}, größter {int(h.steps[h.steps < C.HORIZON_NMAX].max())}); "
            f"Anfangsfehler {de_sci(h.d0, 0)}, Toleranz {de(h.tol, 2)}, Decimal mit {C.DECIMAL_DIGITS} Stellen.")
    if h.formula is not None:
        text += (f" Die Formel ln(Toleranz/Anfangsfehler)/λ mit λ = {de(h.lam, 4)}{' (exakt ln 2)' if h.lam_exact else ' (aus der Bahn geschätzt)'} gibt {de(h.formula, 1)}; "
                 f"Mittel minus Formel: {de(mean - h.formula, 1)} Schritte.")
    eff = effective_initial_error(h.lam, h.tol, h.float_mean)
    if eff is not None and h.float_mean < C.HORIZON_NMAX:
        text += (f" Zum Vergleich weicht die Float64-Bahn selbst im Mittel nach {de(h.float_mean, 1)} Schritten um mehr als die Toleranz von der Decimal-Bahn ab; nach der Formel entspricht das einem "
                 f"wirksamen Anfangsfehler von {de_sci(eff, 1)} (gerechnet, nicht direkt gemessen; zum Vergleich 2⁻⁵³ = 1,1·10⁻¹⁶).")
    if h.reached < len(h.steps):
        text += f" In {len(h.steps) - h.reached} Starts wurde die Toleranz bis {C.HORIZON_NMAX} Schritte nicht erreicht (nicht im Mittel)."
    return "success", text


@dataclass(frozen=True)
class DecadeAnalysis:
    r: float
    lam: float
    per_decade: float | None           # Mittel-Differenz je Zehnerpotenz zwischen δ₀ = 10⁻⁶ und 10⁻¹²
    formula: float | None              # ln(10)/λ


def analyse_decade(r: float, seed: int, tol: float = 0.1) -> DecadeAnalysis:
    x0 = A.start_values(seed, C.E2_STARTS)
    lam = estimate_lambda(r, seed)
    if lam <= 0:
        return DecadeAnalysis(r, lam, None, None)
    a = A.horizon_decimal(r, x0, 1e-6, tol)
    b = A.horizon_decimal(r, x0, 1e-12, tol)
    return DecadeAnalysis(r, lam, float((b.mean() - a.mean()) / 6.0), LN10 / lam)


def verdict_decade(d: DecadeAnalysis) -> tuple[str, str]:
    if d.formula is None:
        return "info", f"Bei r = {de(d.r)} ist λ = {de(d.lam, 4)} ≤ 0: ein kleinerer Anfangsfehler hilft nicht gegen Wachstum, denn es gibt keines; der Fehler schrumpft."
    return "success", (f"Ein zehnmal kleinerer Anfangsfehler kauft nur ln(10)/λ = {de(d.formula, 2)} zusätzliche Schritte (λ = {de(d.lam, 4)}); "
                       f"gemessen (Mittel über {C.E2_STARTS} Starts, Decimal, Anfangsfehler 10⁻⁶ gegen 10⁻¹², Toleranz 0,1): {de(d.per_decade, 2)} Schritte je Zehnerpotenz.")


# ============================================================ Experiment 3 =================================================================
@dataclass(frozen=True)
class ErrorAnalysis:
    n_starts: int
    checks: tuple
    rows4: list
    rows39: list
    slope4: float
    slope39: float
    slope4_rms: float
    slope39_rms: float
    ref39: float


def analyse_errors(seed: int) -> ErrorAnalysis:
    x0 = A.start_values(seed, C.E3_STARTS_LIVE)
    chk = C.E3_CHECKS_LIVE
    e4 = A.lyapunov_estimates(4.0, x0, chk)
    e39 = A.lyapunov_estimates(3.9, x0, chk)
    ref39 = float(A.lyapunov_estimates(3.9, x0, (20000,))[20000].mean())
    rows4, rows39 = A.estimation_errors(e4, A.LN2), A.estimation_errors(e39, ref39)
    return ErrorAnalysis(len(x0), chk, rows4, rows39, A.loglog_slope(chk, [w.median_abs for w in rows4]), A.loglog_slope(chk, [w.median_abs for w in rows39]),
                         A.loglog_slope(chk, [w.rms for w in rows4]), A.loglog_slope(chk, [w.rms for w in rows39]), ref39)


def verdict_errors(e: ErrorAnalysis, frozen: dict) -> tuple[str, str]:
    f4, f39 = frozen["r4"], frozen["r39"]
    return "info", (f"Median des Fehlers von λ_N gegen N im log-log-Bild: r = 4: Steigung {de(e.slope4, 2)} (≈ −1, also 1/N), r = 3,9: {de(e.slope39, 2)} (≈ −0,5, also 1/√N); "
                    f"live an {e.n_starts} Starts, N = {', '.join(str(n) for n in e.checks)}. Vorgerechnet an {frozen['n_starts']} Starts bis N = 10⁵: {de(f4['slope_median'], 3)} und {de(f39['slope_median'], 3)}. "
                    f"Bei r = 4 liegt der Grund in der Teleskopsumme (Experiment 4). Median statt quadratisches Mittel, weil bei r = 4 {len(frozen['zero_hits_r4'])} Bahn von {frozen['n_starts']} in 10⁵ Schritten auf 0 fällt: "
                    f"das quadratische Mittel hat dort die Steigung {de(f4['slope_rms'], 3)}.")


# ============================================================ Experiment 4 =================================================================
def verdict_telescope(t: A.Telescope) -> tuple[str, str]:
    if not (t.xn == t.xn) or not 0.0 < t.xn < 1.0:
        return "warning", f"Die Bahn endet nach {t.n} Schritten bei x_N = {de(t.xn, 4)} (auf dem Rand): Das Randglied ln(x_N(1 − x_N)) ist nicht definiert, die Identität gibt hier nichts."
    gap = abs(t.sum_log - t.formula)
    return "success", (f"Σ ln|f′| = {de(t.sum_log, 8)} gegen N·ln 2 + ½·ln(x_N(1 − x_N)) − ½·ln(x₀(1 − x₀)) = {de(t.formula, 8)} (Differenz {de_sci(gap, 1)}). "
                       f"Der Schätzfehler λ_N − ln 2 = {de(t.error, 6)} ist gleich dem Randterm [ln(x_N(1 − x_N)) − ln(x₀(1 − x₀))]/(2N) = {de(t.error_formula, 6)}: er fällt wie 1/N, solange x_N nicht an 0 oder 1 liegt. Das gilt für die berechnete Float64-Folge; die exakte Bahn von x₀ ist nach rund 50 Schritten eine andere.")


# ============================================================ Experiment 5 =================================================================
@dataclass(frozen=True)
class TwoOrbitAnalysis:
    r: float
    d0: float
    n: int
    derivative: np.ndarray
    two_orbit: np.ndarray
    finite: int
    mean_two: float | None
    max_dev: float | None


def analyse_two_orbit(r: float, exp: int, n: int, seed: int) -> TwoOrbitAnalysis:
    d0 = 10.0 ** -exp
    x0 = A.start_values(seed, C.E5_STARTS)
    der = A.lyapunov_estimates(r, x0, (n,))[n]
    zw = A.lyapunov_two_orbit(r, x0, d0, n)
    ok = np.isfinite(zw)
    return TwoOrbitAnalysis(r, d0, n, der, zw, int(ok.sum()), float(zw[ok].mean()) if ok.any() else None, float(np.abs(zw[ok] - der[ok]).max()) if ok.any() else None)


def verdict_two_orbit(t: TwoOrbitAnalysis) -> tuple[str, str]:
    total = len(t.two_orbit)
    if t.finite == 0:
        return "warning", f"Bei d₀ = {de_sci(t.d0, 0)} ist das Ergebnis des Zweibahn-Verfahrens in keinem der {total} Starts endlich (der Abstand der Bahnen wird durch Rundung 0)."
    if t.finite < total:
        return "warning", (f"Bei d₀ = {de_sci(t.d0, 0)} ist das Ergebnis in nur {t.finite} von {total} Starts endlich (der Abstand der Bahnen wird durch Rundung 0, die Renormierung teilt durch 0); "
                           f"unter den endlichen: Mittel {de(t.mean_two, 5)}, größte Abweichung von der Ableitungsmethode {de_sci(t.max_dev, 1)}.")
    return "success", (f"Bei d₀ = {de_sci(t.d0, 0)} sind alle {total} Ergebnisse endlich; Mittel Zweibahn {de(t.mean_two, 5)} gegen Ableitungsmethode {de(float(t.derivative.mean()), 5)}, "
                       f"größte Abweichung je Start {de_sci(t.max_dev, 1)} (N = {t.n}).")
