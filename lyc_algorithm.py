"""Kern der Demo (numpy, Standardbibliothek, Decimal): Lyapunov-Exponent und Vorhersagehorizont der logistischen Abbildung f(x) = r·x·(1 − x).

Jede Regel ist eine eigene Funktion mit expliziten Ein- und Ausgaben (DEMO-PLAYBOOK §1b):

* Exponent aus der Bahn (Ableitungsmethode)  `lyapunov_estimates`, `lambda_curve`, `cycle_exponent`
* Zweibahn-Verfahren (Benettin, Renormierung)  `lyapunov_two_orbit`
* Konjugation bei r = 4                         `to_theta`, `tent`, `telescope`
* Vorhersagehorizont                            `horizon_decimal`, `horizon_formula`
* Fehler der Schätzung gegen N                  `estimation_errors`, `loglog_slope`

Alles in Float64, der Horizont mit Decimal (130 Stellen) als Wahrheit. Zufall (Startwerte) nur über SplitMix64 (`lyc_rng`)."""

import math
from dataclasses import dataclass
from decimal import Decimal, localcontext

import numpy as np

import lyc_constants as C
from lyc_rng import SplitMix64

LN2 = math.log(2.0)
LOG_FLOOR = 1e-300          # |f′| = 0 genau (Bahn trifft 1/2) gibt log(0): der Wächter setzt |f′| auf diesen Wert und zählt den Fall


def start_values(seed: int, n: int, lo: float = C.E2_START_LO, hi: float = C.E2_START_HI) -> np.ndarray:
    """n Startwerte gleichverteilt in [lo, hi) aus SplitMix64 mit dem Zufalls-Seed `seed`."""
    return SplitMix64(seed).uniform_array(lo, hi, n)


# ============================================================ Exponent aus der Bahn =================================================
def lyapunov_estimates(r: float, x0: np.ndarray, checks, transient: int = 0) -> dict:
    """Schätzung λ_N = (1/N)·Σ ln|f′(x_i)| je Start, f′(x) = r(1 − 2x), für jedes N in `checks` (aufsteigend); gerechnet wird ein Lauf bis max(checks).
    Nach `transient` Einschwingschritten beginnt die Summe. Gibt {N: Feld der Schätzungen}."""
    x = np.array(x0, dtype=float)
    with np.errstate(divide="ignore", invalid="ignore", over="ignore"):
        for _ in range(transient):
            x = r * x * (1.0 - x)
        acc = np.zeros_like(x)
        out = {}
        wanted = set(checks)
        for i in range(1, max(checks) + 1):
            acc += np.log(np.maximum(np.abs(r * (1.0 - 2.0 * x)), LOG_FLOOR))
            x = r * x * (1.0 - x)
            if i in wanted:
                out[i] = acc / i
    return out


def lyapunov_single(r: float, x0: float, n: int, transient: int = 0) -> tuple[float, bool]:
    """Schätzung für einen Start und ob der Wächter ausgelöst wurde (|f′| = 0 genau: die Bahn traf 1/2)."""
    x = x0
    for _ in range(transient):
        x = r * x * (1.0 - x)
    acc = 0.0
    guard = False
    for _ in range(n):
        d = abs(r * (1.0 - 2.0 * x))
        if d == 0.0:
            guard = True
            d = LOG_FLOOR
        acc += math.log(d)
        x = r * x * (1.0 - x)
    return acc / n, guard


def lambda_curve(rs: np.ndarray, n: int, transient: int = C.E1_TRANSIENT, x0: float = C.E1_X0) -> np.ndarray:
    """λ_N für jedes r (alle gleichzeitig, vektorisiert): Startwert x0, `transient` Einschwingschritte, dann Summe über n Schritte."""
    x = np.full(rs.shape, float(x0))
    with np.errstate(divide="ignore", invalid="ignore", over="ignore"):
        for _ in range(transient):
            x = rs * x * (1.0 - x)
        acc = np.zeros_like(x)
        for _ in range(n):
            acc += np.log(np.maximum(np.abs(rs * (1.0 - 2.0 * x)), LOG_FLOOR))
            x = rs * x * (1.0 - x)
    return acc / n


@dataclass(frozen=True)
class CurveCounts:
    n_above: int          # r-Werte oberhalb der Zählgrenze ONSET
    pos: int              # davon λ > Schwelle
    zero: int             # davon |λ| ≤ Schwelle
    neg: int              # davon λ < −Schwelle
    max_below: float | None   # größtes λ unterhalb der Zählgrenze
    n_below_pos: int      # Werte unterhalb mit λ > Schwelle


def count_curve(rs: np.ndarray, lam: np.ndarray, thr: float, onset: float = C.ONSET) -> CurveCounts:
    sel = rs > onset
    below = ~sel
    return CurveCounts(int(sel.sum()), int(np.sum(lam[sel] > thr)), int(np.sum(np.abs(lam[sel]) <= thr)), int(np.sum(lam[sel] < -thr)),
                       float(lam[below].max()) if below.any() else None, int(np.sum(lam[below] > thr)))


def window_counts(rs: np.ndarray, lam: np.ndarray, lo: float = 3.8284, hi: float = C.WINDOW_END) -> tuple[int, int]:
    """(Zahl der r-Werte in [lo, hi], davon mit λ < 0)."""
    w = (rs >= lo) & (rs <= hi)
    return int(w.sum()), int(np.sum(lam[w] < 0))


def cycle_exponent(r: float, x0: float = C.E1_X0, transient: int = 20000, max_period: int = 16, tol: float = 1e-10):
    """Exponent eines stabilen Zyklus aus dem Multiplikator: (Periode p, λ = ln|Multiplikator|/p, Multiplikator); None, wenn die Bahn keinen Zyklus ≤ max_period hat."""
    x = x0
    for _ in range(transient):
        x = r * x * (1.0 - x)
    pts = [x]
    for _ in range(2 * max_period):
        pts.append(r * pts[-1] * (1.0 - pts[-1]))
    for p in range(1, max_period + 1):
        if max(abs(pts[i + p] - pts[i]) for i in range(len(pts) - p)) < tol:
            mult = 1.0
            for v in pts[:p]:
                mult *= r * (1.0 - 2.0 * v)
            return p, math.log(abs(mult)) / p, mult
    return None


# ============================================================ Zweibahn-Verfahren ====================================================
def lyapunov_two_orbit(r: float, x0: np.ndarray, d0: float, n: int) -> np.ndarray:
    """Zweibahn-Verfahren (Benettin): Eine zweite Bahn startet im Abstand d0, wird nach jedem Schritt auf den Abstand d0 zurückgesetzt;
    λ ≈ (1/n)·Σ ln(d_k/d0). Fällt der Abstand durch Rundung auf 0 (d0 nahe der Maschinengenauigkeit), ist das Ergebnis nicht endlich (nan oder ±inf)."""
    x = np.array(x0, dtype=float)
    y = x + d0
    acc = np.zeros_like(x)
    with np.errstate(divide="ignore", invalid="ignore", over="ignore"):
        for _ in range(n):
            x = r * x * (1.0 - x)
            y = r * y * (1.0 - y)
            d = np.abs(y - x)
            acc += np.log(d / d0)
            y = x + (y - x) / d * d0
    return acc / n


# ============================================================ Konjugation und Teleskop-Identität ====================================
def to_theta(x):
    """Konjugation x = sin²(πθ/2), θ = (2/π)·arcsin(√x): macht f(x) = 4x(1 − x) zur Zeltabbildung."""
    return (2.0 / math.pi) * np.arcsin(np.sqrt(np.asarray(x, dtype=float)))


def from_theta(theta):
    return np.sin(math.pi * np.asarray(theta, dtype=float) / 2.0) ** 2


def tent(theta):
    """Zeltabbildung T(θ) = 1 − |1 − 2θ| (Steigung ±2)."""
    return 1.0 - np.abs(1.0 - 2.0 * np.asarray(theta, dtype=float))


@dataclass(frozen=True)
class Telescope:
    x0: float
    n: int
    xn: float
    sum_log: float          # Σ ln|f′(x_i)| (r = 4), i < N
    formula: float          # N·ln 2 + ½·ln(x_N(1 − x_N)) − ½·ln(x_0(1 − x_0))
    estimate: float         # λ_N = Σ/N
    error: float            # λ_N − ln 2
    error_formula: float    # [ln(x_N(1 − x_N)) − ln(x_0(1 − x_0))]/(2N)


def telescope(x0: float, n: int) -> Telescope:
    """Teleskop-Identität bei r = 4: Wegen 1 − f(x) = (1 − 2x)² gilt |f′(x)| = 2·√(f(x)(1 − f(x)))/√(x(1 − x)), also Σ ln|f′| = N ln 2 + ½ ln(x_N(1 − x_N)) − ½ ln(x_0(1 − x_0))."""
    x = x0
    acc = 0.0
    for _ in range(n):
        acc += math.log(max(abs(4.0 * (1.0 - 2.0 * x)), LOG_FLOOR))
        x = 4.0 * x * (1.0 - x)
    boundary = 0.5 * math.log(x * (1.0 - x)) - 0.5 * math.log(x0 * (1.0 - x0)) if 0.0 < x < 1.0 else float("nan")
    return Telescope(x0, n, x, acc, n * LN2 + boundary, acc / n, acc / n - LN2, boundary / n)


def three_cycle_theta():
    """Der 3-Zyklus θ = 2/7 → 4/7 → 6/7 der Zeltabbildung und seine x-Werte sin²(πθ/2); Multiplikator ∏ 4(1 − 2x_i) = +8."""
    th = [2.0 / 7.0, 4.0 / 7.0, 6.0 / 7.0]
    xs = [float(from_theta(t)) for t in th]
    return th, xs, float(np.prod([4.0 * (1.0 - 2.0 * x) for x in xs]))


# ============================================================ Vorhersagehorizont ====================================================
def horizon_decimal(r: float, x0s, d0: float, tol: float, digits: int = C.DECIMAL_DIGITS, nmax: int = C.HORIZON_NMAX) -> np.ndarray:
    """Erster Schritt n mit |a_n − b_n| > tol für zwei Bahnen a, b im Abstand d0 (Decimal, `digits` Stellen), je Start; nmax, wenn die Schranke nicht erreicht wird."""
    out = []
    with localcontext() as ctx:
        ctx.prec = digits
        R = Decimal(repr(float(r)))
        D0 = Decimal(repr(float(d0)))
        T = Decimal(repr(float(tol)))
        for x0 in x0s:
            a = Decimal(float(x0))
            b = a + D0
            n = 0
            while n < nmax:
                a = R * a * (1 - a)
                b = R * b * (1 - b)
                n += 1
                if abs(a - b) > T:
                    break
            out.append(n)
    return np.array(out)


def distance_path(r: float, x0: float, d0: float, steps: int, digits: int = C.DECIMAL_DIGITS) -> np.ndarray:
    """Abstand |a_k − b_k| der beiden Decimal-Bahnen für k = 0 … steps (als float)."""
    res = []
    with localcontext() as ctx:
        ctx.prec = digits
        R = Decimal(repr(float(r)))
        a = Decimal(float(x0))
        b = a + Decimal(repr(float(d0)))
        res.append(float(abs(a - b)))
        for _ in range(steps):
            a = R * a * (1 - a)
            b = R * b * (1 - b)
            res.append(float(abs(a - b)))
    return np.array(res)


def horizon_float(r: float, x0s, tol: float, digits: int = C.DECIMAL_DIGITS, nmax: int = C.HORIZON_NMAX) -> np.ndarray:
    """Vorhersagehorizont der Float64-Bahn: erster Schritt, an dem sie von der Decimal-Bahn (gleicher Start, `digits` Stellen) um mehr als tol abweicht."""
    out = []
    with localcontext() as ctx:
        ctx.prec = digits
        R = Decimal(repr(float(r)))
        T = Decimal(repr(float(tol)))
        for x0 in x0s:
            a = Decimal(float(x0))
            f = float(x0)
            n = 0
            while n < nmax:
                a = R * a * (1 - a)
                f = r * f * (1.0 - f)
                n += 1
                if abs(a - Decimal(f)) > T:
                    break
            out.append(n)
    return np.array(out)


def horizon_formula(lam: float, d0: float, tol: float) -> float | None:
    """n ≈ ln(tol/d0)/λ; für λ ≤ 0 gibt es keinen Horizont (None)."""
    return math.log(tol / d0) / lam if lam > 0 else None


def attractor_extent(r: float, x0: float, transient: int = 3000, n: int = 3000) -> float:
    """Breite der Bahn: Maximum minus Minimum von n Float64-Schritten nach `transient` Einschwingschritten (eine Bahn, gemessen; nur Text des Verdikts zum Horizont)."""
    x = float(x0)
    for _ in range(transient):
        x = r * x * (1.0 - x)
    lo = hi = x
    for _ in range(n):
        x = r * x * (1.0 - x)
        lo, hi = min(lo, x), max(hi, x)
    return hi - lo


# ============================================================ Schätzfehler gegen N ==================================================
@dataclass(frozen=True)
class ErrorRow:
    n: int
    median_abs: float
    rms: float
    over_1e3: int           # Starts mit |Fehler| > 1e-3


def estimation_errors(est: dict, ref: float) -> list[ErrorRow]:
    rows = []
    for n in sorted(est):
        e = est[n] - ref
        rows.append(ErrorRow(n, float(np.median(np.abs(e))), float(np.sqrt(np.mean(e ** 2))), int(np.sum(np.abs(e) > 1e-3))))
    return rows


def loglog_slope(ns, values) -> float:
    return float(np.polyfit(np.log(np.asarray(ns, dtype=float)), np.log(np.asarray(values, dtype=float)), 1)[0])


def first_zero_hit(r: float, x0: np.ndarray, steps: int) -> list[tuple[int, int]]:
    """[(Start-Nr, erster Schritt mit x = 0 exakt)] für die Starts, deren Float64-Bahn auf 0 fällt."""
    x = np.array(x0, dtype=float)
    first = np.full(x.shape, -1)
    for i in range(1, steps + 1):
        x = r * x * (1.0 - x)
        hit = (x == 0.0) & (first < 0)
        first[hit] = i
    return [(int(j), int(first[j])) for j in np.where(first >= 0)[0]]
