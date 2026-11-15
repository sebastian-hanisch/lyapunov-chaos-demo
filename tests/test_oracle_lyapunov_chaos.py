"""Orakel-Tests: die Kernroutinen gegen einen anderen Rechenweg (DEMO-PLAYBOOK §12, ORACLE-PLAYBOOK §1 und §5).

* λ bei stabilen Zyklen gegen den Zyklus-Multiplikator, der Zyklus unabhängig per Newton-Verfahren (komplexe Schritt-Ableitung) gelöst.
* λ(4) gegen ln 2 und gegen die Teleskop-Formel (1000 Zufallsstarts).
* Vorhersagehorizont: Decimal mit 130 gegen 200 Stellen (gleich); Float64-Bahn gegen Decimal bei großem Anfangsfehler (gleich).
* Zweibahn-Verfahren gegen Ableitungsmethode (Abweichung unter 10⁻⁵ bei d₀ = 10⁻⁹).
* λ der konjugierten Abbildung z → z² + (r/2 − r²/4) gleich λ der logistischen (z = r(1/2 − x)), Schritt für Schritt.
Zufall nur über SplitMix64."""

import math

import numpy as np
import pytest

import lyc_algorithm as A


def _f(r, x):
    return r * x * (1 - x)


def _newton_cycle_point(r, p, x_start, iterations=60):
    """Lösung von f^p(x) = x per Newton-Verfahren; die Ableitung von f^p liefert die komplexe Schritt-Ableitung (kein Produkt von f′ aus dem Kern)."""
    x = x_start
    for _ in range(iterations):
        z, y = complex(x, 1e-30), x
        for _ in range(p):
            z, y = _f(r, z), _f(r, y)
        g = y - x
        dg = z.imag / 1e-30 - 1.0
        step = g / dg
        x -= step
        if abs(step) < 1e-15:
            break
    return x


def test_cycle_exponent_equals_the_multiplier_of_an_independently_solved_cycle():
    for r, p, start in ((2.8, 1, 0.5), (3.2, 2, 0.8), (3.83, 3, 0.957)):
        x = _newton_cycle_point(r, p, start)
        y, mult = x, 1.0
        for _ in range(p):
            mult *= r * (1 - 2 * y)
            y = _f(r, y)
        assert y == pytest.approx(x, abs=1e-12)
        lam_newton = math.log(abs(mult)) / p
        per, lam, m = A.cycle_exponent(r)
        assert per == p and lam == pytest.approx(lam_newton, abs=1e-9) and m == pytest.approx(mult, abs=1e-9), r
        assert float(A.lambda_curve(np.array([r]), 30000, 3000)[0]) == pytest.approx(lam_newton, abs=1e-6), r


def test_lambda_4_equals_ln_2_and_the_telescope_formula_on_1000_random_starts():
    x0 = A.start_values(21, 1000)
    n = 1000
    x = x0.copy()
    acc = np.zeros_like(x)
    for _ in range(n):
        acc += np.log(np.abs(4.0 * (1.0 - 2.0 * x)))
        x = 4.0 * x * (1.0 - x)
    pred = n * math.log(2) + 0.5 * np.log(x * (1 - x)) - 0.5 * np.log(x0 * (1 - x0))
    assert np.median(np.abs(acc - pred)) < 1e-9                         # Rundung der Bahn, gemessen: Median 1,8·10⁻¹¹
    est = A.lyapunov_estimates(4.0, x0, (n,))[n]
    assert np.median(np.abs(est - math.log(2) - (pred - n * math.log(2)) / n)) < 1e-9
    assert abs(float(np.median(est)) - math.log(2)) < 1e-3              # der Median der Schätzungen liegt nahe ln 2 (Fehler ~ 1/N)


def test_telescope_identity_holds_on_50_random_orbits_with_n_10(zufall):
    rng = zufall(22)
    for _ in range(50):
        t = A.telescope(rng.uniform(0.05, 0.95), 10)
        assert t.sum_log == pytest.approx(t.formula, abs=1e-6)
        assert t.error == pytest.approx(t.error_formula, abs=1e-6)


def test_decimal_horizon_does_not_depend_on_the_number_of_digits():
    x0 = A.start_values(3, 12)
    for d0 in (1e-9, 1e-30):
        assert A.horizon_decimal(3.9, x0, d0, 0.1, digits=130).tolist() == A.horizon_decimal(3.9, x0, d0, 0.1, digits=200).tolist()


def test_float64_orbit_equals_decimal_for_large_initial_difference():
    """Zwei Bahnen im Abstand 10⁻³ (Float64, selbst gerechnet) haben denselben Horizont wie die Decimal-Rechnung (Rundung 10⁻¹⁶ ist dort ohne Wirkung): 40 Starts."""
    x0 = A.start_values(4, 40)
    dec = A.horizon_decimal(4.0, x0, 1e-3, 0.1)
    flo = []
    for x in x0:
        a, b, n = float(x), float(x) + 1e-3, 0
        while abs(a - b) <= 0.1 and n < 400:
            a, b, n = 4 * a * (1 - a), 4 * b * (1 - b), n + 1
        flo.append(n)
    assert np.max(np.abs(dec - np.array(flo))) <= 1 and np.mean(dec == np.array(flo)) >= 0.9


def test_two_orbit_equals_the_derivative_method_for_d0_1e9():
    x0 = A.start_values(5, 10)
    n = 20000
    der = A.lyapunov_estimates(3.9, x0, (n,))[n]
    zw = A.lyapunov_two_orbit(3.9, x0, 1e-9, n)
    assert np.all(np.isfinite(zw)) and np.max(np.abs(zw - der)) < 1e-5


def test_lambda_of_the_conjugated_quadratic_map_equals_the_logistic_one_step_by_step(zufall):
    """z = r(1/2 − x): z² + c mit c = r/2 − r²/4 gibt dieselbe Bahn, und 2z = r(1 − 2x): die Schritt-Exponenten ln|2z| und ln|r(1 − 2x)| sind gleich (50 Zufallsfälle, 15 Schritte; danach laufen die beiden Rundungswege auseinander)."""
    rng = zufall(23)
    for _ in range(50):
        r = rng.uniform(3.6, 3.99)
        x = rng.uniform(0.05, 0.95)
        z = r * (0.5 - x)
        c = r / 2 - r * r / 4
        for _ in range(15):
            assert 2 * z == pytest.approx(r * (1 - 2 * x), abs=1e-9)
            x, z = _f(r, x), z * z + c
            if abs(r * (0.5 - x) - z) > 1e-3:        # chaotische Bahnen laufen nach vielen Schritten durch Rundung auseinander: dann endet der Vergleich
                break
