"""Eigenschaftstests (ORACLE-PLAYBOOK §5.2): Aussagen, die für die Verfahren gelten müssen, ohne dass man das Ergebnis kennt.

Je Eigenschaft 30 bis 50 Zufallsinstanzen mit festem Zufalls-Seed (SplitMix64). Jede Eigenschaft nennt ihre Voraussetzung."""

import math

import numpy as np
import pytest

import lyc_algorithm as A
import lyc_evaluation as E


def test_lambda_does_not_depend_on_the_start_within_the_statistical_band():
    """30 Starts bei r = 3,9, N = 20000: alle Schätzungen liegen innerhalb ±0,012 um ihr Mittel (Streuung der Einzelschätzung etwa 2,7·10⁻³ bei dieser Länge, also ±4,5 Standardabweichungen)."""
    x0 = A.start_values(61, 30)
    est = A.lyapunov_estimates(3.9, x0, (20000,), transient=3000)[20000]
    assert np.max(np.abs(est - est.mean())) < 0.012
    assert abs(est.mean() - 0.496) < 0.004


def test_telescope_identity_for_50_random_orbits_with_n_10(zufall):
    rng = zufall(62)
    for _ in range(50):
        t = A.telescope(rng.uniform(0.05, 0.95), 10)
        assert abs(t.sum_log - t.formula) < 1e-6


def test_conjugation_maps_logistic_orbits_to_tent_orbits_for_a_few_steps(zufall):
    """θ(f(x)) = T(θ(x)) für 50 Zufallsstarts, 6 Schritte (danach vergrößert sich der Rundungsfehler um den Faktor 2 je Schritt über 10⁻¹²)."""
    rng = zufall(63)
    for _ in range(50):
        x = rng.uniform(0.05, 0.95)
        th = float(A.to_theta(x))
        for _ in range(6):
            x = 4 * x * (1 - x)
            th = float(A.tent(th))
            assert float(A.to_theta(x)) == pytest.approx(th, abs=1e-9)


def test_horizon_grows_monotonically_with_falling_initial_error_on_average():
    """Mittel über 40 Starts: je kleiner δ₀, desto größer der Horizont (r = 4, 3,9, 3,7; δ₀ = 10⁻³, 10⁻⁶, 10⁻⁹, 10⁻¹²)."""
    x0 = A.start_values(64, 40)
    for r in (4.0, 3.9, 3.7):
        means = [float(A.horizon_decimal(r, x0, 10.0 ** -e, 0.1).mean()) for e in (3, 6, 9, 12)]
        assert all(means[i] < means[i + 1] for i in range(3)), (r, means)


def test_horizon_grows_with_the_tolerance():
    x0 = A.start_values(65, 40)
    means = [float(A.horizon_decimal(4.0, x0, 1e-9, tol).mean()) for tol in (0.01, 0.1, 0.5)]
    assert means[0] < means[1] < means[2]


def test_same_seed_gives_the_same_output():
    assert np.array_equal(A.start_values(7, 20), A.start_values(7, 20)) and not np.array_equal(A.start_values(7, 20), A.start_values(8, 20))
    assert E.analyse_horizon(3.9, 9, 0.1, 5).steps.tolist() == E.analyse_horizon(3.9, 9, 0.1, 5).steps.tolist()
    a, b = E.analyse_two_orbit(3.9, 9, 5000, 5), E.analyse_two_orbit(3.9, 9, 5000, 5)
    assert np.array_equal(a.derivative, b.derivative) and np.array_equal(a.two_orbit, b.two_orbit, equal_nan=True)


def test_exponent_does_not_change_when_the_start_moves_along_the_orbit():
    """λ_N einer Bahn ab Schritt 1000 (Startpunkt = x₁₀₀₀) und die Bahn selbst geben bei N → ∞ dasselbe; bei N = 20000 unterscheiden sie sich nur um die Endlichkeit (≤ 0,01, r = 3,9, 20 Starts)."""
    x0 = A.start_values(66, 20)
    r = 3.9
    a = A.lyapunov_estimates(r, x0, (20000,), transient=0)[20000]
    b = A.lyapunov_estimates(r, x0, (20000,), transient=1000)[20000]
    assert np.max(np.abs(a - b)) < 0.01


def test_stable_cycle_exponent_is_negative_and_equals_log_of_multiplier_over_period(zufall):
    """Für 30 zufällige r im Periode-2-Bereich (3,05 bis 3,40): λ = ½·ln|4 + 2r − r²| < 0 (Schätzung aus der Bahn gleich auf 10⁻⁵); Werte mit |Mehrfachwert| < 0,02 (nahe dem superstabilen Punkt 1 + √5, wo λ → −∞) werden übersprungen."""
    rng = zufall(67)
    for _ in range(30):
        r = rng.uniform(3.05, 3.40)
        m = 4 + 2 * r - r * r
        if abs(m) < 0.02:
            continue
        exact = 0.5 * math.log(abs(m))
        assert exact < 0
        assert float(A.lambda_curve(np.array([r]), 30000, 3000)[0]) == pytest.approx(exact, abs=1e-5), r
