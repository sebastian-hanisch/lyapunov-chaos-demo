"""Randfälle, an denen Kernroutinen erfahrungsgemäß brechen (DEMO-PLAYBOOK §12): x₀ ∈ {0, 1/2, 1} bei r = 4 (Nullwächter), r = 3 genau, r < 3, N = 1, d₀ nahe der Maschinengenauigkeit,
Verdikt-Schwelle an Randwerten, Horizont ohne Wachstum, Regler-Ränder."""

import math

import numpy as np
import pytest

import lyc_algorithm as A
import lyc_constants as C
import lyc_evaluation as E


# ---------------------------------------------------------------- Starts 0, 1/2, 1 bei r = 4
def test_start_one_half_triggers_the_zero_guard_exactly_once_and_the_orbit_collapses():
    """0,5 → 1 → 0 → 0 …: f′(0,5) = 0 genau, also log(0): der Wächter setzt |f′| auf 10⁻³⁰⁰ und meldet den Fall; danach liegt die Bahn im Fixpunkt 0 mit f′ = 4, die Schätzung nähert sich ln 4."""
    lam, guard = A.lyapunov_single(4.0, 0.5, 100)
    assert guard
    assert lam == pytest.approx((math.log(A.LOG_FLOOR) + 99 * math.log(4)) / 100, rel=1e-9)      # erster Schritt: Wächter, die übrigen 99: ln 4
    assert lam < math.log(4)
    est = A.lyapunov_estimates(4.0, np.array([0.5]), (100,))[100][0]
    assert est == pytest.approx(lam, abs=1e-9)


def test_start_zero_and_one_stay_at_zero_with_exponent_ln_4():
    for x0 in (0.0, 1.0):
        lam, guard = A.lyapunov_single(4.0, x0, 50)
        assert not guard and lam == pytest.approx(math.log(4), abs=1e-12)


def test_telescope_at_the_critical_start_reports_the_boundary():
    t = A.telescope(0.5, 5)
    assert t.xn == 0.0 and not np.isfinite(t.formula)
    kind, text = E.verdict_telescope(t)
    assert kind == "warning" and "auf dem Rand" in text


# ---------------------------------------------------------------- r = 3 und r < 3
def test_r_exactly_three_has_a_small_exponent_and_no_decision_in_either_direction():
    """Bei r = 3 geht die Bahn nur langsam (algebraisch) gegen den Fixpunkt: λ ≈ 0 (|λ| < 0,02 bei N = 30000), die Schwelle 0,003 entscheidet es nicht."""
    lam = float(A.lambda_curve(np.array([3.0]), 30000, 3000)[0])
    assert abs(lam) < 0.02


def test_r_below_three_is_negative_with_the_exact_value():
    assert float(A.lambda_curve(np.array([2.6]), 5000, 1000)[0]) == pytest.approx(math.log(0.6), abs=1e-6)


def test_n_equals_one_is_the_log_of_the_derivative_at_the_start():
    lam, guard = A.lyapunov_single(3.9, 0.3, 1)
    assert not guard and lam == pytest.approx(math.log(abs(3.9 * (1 - 0.6))))
    assert A.lyapunov_estimates(3.9, np.array([0.3]), (1,))[1][0] == pytest.approx(lam)


# ---------------------------------------------------------------- Zweibahn und Maschinengenauigkeit
def test_two_orbit_with_d0_near_machine_precision_may_be_non_finite_but_does_not_raise():
    x0 = A.start_values(1, 20)
    zw = A.lyapunov_two_orbit(3.9, x0, 1e-15, 2000)
    assert zw.shape == (20,)
    zw12 = A.lyapunov_two_orbit(3.9, x0, 1e-12, 20000)
    assert np.sum(np.isfinite(zw12)) < 20           # bei d₀ = 10⁻¹² (N = 10⁴ bis 10⁵) ist ein Teil der Ergebnisse nicht endlich


def test_two_orbit_verdict_distinguishes_all_some_and_none_finite():
    base = E.analyse_two_orbit(3.9, 9, 3000, 5)
    assert E.verdict_two_orbit(base)[0] == "success"
    some = E.TwoOrbitAnalysis(3.9, 1e-12, 100000, np.ones(20), np.array([0.5] * 6 + [float("nan")] * 14), 6, 0.5, 1e-5)
    k, t = E.verdict_two_orbit(some)
    assert k == "warning" and "nur 6 von 20" in t
    none = E.TwoOrbitAnalysis(3.9, 1e-12, 100000, np.ones(20), np.full(20, float("nan")), 0, None, None)
    assert E.verdict_two_orbit(none)[0] == "warning" and "keinem der 20" in E.verdict_two_orbit(none)[1]


# ---------------------------------------------------------------- Schwelle
def test_threshold_is_exclusive_for_positive_and_inclusive_for_near_zero():
    rs = np.array([3.6, 3.7, 3.8, 3.9, 3.95])
    lam = np.array([0.003, 0.0030001, -0.003, -0.0030001, 0.1])
    cc = A.count_curve(rs, lam, 0.003)
    assert (cc.n_above, cc.pos, cc.zero, cc.neg) == (5, 2, 2, 1)       # +0,003 und −0,003 gelten als nahe null, 0,0030001 als positiv, −0,0030001 als negativ
    assert cc.max_below is None and cc.n_below_pos == 0


def test_count_below_the_onset_is_reported_separately():
    rs = np.array([2.6, 3.2, 3.6])
    cc = A.count_curve(rs, np.array([-0.2, -0.9, 0.3]), 0.003)
    assert cc.n_above == 1 and cc.pos == 1 and cc.max_below == pytest.approx(-0.2) and cc.n_below_pos == 0


def test_window_counts_only_inside_the_window():
    rs = np.array([3.8, 3.83, 3.85, 3.9])
    assert A.window_counts(rs, np.array([0.4, -0.37, 0.2, 0.5])) == (2, 1)
    assert A.window_counts(np.array([3.0, 3.1]), np.array([0.0, 0.0])) == (0, 0)


def test_range_with_the_upper_slider_below_the_lower_has_the_minimum_width():
    assert E.effective_range(3.9, 3.8) == (3.9, 3.9 + 0.005)
    c = E.analyse_curve(3.9, 3.8, C.E1_NCOLS_MIN, 1000, 0.003)
    assert c.hi - c.lo == pytest.approx(0.005) and c.window is None
    assert E.verdict_curve(c)[0] == "info" and "Von 20" in E.verdict_curve(c)[1]


# ---------------------------------------------------------------- Horizont ohne Wachstum, Regler-Ränder
def test_horizon_without_growth_at_r_3_2_has_no_horizon():
    h = E.analyse_horizon(3.2, 9, 0.1, 5)
    assert h.reached == 0 and h.formula is None and h.lam < 0
    kind, text = E.verdict_horizon(h)
    assert kind == "info" and "keinen Vorhersagehorizont" in text


def test_horizon_at_the_slider_extremes():
    for exp in (C.E2_EXP_MIN, C.E2_EXP_MAX):
        for tol in (C.E2_TOL_MIN, C.E2_TOL_MAX):
            h = E.analyse_horizon(4.0, exp, tol, 5)
            assert h.reached == C.E2_STARTS and np.all(h.steps >= 1)


def test_decade_verdict_for_a_non_chaotic_r_and_for_r_4():
    d = E.analyse_decade(3.2, 5)
    assert d.formula is None and E.verdict_decade(d)[0] == "info"
    d4 = E.analyse_decade(4.0, 5)
    assert d4.formula == pytest.approx(math.log(10) / math.log(2)) and abs(d4.per_decade - d4.formula) < 0.3


def test_slider_extremes_of_the_two_orbit_experiment():
    for exp in (C.E5_EXP_MIN, C.E5_EXP_MAX):
        t = E.analyse_two_orbit(3.9, exp, C.E5_N_MIN, 5)
        assert t.finite in range(0, 21)
        assert E.verdict_two_orbit(t)[0] in ("success", "warning")


def test_curve_at_the_range_extremes_has_a_verdict():
    c = E.analyse_curve(C.R_MIN, C.R_MAX, C.E1_NCOLS_MIN, C.E1_N_MIN, C.E1_THR_MAX)
    assert E.verdict_curve(c)[0] == "info" and c.counts.n_above + int(np.sum(c.rs <= C.ONSET)) == C.E1_NCOLS_MIN


# ---------------------------------------------------------------- Horizont: Toleranz über der Breite des Attraktors
def test_default_r_with_tolerance_0_9_gives_no_shrinking_verdict_but_names_the_attractor_width():
    """Voreinstellung r = 3,9, δ₀ = 10⁻⁹, Toleranz 0,9 (Reglermaximum), Zufalls-Seed 2027: λ > 0, aber kein Start erreicht die Toleranz, weil sie über der Breite der Bahn (0,88) liegt."""
    h = E.analyse_horizon(3.9, 9, 0.9, 2027)
    assert h.reached == 0 and h.lam > 0.49 and h.extent == pytest.approx(0.88, abs=0.005) and h.tol > h.extent
    kind, text = E.verdict_horizon(h)
    assert "λ ≈ 0,4960 > 0" in text and "Breite der Bahn" in text and "schrumpfen" not in text and "≤ 0" not in text


def test_r_3_68_and_r_3_57_with_unreached_tolerance_do_not_claim_shrinking_either():
    h = E.analyse_horizon(3.68, 9, 0.9, 2027)
    assert h.reached == 0 and h.lam > 0.3 and h.extent == pytest.approx(0.65, abs=0.005)
    assert "schrumpfen" not in E.verdict_horizon(h)[1] and "Breite der Bahn" in E.verdict_horizon(h)[1]
    h = E.analyse_horizon(3.57, 30, 0.1, 2027)          # Horizont (Formel) über 400 Schritte
    assert h.reached == 0 and 0 < h.lam < 0.1 and h.formula > C.HORIZON_NMAX and h.tol < h.extent
    t = E.verdict_horizon(h)[1]
    assert "schrumpfen" not in t and "Formel" in t
