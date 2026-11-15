"""Verdikte an von Hand gesetzten Werten (DEMO-PLAYBOOK §12): jede Zahl im Text und jede Schwelle der Art steht hier.

Ein Verdikt ist eine reine Funktion (Art, Text) in `lyc_evaluation.py`. Die Zahlen stammen aus handgesetzten Kennzahl-Strukturen und der Handrechnung
(H2: 6,12120307; H3), nicht aus einem Lauf des Verdikts."""

import numpy as np

import lyc_algorithm as A
import lyc_evaluation as E


def _curve(lam, rs, thr=0.003, n=6000):
    rs, lam = np.array(rs), np.array(lam)
    return E.CurveAnalysis(float(rs[0]), float(rs[-1]), thr, n, rs, lam, A.count_curve(rs, lam, thr), A.window_counts(rs, lam) if A.window_counts(rs, lam)[0] else None)


def test_curve_verdict_counts_above_below_and_window():
    rs = [3.0, 3.5, 3.6, 3.7, 3.83, 3.84, 3.9, 3.95]
    lam = [-0.5, -0.0003, 0.3, 0.4, -0.37, 0.2, 0.5, 0.0001]
    kind, text = E.verdict_curve(_curve(lam, rs))
    assert kind == "info"
    assert "Von 6 r-Werten oberhalb von 3,5699456 haben 4 (66,7 %) ein λ über 0,0030, 1 ein |λ| höchstens 0,0030 und 1 ein λ unter −0,0030" in text
    assert "Unterhalb davon (r ≥ 3,000) ist das größte λ −0,0003; 0 Werte liegen über der Schwelle." in text
    assert "Im Bereich 3,8284 bis 3,857 haben 1 von 2 r-Werten ein λ unter 0." in text
    assert "dieses Raster (8 r-Werte, N = 6000)" in text and "nicht für „fast alle r“" in text


def test_curve_verdict_without_values_above_the_onset_or_below():
    k, t = E.verdict_curve(_curve([-0.1, -0.2], [2.6, 3.0]))
    assert "oberhalb" not in t and "Unterhalb davon (r ≥ 2,600)" in t and "Im Bereich" not in t
    k, t = E.verdict_curve(_curve([0.3, 0.4], [3.7, 3.9]))
    assert "Von 2 r-Werten oberhalb von 3,5699456 haben 2 (100,0 %)" in t and "Unterhalb" not in t


def _horizon(r, steps, lam, exact=False, formula=None, tol=0.1, d0=1e-9, float_mean=52.35):
    steps = np.array(steps)
    return E.HorizonAnalysis(r, d0, tol, steps, lam, exact, formula, np.array([1e-9, 1e-8]), int(np.sum(steps < 400)), float_mean)


def test_horizon_verdict_names_mean_range_formula_difference_and_float64():
    steps = [26] * 50 + [28] * 49 + [32]                                          # Mittel 27,04
    h = _horizon(4.0, steps, np.log(2), True, 26.58)
    kind, text = E.verdict_horizon(h)
    assert kind == "success"
    assert "über 100 Starts: 27,0 Schritte (kleinster 26, größter 32)" in text and "Anfangsfehler 1·10⁻⁹, Toleranz 0,10, Decimal mit 130 Stellen" in text
    assert "λ = 0,6931 (exakt ln 2) gibt 26,6; Mittel minus Formel: 0,5 Schritte" in text
    assert "Float64-Bahn selbst im Mittel nach 52,4 Schritten" in text and "wirksamen Anfangsfehler von 1,7·10⁻¹⁷" in text and "2⁻⁵³ = 1,1·10⁻¹⁶" in text
    assert "nicht erreicht" not in text


def test_horizon_verdict_with_starts_that_do_not_reach_the_tolerance_and_with_estimated_lambda():
    steps = [30] * 95 + [400] * 5
    kind, text = E.verdict_horizon(_horizon(3.9, steps, 0.496, False, 37.1))
    assert kind == "success" and "über 95 Starts: 30,0 Schritte" in text and "(aus der Bahn geschätzt)" in text and "In 5 Starts wurde die Toleranz bis 400 Schritte nicht erreicht" in text


def test_horizon_verdict_without_growth():
    kind, text = E.verdict_horizon(_horizon(3.2, [400] * 100, -0.45, False, None))
    assert kind == "info" and "wächst der Abstand der beiden Bahnen nicht auf die Toleranz 0,10" in text and "λ ≈ −0,4500 ≤ 0" in text and "keinen Vorhersagehorizont" in text


def test_effective_initial_error_formula_and_limits():
    assert E.effective_initial_error(np.log(2), 0.1, 52.35) == 0.1 * np.exp(-np.log(2) * 52.35)
    assert E.effective_initial_error(-0.2, 0.1, 50.0) is None and E.effective_initial_error(0.0, 0.1, 50.0) is None


def test_decade_verdict_names_formula_and_measurement():
    kind, text = E.verdict_decade(E.DecadeAnalysis(4.0, 0.6931, 3.32, 3.3219))
    assert kind == "success" and "ln(10)/λ = 3,32 zusätzliche Schritte (λ = 0,6931)" in text and "3,32 Schritte je Zehnerpotenz" in text
    kind, text = E.verdict_decade(E.DecadeAnalysis(3.2, -0.45, None, None))
    assert kind == "info" and "λ = −0,4500 ≤ 0" in text


def _errors():
    def row(n, m, q):
        return A.ErrorRow(n, m, q, 0)

    r4 = [row(100, 3.3e-3, 1e-2), row(1000, 3.3e-4, 1e-3), row(10000, 3.3e-5, 1e-4)]
    r39 = [row(100, 2.7e-2, 4e-2), row(1000, 8.0e-3, 1.2e-2), row(10000, 2.5e-3, 3.8e-3)]
    return E.ErrorAnalysis(200, (100, 1000, 10000), r4, r39, -1.0, -0.5, -0.99, -0.5, 0.496)


def test_errors_verdict_names_slopes_live_and_frozen():
    frozen = {"n_starts": 3000, "r4": {"slope_median": -0.992, "slope_rms": -0.194}, "r39": {"slope_median": -0.503}, "zero_hits_r4": [[770, 62305]]}
    kind, text = E.verdict_errors(_errors(), frozen)
    assert kind == "info"
    assert "r = 4: Steigung −1,00 (≈ −1, also 1/N), r = 3,9: −0,50 (≈ −0,5, also 1/√N)" in text and "live an 200 Starts, N = 100, 1000, 10000" in text
    assert "Vorgerechnet an 3000 Starts bis N = 10⁵: −0,992 und −0,503" in text and "1 Bahn von 3000 in 10⁵ Schritten auf 0 fällt" in text and "Steigung −0,194" in text


def test_telescope_verdict_names_sum_formula_difference_and_error():
    kind, text = E.verdict_telescope(A.telescope(0.3, 10))                          # H2
    assert kind == "success"
    assert "Σ ln|f′| = 6,12120307 gegen" in text and "= 6,12120307 (Differenz" in text and "λ_N − ln 2 = −0,081027" in text and "= −0,081027: er fällt wie 1/N" in text


def test_two_orbit_verdict_all_finite_names_means_and_deviation():
    der = np.full(20, 0.49594)
    t = E.TwoOrbitAnalysis(3.9, 1e-9, 100000, der, der + 2.9e-7, 20, 0.49594029, 2.9e-7)
    kind, text = E.verdict_two_orbit(t)
    assert kind == "success" and "Bei d₀ = 1·10⁻⁹ sind alle 20 Ergebnisse endlich" in text and "Mittel Zweibahn 0,49594 gegen Ableitungsmethode 0,49594" in text and "2,9·10⁻⁷" in text and "N = 100000" in text


def test_number_formats():
    assert E.de(-0.25, 2) == "−0,25" and E.de_sci(3.17e-6, 2) == "3,17·10⁻⁶" and E.de_sci(1.586e17, 1) == "1,6·10¹⁷" and E.de_sci(0, 1) == "0"


def test_horizon_verdict_with_positive_lambda_and_tolerance_above_the_attractor_width_does_not_claim_shrinking_errors():
    """Fehler der Textlogik (Prüfbericht): r = 3,9 (λ ≈ 0,4960 > 0), Toleranz 0,9 über der Breite 0,88 gab „λ ≤ 0 … kleine Fehler schrumpfen“."""
    h = _horizon(3.9, [400] * 100, 0.4960, False, 41.56, tol=0.9)
    h = E.HorizonAnalysis(*[getattr(h, f) for f in ("r", "d0", "tol", "steps", "lam", "lam_exact", "formula", "path", "reached", "float_mean")], extent=0.88)
    kind, text = E.verdict_horizon(h)
    assert kind == "info" and "λ ≈ 0,4960 > 0" in text and "Breite der Bahn" in text and "0,88" in text and "Eine kleinere Toleranz wählen" in text
    assert "≤ 0" not in text and "schrumpfen" not in text and "keinen Vorhersagehorizont" not in text


def test_horizon_verdict_with_positive_lambda_and_too_few_steps_names_the_formula_and_no_shrinking():
    h = _horizon(3.57, [400] * 100, 0.0111, False, 6005.2, tol=0.1, d0=1e-30)
    h = E.HorizonAnalysis(*[getattr(h, f) for f in ("r", "d0", "tol", "steps", "lam", "lam_exact", "formula", "path", "reached", "float_mean")], extent=0.55)
    kind, text = E.verdict_horizon(h)
    assert kind == "info" and "λ ≈ 0,0111 > 0" in text and "ergibt 6005 Schritte" in text and "Breite der Bahn" not in text
    assert "≤ 0" not in text and "schrumpfen" not in text
