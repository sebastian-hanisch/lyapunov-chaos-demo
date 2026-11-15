"""Claims-Tests: jede Zahl der Tabelle „Befunde“ in der README wird hier nachgerechnet (DEMO-PLAYBOOK §11a).

Die schweren Messungen (3000 Starts bis 10⁵ Schritte, 3000 und 6000 r-Werte mit 33000 Schritten) sind vorgerechnet (`tools/messung_schaetzfehler.py`, `data/messungen.json`);
die Tests prüfen die README-Zahlen gegen die Datei und rechnen so viel davon neu, wie in einigen Sekunden geht (gleiche Zufalls-Seeds, daher gleiche Zahlen).
Die README-Tabelle „Zahl → Test“ nennt für jede Zahl den Test und die Art der Prüfung (Band oder exakt)."""

import math

import numpy as np
import pytest

import lyc_algorithm as A
import lyc_evaluation as E

F = E.load_frozen()


# ---------------------------------------------------------------- λ(r) mit Fenstern (vorgerechnet, neu gerechnet)
def test_claim_lambda_raster_counts_and_recomputation():
    """3000 r-Werte von 2,5 bis 3,9999, Einschwingen 3000, N = 30000: oberhalb 3,5699456 liegen 860 Werte, 773 (89,9 %) mit λ > 0,003, 3 nahe null, 84 negativ."""
    k = F["kurve"]
    assert (k["oberhalb"], k["pos"], k["nahe_null"], k["neg"]) == (860, 773, 3, 84)
    assert 100 * k["pos"] / k["oberhalb"] == pytest.approx(89.9, abs=0.05) and k["pos"] + k["nahe_null"] + k["neg"] == k["oberhalb"]
    rs = np.linspace(2.5, 3.9999, 3000)
    lam = A.lambda_curve(rs, 30000, 3000)
    cc = A.count_curve(rs, lam, 0.003)
    assert (cc.n_above, cc.pos, cc.zero, cc.neg, cc.n_below_pos) == (860, 773, 3, 84, 0)
    assert cc.max_below == pytest.approx(-0.0003, abs=5e-5) and k["max_unterhalb"] == pytest.approx(-0.0003, abs=5e-5)
    assert A.window_counts(rs, lam) == (57, 44) and (k["fenster_n"], k["fenster_neg"]) == (57, 44)
    assert float(lam.max()) == pytest.approx(0.6886, abs=5e-5) and float(rs[int(np.argmax(lam))]) == pytest.approx(3.9999)


def test_claim_lambda_raster_with_twice_the_density():
    """Gegenprobe mit 6000 r-Werten: 1540 von 1720 (89,5 %) mit λ > 0,003, 6 nahe null, 174 negativ (gegen 89,9 % auf dem Raster mit 3000)."""
    k = F["kurve6000"]
    assert (k["oberhalb"], k["pos"], k["nahe_null"], k["neg"]) == (1720, 1540, 6, 174)
    assert 100 * k["pos"] / k["oberhalb"] == pytest.approx(89.5, abs=0.05)
    assert abs(100 * k["pos"] / k["oberhalb"] - 100 * F["kurve"]["pos"] / F["kurve"]["oberhalb"]) < 0.5


def test_claim_exact_exponents_of_stable_cycles():
    """r = 2,8: −0,223144; 3,2: −0,916291; 3,83: −0,369674 (Multiplikator +0,329882), geschätzt gleich auf 10⁻⁹."""
    exact = {"2.8": (-0.223144, 1), "3.2": (-0.916291, 2), "3.83": (-0.369674, 3)}
    for r, (value, period) in exact.items():
        row = F["kurve"]["exakt"][r]
        assert row["periode"] == period and row["lambda_exakt"] == pytest.approx(value, abs=5e-7) and row["lambda_geschaetzt"] == pytest.approx(row["lambda_exakt"], abs=1e-9)
    assert F["kurve"]["exakt"]["3.83"]["multiplikator"] == pytest.approx(0.329882, abs=5e-7)


# ---------------------------------------------------------------- Schätzfehler gegen N
def test_claim_estimation_error_medians_and_slopes():
    """3000 Starts: Median |Fehler| bei N = 10², 10³, 10⁴, 10⁵: r = 4: 3,26·10⁻³, 3,22·10⁻⁴, 3,33·10⁻⁵, 3,42·10⁻⁶ (Steigung −0,992); r = 3,9: 2,68·10⁻², 7,92·10⁻³, 2,55·10⁻³, 8,23·10⁻⁴ (Steigung −0,503)."""
    f = F["fehler"]
    assert f["r4"]["median_abs"] == pytest.approx([3.26e-3, 3.22e-4, 3.33e-5, 3.42e-6], rel=5e-3)
    assert f["r39"]["median_abs"] == pytest.approx([2.68e-2, 7.92e-3, 2.55e-3, 8.23e-4], rel=5e-3)
    assert f["r4"]["slope_median"] == pytest.approx(-0.992, abs=5e-4) and f["r39"]["slope_median"] == pytest.approx(-0.503, abs=5e-4)
    assert f["r39"]["slope_rms"] == pytest.approx(-0.500, abs=5e-4) and f["r4"]["slope_rms"] == pytest.approx(-0.194, abs=5e-4)


def test_claim_estimation_errors_recomputed_up_to_n_10000():
    x0 = A.start_values(31, 3000)
    est4 = A.lyapunov_estimates(4.0, x0, (100, 1000, 10000))
    rows = A.estimation_errors(est4, A.LN2)
    assert [w.median_abs for w in rows] == pytest.approx(F["fehler"]["r4"]["median_abs"][:3], rel=1e-12)
    est39 = A.lyapunov_estimates(3.9, x0, (100, 1000, 10000))
    rows39 = A.estimation_errors(est39, F["fehler"]["ref_39"])
    assert [w.median_abs for w in rows39] == pytest.approx(F["fehler"]["r39"]["median_abs"][:3], rel=1e-12)


def test_claim_over_1e3_counts_and_the_one_orbit_that_falls_on_zero():
    f = F["fehler"]
    assert f["r4"]["over_1e3"] == [2307, 572, 1, 1] and f["r39"]["over_1e3"] == [2951, 2788, 2380, 1245]
    assert f["zero_hits_r4"] == [[770, 62305]]
    x = float(A.start_values(31, 3000)[770])
    first = None
    for i in range(1, 70000):
        x = 4.0 * x * (1.0 - x)
        if x == 0.0:
            first = i
            break
    assert first == 62305                                                      # bis Schritt 62304 nie 0, bei 62305 exakt 0
    assert f["r4"]["rms"][3] > 20 * f["r4"]["rms"][2]                          # das quadratische Mittel springt bei N = 10⁵ (eine Bahn auf 0)


def test_claim_reference_for_r_3_9_and_lambda_37():
    f = F["fehler"]
    assert f["ref_39"] == pytest.approx(0.49602, abs=5e-6) and f["ref_39_stderr"] == pytest.approx(2.3e-5, abs=5e-7)
    assert f["lambda_37"] == pytest.approx(0.35495, abs=5e-6) and f["lambda_37_stderr"] == pytest.approx(1.8e-5, abs=5e-7)
    live = A.lyapunov_estimates(3.7, A.start_values(31, 200), (20000,))[20000]
    assert float(live.mean()) == pytest.approx(f["lambda_37"], abs=0.003)


def test_claim_lambda_4_after_a_million_steps_equals_ln_2_plus_the_boundary_term():
    """x₀ = 0,3, N = 10⁶: λ_N − ln 2 = [ln(x_N(1 − x_N)) − ln(x₀(1 − x₀))]/(2N) = −1,49·10⁻⁶ (Rest der Teleskopformel unter 10⁻⁸)."""
    t = A.telescope(0.3, 10 ** 6)
    assert t.error == pytest.approx(-1.49e-6, abs=5e-9) and t.error == pytest.approx(t.error_formula, abs=1e-8)


# ---------------------------------------------------------------- Vorhersagehorizont
def test_claim_horizon_table_matches_the_file_and_recomputation():
    """Mittlerer Horizont, Decimal, Toleranz 0,1, 100 Starts, δ₀ = 10⁻³ / 10⁻⁶ / 10⁻⁹ / 10⁻¹² / 10⁻¹⁵ / 10⁻³⁰: r = 4: 7,6 / 17,5 / 27,5 / 37,3 / 47,4 / 97,4; r = 3,9: 9,9 / 23,7 / 37,5 / 51,2 / 66,1 / 136,4; r = 3,7: 12,8 / 32,9 / 51,8 / 70,8 / 90,4 / 188,9."""
    h = F["horizont"]
    exp = {"4.0": [7.55, 17.46, 27.48, 37.3, 47.39, 97.39], "3.9": [9.94, 23.73, 37.54, 51.16, 66.06, 136.42], "3.7": [12.82, 32.88, 51.84, 70.8, 90.41, 188.91]}
    for r, values in exp.items():
        assert [h[r][e]["mittel"] for e in ("3", "6", "9", "12", "15", "30")] == pytest.approx(values, abs=5e-3)
    x0 = A.start_values(4, 100)
    for r, e in ((4.0, 9), (4.0, 15), (3.9, 9), (3.7, 6)):
        steps = A.horizon_decimal(r, x0, 10.0 ** -e, 0.1)
        row = h[str(r)][str(e)]
        assert (float(steps.mean()), int(steps.min()), int(steps.max())) == (row["mittel"], row["min"], row["max"]), (r, e)
    assert (h["4.0"]["9"]["min"], h["4.0"]["9"]["max"]) == (26, 32) and (h["4.0"]["15"]["min"], h["4.0"]["15"]["max"]) == (46, 51)
    assert (h["3.9"]["9"]["min"], h["3.9"]["9"]["max"]) == (29, 57)


def test_claim_horizon_against_the_formula_for_r_4_and_3_9():
    """Mittel minus ln(Toleranz/δ₀)/λ (λ = ln 2 bzw. 0,49602): r = 4: 0,76 bis 1,05 Schritte über die sechs δ₀, r = 3,9: 0,10 bis 1,80."""
    h = F["horizont"]
    for r, lam, lo, hi in (("4.0", math.log(2), 0.76, 1.05), ("3.9", F["fehler"]["ref_39"], 0.10, 1.80)):
        dev = [h[r][e]["mittel"] - math.log(0.1 / 10.0 ** -int(e)) / lam for e in ("3", "6", "9", "12", "15", "30")]
        assert min(dev) == pytest.approx(lo, abs=0.01) and max(dev) == pytest.approx(hi, abs=0.01), r
        assert all(d > 0 for d in dev)


def test_claim_steps_per_decade():
    """Je Zehnerpotenz kleineres δ₀ (Mittel zwischen 10⁻⁶ und 10⁻¹²): r = 4: 3,31 Schritte gegen ln(10)/ln 2 = 3,32; r = 3,9: 4,57 gegen ln(10)/0,49602 = 4,64."""
    h = F["horizont"]
    for r, lam, per in (("4.0", math.log(2), 3.31), ("3.9", F["fehler"]["ref_39"], 4.57)):
        measured = (h[r]["12"]["mittel"] - h[r]["6"]["mittel"]) / 6
        assert measured == pytest.approx(per, abs=0.005) and abs(measured - math.log(10) / lam) < 0.1
    assert math.log(10) / math.log(2) == pytest.approx(3.32, abs=0.005)


def test_claim_tolerance_dependence_at_r_4():
    t = F["horizont"]["tol_r4_d09"]
    assert [t[k]["mittel"] for k in ("0.01", "0.1", "0.5")] == pytest.approx([24.23, 27.48, 30.27], abs=5e-3)
    assert [math.log2(tol / 1e-9) for tol in (0.01, 0.1, 0.5)] == pytest.approx([23.3, 26.6, 28.9], abs=0.05)


def test_claim_float64_orbit_horizon_and_effective_initial_error():
    """Float64 gegen Decimal (100 Starts, Toleranz 0,1, r = 4): mittlerer Horizont 52,66 (44 bis 59); nach der Formel entspricht das δ₀ = 0,1·e^(−ln 2·52,66) = 1,4·10⁻¹⁷; Interpolation in der Tabelle (log-linear zwischen 10⁻¹⁵ und 10⁻³⁰) gibt 10^(−16,6) = 2,6·10⁻¹⁷; beide unter 2⁻⁵³ = 1,1·10⁻¹⁶."""
    steps = A.horizon_float(4.0, A.start_values(4, 100), 0.1)
    assert float(steps.mean()) == pytest.approx(52.66, abs=5e-3) and (int(steps.min()), int(steps.max())) == (44, 59)
    formula = E.effective_initial_error(math.log(2), 0.1, float(steps.mean()))
    assert formula == pytest.approx(1.4e-17, rel=0.03)
    h = F["horizont"]["4.0"]
    exp_eff = 15 + (float(steps.mean()) - h["15"]["mittel"]) / (h["30"]["mittel"] - h["15"]["mittel"]) * 15
    assert 10.0 ** -exp_eff == pytest.approx(2.6e-17, rel=0.03)
    assert formula < 2.0 ** -53 and 10.0 ** -exp_eff < 2.0 ** -53


# ---------------------------------------------------------------- Zweibahn-Verfahren
def test_claim_two_orbit_counts_and_deviations():
    z = F["zweibahn"]
    assert [z["3.9"]["zeilen"][k]["endlich"] for k in ("0.001", "1e-06", "1e-09", "1e-12")] == [20, 20, 20, 6]
    assert [z["4.0"]["zeilen"][k]["endlich"] for k in ("1e-06", "1e-09", "1e-12")] == [20, 20, 8]
    assert z["3.9"]["zeilen"]["1e-09"]["max_abw"] == pytest.approx(2.0e-7, rel=0.03) and z["3.9"]["zeilen"]["1e-06"]["max_abw"] == pytest.approx(9.0e-6, rel=0.03)
    assert z["3.9"]["zeilen"]["0.001"]["max_abw"] == pytest.approx(3.1e-4, rel=0.03) and z["4.0"]["zeilen"]["1e-09"]["max_abw"] == pytest.approx(2.8e-5, rel=0.03)
    assert z["3.9"]["ableitung_mittel"] == pytest.approx(0.49599, abs=5e-6) and z["4.0"]["ableitung_mittel"] == pytest.approx(0.69314, abs=5e-6)


def test_claim_two_orbit_recomputed_for_d0_1e9_and_1e12():
    x0 = A.start_values(41, 20)
    n = 100000
    der = A.lyapunov_estimates(3.9, x0, (n,))[n]
    for d0, finite in ((1e-9, 20), (1e-12, 6)):
        zw = A.lyapunov_two_orbit(3.9, x0, d0, n)
        assert int(np.sum(np.isfinite(zw))) == finite, d0
    zw9 = A.lyapunov_two_orbit(3.9, x0, 1e-9, n)
    assert float(np.max(np.abs(zw9 - der))) == pytest.approx(F["zweibahn"]["3.9"]["zeilen"]["1e-09"]["max_abw"], rel=1e-9)
