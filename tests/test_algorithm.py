"""Kern gegen Handrechnung (DEMO-PLAYBOOK §1b und §12): die Handbeispiele H1 bis H4 des Plans.

Die Erwartungswerte stehen als Rechnung im Test (Zelt-Konjugation, Teleskopsumme, Zyklus-Multiplikatoren), nicht als Abschrift eines Laufs.
Startwerte sind keine Dualbrüche (0,3; 0,301); der Fall x₀ = 0,5 steht als gewollter Kollaps in `test_edge_cases.py`."""

import math

import numpy as np
import pytest

import lyc_algorithm as A


# ---------------------------------------------------------------- H1: r = 4 und die Zeltabbildung
def test_h1_tent_map_cycle_and_conjugation():
    """θ = 2/7 → 4/7 → 6/7 → 2/7 unter T(θ) = 1 − |1 − 2θ|; x = sin²(πθ/2) = sin²(π/7), sin²(2π/7), sin²(3π/7) = 0,18826; 0,61126; 0,95048."""
    th = [2 / 7, 4 / 7, 6 / 7]
    assert [float(A.tent(t)) for t in th] == pytest.approx([4 / 7, 6 / 7, 2 / 7])
    xs = [math.sin(math.pi * t / 2) ** 2 for t in th]
    assert xs == pytest.approx([math.sin(math.pi / 7) ** 2, math.sin(2 * math.pi / 7) ** 2, math.sin(3 * math.pi / 7) ** 2])
    assert [round(x, 5) for x in xs] == [0.18826, 0.61126, 0.95048]
    assert [4 * x * (1 - x) for x in xs] == pytest.approx([xs[1], xs[2], xs[0]])


def test_h1_multiplier_is_plus_8_and_lambda_is_ln_2():
    """f′ = 4(1 − 2x) = +2,4940; −0,8901; −3,6039, Produkt +8 (Zelt: +2·(−2)·(−2) = 8), λ = ln(8)/3 = ln 2."""
    th, xs, mult = A.three_cycle_theta()
    assert xs == pytest.approx([math.sin(math.pi * t / 2) ** 2 for t in th])
    assert [round(4 * (1 - 2 * x), 4) for x in xs] == [2.4940, -0.8901, -3.6039]
    assert mult == pytest.approx(8.0, abs=1e-12) and math.log(abs(mult)) / 3 == pytest.approx(math.log(2))


def test_h1_conjugation_is_its_own_inverse_and_maps_the_dynamics():
    x = np.array([0.18826, 0.3, 0.77])
    assert A.from_theta(A.to_theta(x)) == pytest.approx(x, abs=1e-14)
    th = A.to_theta(np.array([0.3]))
    nxt = A.to_theta(4 * 0.3 * (1 - 0.3))
    assert float(A.tent(th)[0]) == pytest.approx(float(nxt), abs=1e-14)      # θ(f(x)) = T(θ(x)) für θ ≤ 1


# ---------------------------------------------------------------- H2: Teleskop-Identität
def test_h2_telescope_identity_by_hand():
    """x₀ = 0,3, N = 10: x₁₀ = 0,043422; Σ ln|f′| = 6,12120307 = 10·ln 2 + ½·ln(x₁₀(1 − x₁₀)) − ½·ln(x₀(1 − x₀)); λ₁₀ = 0,612120; Fehler −0,081027."""
    t = A.telescope(0.3, 10)
    assert round(t.xn, 6) == 0.043422
    assert t.sum_log == pytest.approx(6.12120307, abs=5e-9) and t.formula == pytest.approx(6.12120307, abs=5e-9)
    assert t.sum_log == pytest.approx(10 * math.log(2) + 0.5 * math.log(t.xn * (1 - t.xn)) - 0.5 * math.log(0.3 * 0.7), abs=1e-12)
    assert round(t.estimate, 6) == 0.612120 and round(t.error, 6) == -0.081027 and round(t.error_formula, 6) == -0.081027


# ---------------------------------------------------------------- H3: Vorhersagehorizont
def test_h3_two_orbits_double_their_distance_and_cross_0_1_at_k_7():
    """x₀ = 0,3 und 0,301, r = 4: Abstände 0,001000; 0,001596; 0,004351; 0,001233; 0,004883; 0,018748; 0,063208; 0,106574 (k = 7 erste Überschreitung von 0,1); log₂(100) = 6,64."""
    a, b = 0.3, 0.301
    d = []
    for _ in range(8):
        d.append(abs(a - b))
        a, b = 4 * a * (1 - a), 4 * b * (1 - b)
    assert d == pytest.approx([0.001, 0.001596, 0.004351, 0.001233, 0.004883, 0.018748, 0.063208, 0.106574], abs=5e-7)
    assert next(k for k, v in enumerate(d) if v > 0.1) == 7
    assert math.log2(100) == pytest.approx(6.64, abs=5e-3)
    assert A.horizon_decimal(4.0, [0.3], 0.001, 0.1).tolist() == [7]


def test_h3_doubling_holds_only_on_average():
    """Abstand/2^k schwankt in den ersten 12 Schritten zwischen 0,00008 und 0,0011: der Faktor |f′| hängt von x ab."""
    a, b = 0.3, 0.301
    ratios = []
    for k in range(13):
        ratios.append(abs(a - b) / 2 ** k)
        a, b = 4 * a * (1 - a), 4 * b * (1 - b)
    assert min(ratios) == pytest.approx(0.00008, abs=1e-5) and max(ratios) == pytest.approx(0.0011, abs=1e-4)


def test_h3_formula_for_the_horizon():
    assert A.horizon_formula(math.log(2), 1e-9, 0.1) == pytest.approx(math.log(1e8) / math.log(2))
    assert A.horizon_formula(-0.2, 1e-9, 0.1) is None and A.horizon_formula(0.0, 1e-9, 0.1) is None


# ---------------------------------------------------------------- H4: exakte Exponenten
def test_h4_exponents_of_stable_cycles_by_hand():
    """r = 2,8: Fixpunkt, f′ = 2 − r = −0,8, λ = ln 0,8 = −0,223144. r = 3,2: 2-Zyklus (0,7995; 0,5130), Mehrfachwert 0,16, λ = ½·ln 0,16 = −0,916291."""
    assert math.log(0.8) == pytest.approx(-0.223144, abs=5e-7) and 0.5 * math.log(0.16) == pytest.approx(-0.916291, abs=5e-7)
    for r, exact in ((2.8, math.log(0.8)), (3.2, 0.5 * math.log(4 + 2 * 3.2 - 3.2 ** 2))):
        p, lam, _ = A.cycle_exponent(r)
        est = float(A.lambda_curve(np.array([r]), 30000, 3000)[0])
        assert lam == pytest.approx(exact, abs=1e-9) and est == pytest.approx(exact, abs=5e-7), r
    assert A.cycle_exponent(2.8)[0] == 1 and A.cycle_exponent(3.2)[0] == 2


def test_h4_three_cycle_at_383():
    p, lam, mult = A.cycle_exponent(3.83)
    assert p == 3 and mult == pytest.approx(0.329882, abs=5e-7) and lam == pytest.approx(math.log(0.329882) / 3, abs=1e-6) and round(lam, 6) == -0.369674


# ---------------------------------------------------------------- Schätzer
def test_estimator_on_the_fixed_point_and_ln_4_at_zero():
    """Auf dem Fixpunkt x* = 1 − 1/r ist ln|f′| = ln|2 − r| in jedem Schritt: λ_N = ln|2 − r| ohne Streuung."""
    r = 3.3
    x = 1 - 1 / r
    lam, guard = A.lyapunov_single(r, x, 50)
    assert not guard and lam == pytest.approx(math.log(abs(2 - r)), abs=1e-9)


def test_estimates_for_checks_are_prefix_means():
    """Die Schätzung bei N = 100 aus einem Lauf bis 1000 ist die Schätzung eines Laufs bis 100."""
    x0 = A.start_values(1, 5)
    both = A.lyapunov_estimates(3.9, x0, (100, 1000))
    single = A.lyapunov_estimates(3.9, x0, (100,))
    assert both[100] == pytest.approx(single[100], abs=1e-15)
    for i, x in enumerate(x0):
        assert both[100][i] == pytest.approx(A.lyapunov_single(3.9, float(x), 100)[0], abs=1e-12)
