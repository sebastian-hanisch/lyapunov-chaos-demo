"""Schnellstart-Szenarien: Vollständigkeit, gültige Werte, das erste Szenario entspricht den Voreinstellungen, Adresszeile (Permalink) begrenzt Werte;
r = 4 kommt nur als Szenario vor (Zielfall), nie als Voreinstellung, und kein Startwert ist ein Dualbruch."""

import math

import lyc_constants as C
import lyc_evaluation as E
import lyc_presets as P


def test_every_preset_has_help_and_only_known_keys():
    assert set(C.PRESETS) == set(C.PRESET_HELP)
    for name, preset in C.PRESETS.items():
        assert set(preset) <= set(P.SETTING_SPECS), name
        assert C.PRESET_HELP[name]


def test_preset_values_are_inside_the_slider_bounds():
    for name, preset in C.PRESETS.items():
        for key, value in preset.items():
            spec = P.SETTING_SPECS[key]
            assert spec.lo <= value <= spec.hi, (name, key)


def test_first_preset_is_the_default_state():
    first = next(iter(C.PRESETS))
    assert first == C.DEFAULT_PRESET and C.PRESETS[first] == {}
    assert E.Settings() == E.Settings(**{key: spec.default for key, spec in P.SETTING_SPECS.items()})


def test_presets_differ_pairwise_in_at_least_one_setting():
    full = {name: tuple(p.get(key, spec.default) for key, spec in P.SETTING_SPECS.items()) for name, p in C.PRESETS.items()}
    assert len(set(full.values())) == len(full)


def test_setting_specs_match_the_settings_dataclass():
    assert set(P.SETTING_SPECS) == set(E.Settings.__dataclass_fields__)


def test_defaults_are_no_forbidden_dual_fractions_and_r_4_is_only_a_preset():
    """Voreinstellungen sind keine Zweierpotenzen (r, e4_x0, e1_thr, e2_tol …); r = 4 ist als Szenario erlaubt (Zielfall), nicht als Voreinstellung."""
    for key in ("r", "e1_lo", "e1_hi", "e1_thr", "e2_tol", "e4_x0"):
        v = float(P.SETTING_SPECS[key].default)
        m, _ = math.frexp(abs(v))
        assert m != 0.5, (key, v)
    assert P.SETTING_SPECS["r"].default != 4.0 and C.PRESETS["r = 4: ln 2"]["r"] == 4.0
    assert C.DEFAULT_E4_X0 not in (0.0, 0.5, 1.0)


def test_the_window_preset_sets_the_zoom_of_the_first_experiment():
    p = C.PRESETS["r = 3,83: Fenster"]
    assert (p["e1_lo"], p["e1_hi"]) == (3.8, 3.87) and p["e1_lo"] < C.WINDOW_R < p["e1_hi"]


def test_permalink_values_are_clamped_and_invalid_ones_ignored():
    spec = P.SETTING_SPECS["r"]
    assert P._clean(spec, "9") == spec.hi and P._clean(spec, "-3") == spec.lo and P._clean(spec, "3.2") == 3.2
    assert P._clean(spec, "abc") is None and P._clean(spec, "nan") is None
    assert P._clean(P.SETTING_SPECS["e1_n"], "999999") == C.E1_N_MAX and P._clean(P.SETTING_SPECS["seed"], "-5") == 0 and P._clean(P.SETTING_SPECS["seed"], "x") is None
