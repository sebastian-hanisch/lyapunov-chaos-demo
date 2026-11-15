"""Layout (DEMO-PLAYBOOK §2): fester Kopf, Schnellstart oben in der Seitenleiste, aktives Szenario hervorgehoben, Regler der Experimente im Hauptbereich.

Die Hervorhebung ist zustandslos (`aktives_preset` vergleicht jeden Regler aus SETTING_SPECS mit dem Szenario). Die Tests unten stellen sicher, dass wirklich JEDER Regler
der Seite dort steht (sonst bliebe das Szenario trotz Änderung markiert) und dass jede einzelne Änderung die Hervorhebung aufhebt."""

from pathlib import Path

import pytest
from streamlit.testing.v1 import AppTest

import lyc_constants as C
import lyc_presets as P

APP = str(Path(__file__).resolve().parent.parent / "app.py")
REGLER_ARTEN = ("slider", "number_input", "select_slider", "selectbox", "radio", "toggle", "checkbox", "text_input")


def _app() -> AppTest:
    at = AppTest.from_file(APP, default_timeout=180)
    at.run()
    assert not at.exception, [str(e) for e in at.exception]
    return at


def _typen(at) -> dict:
    """Button-Typ je Szenario-Name (die Schnellstart-Knöpfe der Seitenleiste)."""
    return {b.label: b.proto.type for b in at.sidebar.button if b.label in C.PRESETS}


def _regler(at, key):
    for art in REGLER_ARTEN:
        try:
            return getattr(at, art)(key=key)
        except KeyError:
            continue
    raise AssertionError(f"Regler {key} steht nicht auf der Seite")


def _anderer_wert(regler, spec):
    """Ein Wert, der vom aktuellen abweicht und kein Wert eines Szenarios ist (r = 4,0 ist der Höchstwert und zugleich das Szenario „r = 4: ln 2“): die Reglermitte."""
    if spec.choices is not None:
        return next(c for c in spec.choices if c != regler.value)
    mitte = (spec.lo + spec.hi) / 2
    if spec.caster is int:
        mitte = int(round(mitte))
    return mitte if mitte != regler.value else spec.lo


def test_presets_stehen_in_der_seitenleiste_und_nicht_im_hauptbereich():
    at = _app()
    assert [b.label for b in at.sidebar.button][: len(C.PRESETS)] == list(C.PRESETS)
    assert not [b for b in at.main.button if b.label in C.PRESETS]


def test_allgemeine_einstellung_steht_in_der_seitenleiste_und_die_regler_der_experimente_im_hauptbereich():
    at = _app()
    seitenleiste = {w.key for art in REGLER_ARTEN for w in getattr(at.sidebar, art) if w.key}
    assert seitenleiste == {"r", "seed"}
    haupt = {w.key for art in REGLER_ARTEN for w in getattr(at.main, art) if w.key}
    assert haupt == set(P.SETTING_SPECS) - {"r", "seed"}


def test_titel_steht_im_festen_kopf():
    at = _app()
    assert "st-key-kopf" in " ".join(h.proto.body for h in at.get("html"))
    assert at.title[0].value


def test_beim_start_ist_genau_das_erste_szenario_hervorgehoben():
    typen = _typen(_app())
    assert [name for name, typ in typen.items() if typ == "primary"] == [next(iter(C.PRESETS))]


@pytest.mark.parametrize("name", list(C.PRESETS))
def test_angeklicktes_szenario_setzt_die_regler_ist_hervorgehoben_und_bleibt_es(name):
    at = _app()
    at.sidebar.button(key=f"preset_{name}").click().run()
    assert not at.exception, [str(e) for e in at.exception]
    assert [n for n, typ in _typen(at).items() if typ == "primary"] == [name]
    for key, spec in P.SETTING_SPECS.items():
        assert _regler(at, key).value == C.PRESETS[name].get(key, spec.default), (name, key)
    at.run()
    assert [n for n, typ in _typen(at).items() if typ == "primary"] == [name]


def test_jeder_regler_der_seite_steht_in_setting_specs():
    at = _app()
    schluessel = {w.key for art in REGLER_ARTEN for w in getattr(at, art) if w.key}
    fehlt = schluessel - set(P.SETTING_SPECS)
    assert not fehlt, f"Regler {sorted(fehlt)} fehlen in SETTING_SPECS: ihre Änderung hebt das aktive Szenario nicht auf"
    assert set(P.SETTING_SPECS) <= schluessel


@pytest.mark.parametrize("state_key", list(P.SETTING_SPECS))
def test_jede_aenderung_eines_reglers_hebt_die_hervorhebung_auf(state_key):
    at = _app()
    spec = P.SETTING_SPECS[state_key]
    regler = _regler(at, state_key)
    regler.set_value(_anderer_wert(regler, spec)).run()
    assert not at.exception, [str(e) for e in at.exception]
    assert "primary" not in _typen(at).values()
    name = next(iter(C.PRESETS))
    at.sidebar.button(key=f"preset_{name}").click().run()
    assert _typen(at)[name] == "primary"
