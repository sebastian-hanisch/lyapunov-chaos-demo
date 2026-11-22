"""Chaos messen – Lyapunov-Exponent und Vorhersagehorizont - interaktive Konzept-Demo
Sebastian Hanisch - Operations Research und Machine Learning

Stück 3 der Linie „Dynamische Systeme“: fünf Experimente zur Frage, wie stark die Empfindlichkeit der logistischen Abbildung ist und wie lange eine Vorhersage hält. Aufbau nach DEMO-PLAYBOOK §2:
fester Kopf, Schnellstart und allgemeine Einstellungen (r, Zufalls-Seed) in der Seitenleiste, die Regler der Experimente im Hauptbereich beim Experiment, Kernfrage vor den Annahmen,
Formeln im Expander am Ende. Schwere Messreihen (3000 Starts bis 10⁵ Schritte) sind vorgerechnet (`tools/messung_schaetzfehler.py`, `data/messungen.json`) und als solche gekennzeichnet.

Lauffähig mit: streamlit run app.py
"""

import math

import streamlit as st

import lyc_algorithm as A
import lyc_constants as C
import lyc_evaluation as E
from lyc_evaluation import de, de_sci
from lyc_layout import dunkles_design, kopf_css
from lyc_presets import (
    SETTING_SPECS,
    bounds,
    init_session_state_defaults,
    load_permalink_settings,
    randomize_seed,
    sidebar_presets,
    sync_query_params,
)
from lyc_visualization import (
    build_conjugation,
    build_curve,
    build_distance,
    build_errors,
    build_tent,
    build_two_methods,
)

st.set_page_config(page_title="Chaos messen – Sebastian Hanisch", layout="wide")


@st.cache_data(show_spinner=False)
def _curve(lo, hi, n_cols, n, thr):
    return E.analyse_curve(lo, hi, n_cols, n, thr)


@st.cache_data(show_spinner="Rechnet mit 130 Dezimalstellen …")
def _horizon(r, exp, tol, seed):
    return E.analyse_horizon(r, exp, tol, seed)


@st.cache_data(show_spinner="Rechnet mit 130 Dezimalstellen …")
def _decade(r, seed):
    return E.analyse_decade(r, seed)


@st.cache_data(show_spinner=False)
def _errors(seed):
    return E.analyse_errors(seed)


@st.cache_data(show_spinner=False)
def _telescope(x0, n):
    return A.telescope(x0, n)


@st.cache_data(show_spinner=False)
def _two_orbit(r, exp, n, seed):
    return E.analyse_two_orbit(r, exp, n, seed)


@st.cache_data(show_spinner=False)
def _frozen():
    return E.load_frozen()


# Fester Kopf: nur der Titel bleibt beim Scrollen oben (Stylesheet in lyc_layout.py).
st.html(kopf_css(dunkles_design()))
with st.container(key="kopf"):
    st.title("🌪️ Chaos messen – Lyapunov-Exponent und Vorhersagehorizont")
st.markdown(
    """
Das Bifurkationsdiagramm im vorigen Stück ([logistische-abbildung-demo](https://github.com/sebastian-hanisch/logistische-abbildung-demo)) zeigt, **dass** die logistische Abbildung bei r > 3,57 unregelmäßig wird, und dass Fenster darin liegen; es sagt nicht, **wie stark** die Empfindlichkeit ist. Dafür braucht es eine Zahl: den **Lyapunov-Exponenten λ**, die mittlere Wachstumsrate kleiner Fehler je Schritt. Aus ihm folgt der **Vorhersagehorizont** n ≈ ln(Toleranz/Anfangsfehler)/λ.

Fünf Experimente: **λ(r) mit Fenstern**, **Vorhersagehorizont** gegen Anfangsfehler (Decimal als Wahrheit), **Schätzfehler gegen N**, die **Konjugation bei r = 4** (λ = ln 2 exakt) und **zwei Methoden** zur Messung von λ. Danach die Kernfrage, **🚧 Wo die Annahmen enden** und der Expander **📐 Mathematische Formulierung**.
"""
)

# erklaerseite:einfach:start (erzeugt von tools/set_demo_explainer.py (Website-Repo) aus tools/content/explainers/vorhersagehorizont-erklaert.json, nicht von Hand ändern)
with st.expander("💬 In einfachen Worten"):
    st.markdown(
        """
Eine Rechenregel macht aus einer Zahl die nächste, und man wendet sie immer wieder an. Startet man zweimal mit fast gleichen Zahlen, laufen die beiden Folgen bei manchen Einstellungen auseinander. Der **Lyapunov-Exponent** misst, wie schnell: Er ist die mittlere Wachstumsrate eines kleinen Abstands je Schritt. Aus ihm lässt sich ausrechnen, nach wie vielen Schritten ein Anfangsfehler eine Vorhersage unbrauchbar macht, dem **Vorhersagehorizont**. Wer den Anfangswert um ein Vielfaches genauer misst, gewinnt dadurch nur wenige Schritte. Die Demo prüft das mit einer Rechnung, die weit mehr Stellen mitführt als der Rechner üblicherweise, und zeigt, dass schon die Rundung im Rechner den Horizont begrenzt. Es ist ein Spielmodell mit einer einzigen Gleichung, keine Wettervorhersage.

[Ausführlich und verständlich erklärt, mit Fachteil](https://sebastianhanisch.net/vorhersagehorizont-erklaert.html)
        """
    )
# erklaerseite:einfach:end

with st.expander("So funktioniert das Messen von Chaos", expanded=True):
    st.markdown(
        """
1. **Ableitungsmethode:** Entlang einer Bahn (englisch Orbit) x₀, x₁, … mittelt man ln|f′(x_i)| mit f′(x) = r(1 − 2x): λ_N = (1/N)·Σ ln|f′(x_i)|. Ein positives λ heißt: Kleine Fehler wachsen im Mittel wie e^(λk).
2. **Zweibahn-Verfahren:** Eine zweite Bahn im Abstand d₀ wird nach jedem Schritt auf diesen Abstand zurückgesetzt (renormiert); gemittelt wird ln(Abstand/d₀). Das braucht keine Ableitung, aber ein passendes d₀.
3. **Vorhersagehorizont (englisch Prediction Horizon):** Der erste Schritt, an dem zwei Bahnen mit Anfangsabstand δ₀ weiter als die Toleranz auseinander liegen. Gerechnet wird mit 130 Dezimalstellen als Wahrheit; die Float64-Bahn ist daneben die Rechnung, die man sonst hat.
        """
    )

st.caption("🔗 Die Adresszeile oben spiegelt Ihre aktuelle Konfiguration wider – einfach kopieren, um ein Szenario zu teilen.")

load_permalink_settings()
init_session_state_defaults()

with st.sidebar:
    sidebar_presets()  # Schnellstart oben, das aktive Szenario ist hervorgehoben
    st.header("⚙️ Einstellungen")
    st.markdown("**Abbildung**")
    r = st.slider("Parameter r", *bounds("r"), step=0.01, format="%.2f", key="r", help="Experiment 2 und 5 rechnen bei diesem r; Experiment 1 zeigt es als Linie im λ(r)-Diagramm. Experiment 3 und 4 haben feste r (4 und 3,9).")
    st.markdown("**Stichproben**")
    seed = st.number_input("Zufalls-Seed der Startwerte", *bounds("seed"), key="seed", step=1, help="Bestimmt die Startwerte der Stichproben in Experiment 2, 3 und 5 (SplitMix64).")
    st.button("🎲 Neue Startwerte ziehen", width="stretch", on_click=randomize_seed)

frozen = _frozen()

# ---------------------------------------------------------------------------------------------------------------------------------------
# Experiment 1: λ(r)
# ---------------------------------------------------------------------------------------------------------------------------------------
st.markdown("---")
st.subheader("🔬 Experiment 1: λ(r) mit Fenstern")
st.caption(
    "Für jedes r der Exponent aus einer Bahn (Startwert 0,3141…, 3000 Einschwingschritte, dann N Schritte). Orange: λ über der Schwelle (unregelmäßig), grau: |λ| höchstens die Schwelle (nahe null), grün: λ unter −Schwelle (stabiler Zyklus, auch die Fenster im Chaos). "
    "Die schwarze Linie ist das r der Seitenleiste."
)
c1, c2, c3 = st.columns(3)
e1_lo = c1.slider("r von", *bounds("e1_lo"), step=0.001, format="%.3f", key="e1_lo")
e1_hi = c2.slider("r bis", *bounds("e1_hi"), step=0.001, format="%.3f", key="e1_hi", help="Steht der obere Regler unter dem unteren, gilt die Mindestbreite 0,005.")
e1_ncols = c3.slider("Zahl der r-Werte", *bounds("e1_ncols"), key="e1_ncols")
c4, c5 = st.columns(2)
e1_n = c4.slider("Bahnlänge N", *bounds("e1_n"), step=500, key="e1_n", help="Je größer N, desto kleiner der Schätzfehler; die Rechenzeit wächst mit N.")
e1_thr = c5.slider("Schwelle für „chaotisch“", *bounds("e1_thr"), step=0.0005, format="%.4f", key="e1_thr", help="λ über der Schwelle gilt als positiv, |λ| höchstens die Schwelle als nahe null.")
cv = _curve(float(e1_lo), float(e1_hi), int(e1_ncols), int(e1_n), float(e1_thr))
st.plotly_chart(build_curve(cv, float(r)), width="stretch", key="e1_curve")
kind, text = E.verdict_curve(cv)
getattr(st, kind)(text)
k1 = frozen["kurve"]
st.caption(
    f"**Vorgerechnet** ({k1['n_r']} r-Werte von 2,5 bis 3,9999, N = {k1['n']}, Schwelle 0,003): oberhalb von {de(C.ONSET, 7)} liegen {k1['oberhalb']} Werte, davon {k1['pos']} ({de(100 * k1['pos'] / k1['oberhalb'], 1)} %) mit λ über der Schwelle, "
    f"{k1['nahe_null']} nahe null und {k1['neg']} negativ; im Bereich 3,8284 bis 3,857: {k1['fenster_neg']} von {k1['fenster_n']} negativ; größtes λ {de(k1['lambda_max'], 4)} bei r = {de(k1['r_bei_lambda_max'], 4)} (ln 2 = {de(A.LN2, 4)}). "
    "Gegenproben bei stabilen Zyklen: "
    + "; ".join(f"r = {de(float(k), 2)}: {de(v['lambda_geschaetzt'], 6)} gegen {de(v['lambda_exakt'], 6)}" for k, v in k1["exakt"].items())
    + "."
)

# ---------------------------------------------------------------------------------------------------------------------------------------
# Experiment 2: Vorhersagehorizont
# ---------------------------------------------------------------------------------------------------------------------------------------
st.markdown("---")
st.subheader("🔬 Experiment 2: Vorhersagehorizont gegen Anfangsfehler")
st.caption(
    "Zwei Bahnen im Abstand δ₀ (Decimal, 130 Stellen); der Horizont ist der erste Schritt mit Abstand über der Toleranz, im Mittel über 100 Startwerte aus (0,05; 0,95) (Zufalls-Seed der Seitenleiste). "
    "Daneben die Formel ln(Toleranz/δ₀)/λ. Das r steht in der Seitenleiste."
)
c1, c2 = st.columns(2)
e2_exp = c1.slider("Anfangsfehler δ₀ = 10^(−x), x =", *bounds("e2_exp"), key="e2_exp", help="x = 9 heißt δ₀ = 10⁻⁹.")
e2_tol = c2.slider("Toleranz", *bounds("e2_tol"), step=0.01, key="e2_tol")
h2 = _horizon(float(r), int(e2_exp), float(e2_tol), int(seed))
m1, m2, m3 = st.columns(3)
m1.metric("Mittlerer Horizont", "–" if h2.reached == 0 else de(float(h2.steps[h2.steps < C.HORIZON_NMAX].mean()), 1))
m2.metric("Formel ln(Toleranz/δ₀)/λ", "–" if h2.formula is None else de(h2.formula, 1), help="λ = ln 2 bei r = 4 (exakt), sonst aus der Bahn geschätzt.")
m3.metric("Exponent λ", de(h2.lam, 4), delta="exakt" if h2.lam_exact else "geschätzt", delta_color="off")
st.plotly_chart(build_distance(h2), width="stretch", key="e2_distance")
kind, text = E.verdict_horizon(h2)
getattr(st, kind)(text)
fh = frozen["horizont"]
st.caption(
    "**Vorgerechnet** (Decimal, Toleranz 0,1, 100 Starts, mittlerer Horizont in Schritten für δ₀ = 10⁻³ / 10⁻⁶ / 10⁻⁹ / 10⁻¹² / 10⁻¹⁵ / 10⁻³⁰): "
    + "; ".join(f"r = {de(float(rr), 1)}: " + " / ".join(de(fh[rr][e]["mittel"], 1) for e in ("3", "6", "9", "12", "15", "30")) for rr in ("4.0", "3.9", "3.7"))
    + "."
)

# ---------------------------------------------------------------------------------------------------------------------------------------
# Experiment 3: Schätzfehler gegen N
# ---------------------------------------------------------------------------------------------------------------------------------------
st.markdown("---")
st.subheader("🔬 Experiment 3: Schätzfehler gegen N")
st.caption(
    "Der Fehler der Schätzung λ_N, gemessen am Median über die Starts (nicht am quadratischen Mittel). Referenz bei r = 4 ist ln 2, bei r = 3,9 das Mittel der Schätzungen bei N = 20000 (selbst eine Schätzung). "
    f"Live an {C.E3_STARTS_LIVE} Starts bis N = 10⁴; darunter die vorgerechnete Messung an {C.E3_STARTS_FROZEN} Starts bis N = 10⁵."
)
e3 = _errors(int(seed))
st.plotly_chart(build_errors(e3), width="stretch", key="e3_errors")
fe = frozen["fehler"]
kind, text = E.verdict_errors(e3, fe)
getattr(st, kind)(text)
st.dataframe(
    {
        "N": [f"{n:,}".replace(",", ".") for n in fe["checks"]],
        "r = 4: Median |Fehler|": [de_sci(v, 2) for v in fe["r4"]["median_abs"]],
        "r = 4: quadratisches Mittel": [de_sci(v, 2) for v in fe["r4"]["rms"]],
        "r = 3,9: Median |Fehler|": [de_sci(v, 2) for v in fe["r39"]["median_abs"]],
        "r = 3,9: quadratisches Mittel": [de_sci(v, 2) for v in fe["r39"]["rms"]],
    },
    hide_index=True,
    width="stretch",
)
zh = fe["zero_hits_r4"]
hits = "; ".join(f"Start {j}, Schritt {k}" for j, k in zh) if zh else "keine"
st.caption(
    f"**Vorgerechnet** ({fe['n_starts']} Starts, Zufalls-Seed {fe['seed']}). Bei r = 4 fallen in 10⁵ Schritten {len(zh)} von {fe['n_starts']} Float64-Bahnen exakt auf 0 ({hits}); "
    "warum das so ist, steht im Stück zum Gleitkomma-Chaos. Referenz r = 3,9: " + f"{de(fe['ref_39'], 5)} (Standardfehler des Mittels {de_sci(fe['ref_39_stderr'], 1)})."
)

# ---------------------------------------------------------------------------------------------------------------------------------------
# Experiment 4: Konjugation
# ---------------------------------------------------------------------------------------------------------------------------------------
st.markdown("---")
st.subheader("🔬 Experiment 4: Konjugation bei r = 4 und die Teleskop-Identität")
st.caption(
    "Mit x = sin²(πθ/2) wird f(x) = 4x(1 − x) zur Zeltabbildung T(θ) = 1 − |1 − 2θ| mit Steigung ±2: Daher ist λ = ln 2 exakt. Wegen 1 − f(x) = (1 − 2x)² gilt außerdem |f′(x)| = 2·√(f(x)(1 − f(x)))/√(x(1 − x)); "
    "die Summe der ln|f′| ist eine Teleskopsumme."
)
c1, c2 = st.columns(2)
e4_x0 = c1.slider("Start x₀", *bounds("e4_x0"), step=0.01, key="e4_x0", help="0, 1/2 und 1 sind keine guten Starts bei r = 4 (die Bahn fällt auf 0).")
e4_n = c2.slider("Schritte N", *bounds("e4_n"), key="e4_n")
t4 = _telescope(float(e4_x0), int(e4_n))
m1, m2, m3, m4 = st.columns(4)
m1.metric("Σ ln|f′(x_i)|", de(t4.sum_log, 6))
m2.metric("N·ln 2 + Randterm", de(t4.formula, 6))
m3.metric("Schätzung λ_N", de(t4.estimate, 6), delta=f"Fehler {de(t4.error, 6)}", delta_color="off")
m4.metric("ln 2", de(A.LN2, 6))
col_a, col_b = st.columns(2)
col_a.plotly_chart(build_conjugation(float(e4_x0)), width="stretch", key="e4_conj")
col_b.plotly_chart(build_tent(), width="stretch", key="e4_tent")
kind, text = E.verdict_telescope(t4)
getattr(st, kind)(text)
th, xs3, mult3 = A.three_cycle_theta()
st.caption(
    f"3-Zyklus der Zeltabbildung θ = 2/7 → 4/7 → 6/7 entspricht x = sin²(π/7), sin²(2π/7), sin²(3π/7) = {', '.join(de(v, 5) for v in xs3)}; Produkt der Ableitungen {de(mult3, 6)} (Zelt: +2·(−2)·(−2) = 8), "
    f"also λ = ln(8)/3 = {de(math.log(8) / 3, 6)} = ln 2."
)

# ---------------------------------------------------------------------------------------------------------------------------------------
# Experiment 5: zwei Methoden
# ---------------------------------------------------------------------------------------------------------------------------------------
st.markdown("---")
st.subheader("🔬 Experiment 5: Zwei Methoden, λ zu messen")
st.caption(
    f"Ableitungsmethode gegen Zweibahn-Verfahren an {C.E5_STARTS} Starts (Zufalls-Seed der Seitenleiste), beide an derselben Bahn, bei dem r der Seitenleiste. "
    "Bei sehr kleinem d₀ fällt der Abstand der Bahnen durch Rundung auf 0, und das Ergebnis ist nicht endlich."
)
c1, c2 = st.columns(2)
e5_exp = c1.slider("Anfangsabstand d₀ = 10^(−x), x =", *bounds("e5_exp"), key="e5_exp")
e5_n = c2.slider("Bahnlänge N", *bounds("e5_n"), step=1000, key="e5_n")
t5 = _two_orbit(float(r), int(e5_exp), int(e5_n), int(seed))
m1, m2, m3 = st.columns(3)
m1.metric("Endliche Ergebnisse", f"{t5.finite} von {len(t5.two_orbit)}")
m2.metric("Mittel Zweibahn", "–" if t5.mean_two is None else de(t5.mean_two, 5))
m3.metric("Mittel Ableitung", de(float(t5.derivative.mean()), 5))
st.plotly_chart(build_two_methods(t5), width="stretch", key="e5_methods")
kind, text = E.verdict_two_orbit(t5)
getattr(st, kind)(text)
fz = frozen["zweibahn"]
st.caption(
    f"**Vorgerechnet** (N = {fz['n']}, {C.E5_STARTS} Starts, Zufalls-Seed {fz['seed']}): endliche Ergebnisse bei r = 3,9 für d₀ = 10⁻³ / 10⁻⁶ / 10⁻⁹ / 10⁻¹²: "
    + " / ".join(f"{fz['3.9']['zeilen'][k]['endlich']}" for k in ("0.001", "1e-06", "1e-09", "1e-12"))
    + "; bei r = 4 für 10⁻⁶ / 10⁻⁹ / 10⁻¹²: "
    + " / ".join(f"{fz['4.0']['zeilen'][k]['endlich']}" for k in ("1e-06", "1e-09", "1e-12"))
    + "."
)

# ---------------------------------------------------------------------------------------------------------------------------------------
# Kernfrage
# ---------------------------------------------------------------------------------------------------------------------------------------
st.markdown("---")
st.markdown("## ❓ Kernfrage: Was kaufen genauere Anfangswerte?")
dc = _decade(float(r), int(seed))
m1, m2 = st.columns(2)
m1.metric("Formel ln(10)/λ (Schritte je Zehnerpotenz)", "–" if dc.formula is None else de(dc.formula, 2))
m2.metric("Gemessen (Decimal, Mittel über 100 Starts)", "–" if dc.per_decade is None else de(dc.per_decade, 2))
kind, text = E.verdict_decade(dc)
getattr(st, kind)(text)

st.markdown("---")

# ---------------------------------------------------------------------------------------------------------------------------------------
# Grenzen
# ---------------------------------------------------------------------------------------------------------------------------------------
st.subheader("🚧 Wo die Annahmen enden")
st.markdown(
    """
| Annahme | Was passiert, wenn sie verletzt ist | Wer setzt an |
|---|---|---|
| **λ aus einer Bahn gibt die Empfindlichkeit** | Die Schätzung konvergiert nur langsam (bei r = 3,9 gemessen wie 1/√N, bei r = 4 wie 1/N); die Aussagen gelten für die logistische Abbildung bei den gemessenen r und Startwerten, nicht für Chaos allgemein. | – |
| **„Chaotisch“ heißt λ über der Schwelle** | Die Zählung (773 von 860 Werten) gilt für dieses Raster und diese Schwelle, nicht für „fast alle r“; Werte nahe null und Fenster sind Teil der Zählung. | – |
| **Saubere Bahnen der bekannten Abbildung** | Mit Messrauschen steigt der geschätzte Exponent stark (Rosenstein-Schätzer: +9 % bis +52 % bei Rauschen von 0,001); dieses Stück misst nur saubere Bahnen. | [exponent-aus-daten-demo](https://github.com/sebastian-hanisch/exponent-aus-daten-demo) |
| **Float64 als Rechnung** | Der Rundungsfehler wirkt wie ein Anfangsfehler von etwa 1,4·10⁻¹⁷ bis 2,6·10⁻¹⁷ (gerechnet aus dem Float64-Horizont, nicht direkt gemessen); eine Float-Bahn ist trotzdem periodisch und bei r = 4 fällt gelegentlich eine auf 0. | [gleitkomma-chaos-demo](https://github.com/sebastian-hanisch/gleitkomma-chaos-demo) |
| **Zweibahn-Verfahren mit passendem d₀** | Bei d₀ nahe der Maschinengenauigkeit ist das Ergebnis oft nicht endlich; im Lager-Modell dient das Verfahren als Messmethode. | [lager-chaos-demo](https://github.com/sebastian-hanisch/lager-chaos-demo) |
"""
)

st.markdown("---")

with st.expander("📐 Mathematische Formulierung"):
    # erklaerseite:lesehilfe:start (erzeugt von tools/set_demo_explainer.py (Website-Repo) aus tools/content/explainers/vorhersagehorizont-erklaert.json, nicht von Hand ändern)
    st.markdown(
        r"""
**Lesehilfe.** Gegeben ist die logistische Abbildung $f(x) = r\,x\,(1-x)$ auf $[0,1]$ mit $f'(x) = r(1-2x)$ und ein Startwert $x_0$; gesucht ist die mittlere Wachstumsrate kleiner Fehler (der Lyapunov-Exponent) und die Zeit, nach der ein Anfangsfehler die Vorhersage unbrauchbar macht. Das Modell ist vollständig für diese Abbildung und die fünf Experimente (Float64-Rechnung, der Horizont mit 130 Dezimalstellen); für andere Systeme folgt aus ihm nichts.

| Symbol | Bedeutung |
|---|---|
| $\lambda$ | Lyapunov-Exponent, mittlere Wachstumsrate kleiner Fehler je Schritt |
| $\lambda_N$ | Schätzung aus einer Bahn der Länge $N$ |
| $\delta_0$ | Anfangsfehler (Abstand zweier Bahnen bei $k = 0$) |
| $\varepsilon$ | Toleranz, ab der die Vorhersage als falsch gilt |
| $T(\theta)$ | Zeltabbildung $1 - \lvert 1 - 2\theta \rvert$ |
| $d_0$ | Rücksetzabstand des Zweibahn-Verfahrens |

[Schritt für Schritt erklärt, mit Herleitung und Literatur](https://sebastianhanisch.net/vorhersagehorizont-erklaert.html#fachleute)
        """
    )
    # erklaerseite:lesehilfe:end
    st.markdown(
        r"""
**Lesehilfe.** Gegeben ist die logistische Abbildung $f(x) = r\,x\,(1-x)$ auf $[0,1]$ mit $f'(x) = r(1-2x)$ und ein Startwert $x_0$; gesucht ist die mittlere Wachstumsrate kleiner Fehler (der Lyapunov-Exponent) und die Zeit, nach der ein Anfangsfehler die Vorhersage unbrauchbar macht.
Das Modell ist vollständig für diese Abbildung und die fünf Experimente (Float64-Rechnung, der Horizont mit 130 Dezimalstellen); für andere Systeme folgt aus ihm nichts.

| Symbol | Bedeutung |
|---|---|
| $\lambda$ | Lyapunov-Exponent, mittlere Wachstumsrate kleiner Fehler je Schritt |
| $\lambda_N$ | Schätzung aus einer Bahn der Länge $N$ |
| $\delta_0$ | Anfangsfehler (Abstand zweier Bahnen bei $k = 0$) |
| $\varepsilon$ | Toleranz, ab der die Vorhersage als falsch gilt |
| $T(\theta)$ | Zeltabbildung $1 - \lvert 1 - 2\theta \rvert$ |
| $d_0$ | Rücksetzabstand des Zweibahn-Verfahrens |

**Exponent und Schätzung.** Für kleine Fehler gilt $\delta_k \approx \delta_0\, e^{\lambda k}$, mit

$$\lambda = \lim_{N\to\infty} \frac1N \sum_{i=0}^{N-1} \ln \lvert f'(x_i) \rvert , \qquad \lambda_N = \frac1N \sum_{i=0}^{N-1} \ln \lvert f'(x_i) \rvert .$$

Bei stabilen Zyklen der Länge $p$ ist $\lambda = \frac1p \ln \lvert \prod_{j} f'(x_j) \rvert$ (Fixpunkt: $\ln\lvert 2 - r\rvert$; 2-Zyklus: $\tfrac12 \ln(4 + 2r - r^2)$).

**Vorhersagehorizont.** Wächst der Abstand wie $\delta_0 e^{\lambda n}$, ist er bei der Toleranz $\varepsilon$ erreicht nach

$$n \approx \frac{\ln(\varepsilon / \delta_0)}{\lambda} ;$$

ein zehnmal kleineres $\delta_0$ kauft $\ln(10)/\lambda$ Schritte ($3{,}3$ bei $r = 4$). Die Formel gilt im Mittel über Startwerte, denn der Faktor $\lvert f' \rvert$ hängt von $x$ ab.

**r = 4.** Mit $x = \sin^2(\pi\theta/2)$ wird $f$ zur Zeltabbildung $T$:

$$f(x) = 4\sin^2\!\Big(\tfrac{\pi\theta}{2}\Big)\cos^2\!\Big(\tfrac{\pi\theta}{2}\Big) = \sin^2(\pi\theta) = \sin^2\!\Big(\tfrac{\pi\, T(\theta)}{2}\Big) \quad (\theta \le \tfrac12),$$

also $\lvert T' \rvert = 2$ und $\lambda = \ln 2$. Wegen $1 - f(x) = (1-2x)^2$ gilt $\lvert f'(x) \rvert = 2\sqrt{f(x)(1-f(x))}\,/\sqrt{x(1-x)}$ und damit die Teleskopsumme

$$\sum_{i=0}^{N-1} \ln\lvert f'(x_i)\rvert = N \ln 2 + \tfrac12 \ln\big(x_N(1-x_N)\big) - \tfrac12 \ln\big(x_0(1-x_0)\big),$$

$$\lambda_N - \ln 2 = \frac{\ln\big(x_N(1-x_N)\big) - \ln\big(x_0(1-x_0)\big)}{2N} .$$

Der Fehler fällt deshalb wie $1/N$ (solange $x_N$ nicht an $0$ oder $1$ liegt); bei $r = 3{,}9$ gibt es diese Identität nicht, der Fehler fällt dort gemessen wie $1/\sqrt N$.

**Zweibahn-Verfahren (Benettin).** Nach jedem Schritt wird die zweite Bahn auf den Abstand $d_0$ zurückgesetzt, $y \leftarrow x + d_0\,(y - x)/\lvert y - x \rvert$, und $\lambda \approx \frac1N \sum_k \ln(\lvert y_k - x_k \rvert / d_0)$.

**Literatur.** Das Zweibahn-Verfahren nach Benettin und Wolf und der Lyapunov-Exponent sind Lehrbuchwissen und hier aus dem Gedächtnis genannt, nicht anhand einer Quelle geprüft; die Tests prüfen die Formeln gegen unabhängige Rechnungen.
        """
    )

st.markdown("---")
st.caption(
    "Diese Demo ist Teil des Portfolios von [Sebastian Hanisch](https://sebastianhanisch.net) – "
    "Operations Research und Machine Learning ([Über mich](https://sebastianhanisch.net/ueber-mich.html)). "
    "Mehr zur Reihe: [Dynamische Systeme: Ruhepunkt, Kreislauf, Chaos](https://sebastianhanisch.net/konzepte-dynamische-systeme.html)."
)

sync_query_params({key: st.session_state[key] for key in SETTING_SPECS})
