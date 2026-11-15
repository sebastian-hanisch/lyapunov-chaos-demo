# Chaos messen – Lyapunov-Exponent und Vorhersagehorizont – Streamlit-Demo

<!-- TODO nach dem Deploy: direkt hierher den Live-Link setzen (Leerzeile davor und danach):
**[→ Demo live ausprobieren](https://sebastianhanisch-lyapunov-chaos-demo.streamlit.app/)** -->

Stück 3 der Linie **Dynamische Systeme** in der „Konzepte“-Reihe für die Website „Sebastian Hanisch – Operations Research und Machine Learning“. Es setzt auf [logistische-abbildung-demo](https://github.com/sebastian-hanisch/logistische-abbildung-demo) auf: Das Bifurkationsdiagramm zeigt dort, **dass** die Abbildung bei r > 3,57 unregelmäßig wird und Fenster darin liegen, sagt aber nicht, **wie stark** die Empfindlichkeit ist und wie lange eine Vorhersage hält. Dafür braucht es eine Zahl, den Lyapunov-Exponenten. Nachfolger: [gleitkomma-chaos-demo](https://github.com/sebastian-hanisch/gleitkomma-chaos-demo) (warum eine Rechnerbahn immer periodisch ist) und [exponent-aus-daten-demo](https://github.com/sebastian-hanisch/exponent-aus-daten-demo) (der Exponent aus verrauschten Messreihen).

Vehikel ist die logistische Abbildung x → r·x·(1 − x). Gegenproben sind exakte Werte bei stabilen Zyklen, die Konjugation mit der Zeltabbildung bei r = 4 (λ = ln 2) samt Teleskop-Identität, Decimal-Rechnung mit 130 Stellen für den Horizont und das Zweibahn-Verfahren gegen die Ableitungsmethode.

## Warum dieses Problem

**Der Lyapunov-Exponent λ ist die mittlere Wachstumsrate kleiner Fehler je Schritt; bei r = 4 ist er exakt ln 2 (Konjugation mit der Zeltabbildung), und aus ihm folgt der Vorhersagehorizont n ≈ ln(Toleranz/Anfangsfehler)/λ: Ein zehnmal genauerer Anfangswert kauft im Mittel über Startwerte nur ln(10)/λ Schritte (3,32 bei r = 4). Die Schätzung des Exponenten aus einer Bahn konvergiert bei r = 4 wie 1/N (Teleskopsumme), bei r = 3,9 gemessen nur wie 1/√N.**

Alle Aussagen gelten für die logistische Abbildung bei den gemessenen r und Startwerten (Float64 oder Decimal mit 130 Stellen), nicht für Chaos allgemein.

## Modell

Fünf Experimente (Formeln im 📐-Expander der App):

1. **λ(r) mit Fenstern:** Für jedes r aus einem Bereich der Exponent aus einer Bahn (Startwert 0,3141…, 3000 Einschwingschritte, dann N Schritte, Regler für Bereich, Zahl der r-Werte, N und Schwelle). Zählung oberhalb von 3,5699456 (λ > Schwelle, |λ| ≤ Schwelle, λ < −Schwelle) und im Bereich 3,8284 bis 3,857.
2. **Vorhersagehorizont gegen Anfangsfehler:** zwei Bahnen im Abstand δ₀ (10⁻³ bis 10⁻³⁰) in Decimal mit 130 Stellen; Horizont = erster Schritt mit Abstand über der Toleranz, Mittel über 100 Starts aus (0,05; 0,95); daneben die Formel mit λ = ln 2 (r = 4) bzw. λ aus 20 Bahnen (N = 20000), und der Horizont der Float64-Bahn gegen die Decimal-Bahn.
3. **Schätzfehler gegen N:** Median |λ_N − λ| über Starts, live an 200 Starts bis N = 10⁴, dazu die vorgerechnete Messung an 3000 Starts bis N = 10⁵ (r = 4: Referenz ln 2; r = 3,9: Referenz das Mittel der Schätzungen, selbst eine Schätzung).
4. **Konjugation bei r = 4:** Bahn in x und in θ = (2/π)·arcsin √x, Zeltabbildung, 3-Zyklus 2/7 → 4/7 → 6/7, Teleskop-Identität live (Summe, Formel, Differenz, Schätzfehler gegen Randterm).
5. **Zwei Methoden:** Ableitungsmethode gegen Zweibahn-Verfahren (Renormierung auf d₀ nach jedem Schritt) an 20 Starts, d₀ = 10⁻¹ bis 10⁻¹².

Dazu eine **Kernfrage** („Was kaufen genauere Anfangswerte?“) und der Abschnitt „Wo die Annahmen enden“.

## Methodik

- **Eingabe:** r und der Zufalls-Seed der Startwerte stehen in der Seitenleiste (r gilt für Experiment 2 und 5; Experiment 3 und 4 haben feste r), ebenso der Schnellstart (fünf Szenarien); die übrigen Regler stehen im Hauptbereich bei ihrem Experiment. **r = 4 ist als Szenario erlaubt, nicht als Voreinstellung**; keine Voreinstellung ist eine Zweierpotenz, und kein Startwert ist 0, 1/2 oder 1 (die Bahn 0,5 → 1 → 0 ist richtige Mathematik, aber ein Randfall).
- **Zufall nur über SplitMix64** (`lyc_rng.py`): Startwerte gleichverteilt in (0,05; 0,95). Zufalls-Seeds der vorgerechneten Messungen: 31 (Fehler, 3000 Starts), 4 (Horizont, 100 Starts), 41 (Zweibahn, 20 Starts).
- **Horizont nicht erreicht:** Erreicht kein Start die Toleranz, unterscheidet das Verdikt drei Gründe: λ ≤ 0 (Fehler schrumpfen), Toleranz nicht kleiner als die Breite der Bahn (r = 3,9, Toleranz 0,9: Breite 0,88; gemessen an einer Bahn) und zu wenige Schritte (400) für den gewählten Anfangsfehler (`tests/test_edge_cases.py`, `tests/test_verdicts.py`).
- **Ableitungsmethode:** λ_N = (1/N)·Σ ln|r(1 − 2x_i)|. Trifft die Bahn genau 1/2, ist |f′| = 0; der Wächter setzt |f′| auf 10⁻³⁰⁰ und meldet den Fall (Randfall-Test).
- **Vorgerechnet** (zu schwer für die Live-Oberfläche) sind: die Messung des Schätzfehlers an 3000 Starts bis N = 10⁵, das λ(r)-Raster mit 3000 und 6000 r-Werten (N = 30000), die Zweibahn-Messung an 20 Starts (N = 10⁵) und die Horizont-Tabelle. Das Skript `tools/messung_schaetzfehler.py` erzeugt `data/messungen.json` neu (gemessen: etwa 25 Sekunden auf dem Entwicklungsrechner) und ist in App und README als „vorgerechnet“ gekennzeichnet; `--check` rechnet Teile neu.
- **Rechenzeit der Live-Rechnungen** (Messung dieser Entwicklung, ein Aufruf, danach zwischengespeichert): λ(r) mit 700 r-Werten und N = 6000 0,06 s, mit 1500 r-Werten und N = 20000 (Reglermaximum) 0,33 s; Horizont über 100 Starts samt Float64-Vergleich 0,15 s (auch bei δ₀ = 10⁻³⁰); Kernfrage 0,12 s; Schätzfehler an 200 Starts 0,26 s; Zweibahn an 20 Starts bei N = 20000 0,23 s, bei N = 10⁵ (Reglermaximum) 1,1 s. Der Test `test_claim_two_orbit_recomputed_for_d0_1e9_and_1e12` rechnet die schwerste Konfiguration neu.

## Befunde (gemessen, keine Behauptungen)

Alle Zahlen sind Messungen dieser README (Messskript `tools/messung_schaetzfehler.py`, Ergebnis `data/messungen.json`, Startwerte aus SplitMix64) und werden in `tests/test_claims.py` gegen die Datei geprüft und, soweit in Sekunden möglich, neu gerechnet. Voraussetzung jeweils: logistische Abbildung, Float64 (Horizont: Decimal, 130 Stellen), die genannten Starts und Zufalls-Seeds.

| Frage | Befund | Test (Art) |
|---|---|---|
| λ(r) auf 3000 r-Werten von 2,5 bis 3,9999 (Einschwingen 3000, N = 30000, Schwelle 0,003) | Oberhalb von 3,5699456 liegen 860 Werte: **773 (89,9 %)** mit λ > 0,003, 3 nahe null, 84 negativ; unterhalb (r ≥ 2,5) ist das größte λ −0,0003, kein Wert über der Schwelle; größtes λ 0,6886 bei r = 3,9999 (ln 2 = 0,6931) | `test_claim_lambda_raster_counts_and_recomputation` (exakt, neu gerechnet) |
| Gegenprobe mit 6000 r-Werten | 1720 oberhalb, davon **1540 (89,5 %)** über der Schwelle, 6 nahe null, 174 negativ (Abweichung 0,4 Prozentpunkte gegen das Raster mit 3000) | `test_claim_lambda_raster_with_twice_the_density` (exakt aus der Datei) |
| Fenster 3,8284 bis 3,857 | 44 von 57 r-Werten mit λ < 0 | wie oben (exakt) |
| Exakte Exponenten stabiler Zyklen | r = 2,8 (Fixpunkt): −0,223144; 3,2 (2-Zyklus, Mehrfachwert 0,16): −0,916291; 3,83 (3-Zyklus, Mehrfachwert +0,329882): −0,369674; die Schätzung aus der Bahn ist gleich auf 10⁻⁹ | `test_claim_exact_exponents_of_stable_cycles` (Band) |
| Schätzfehler: Median \|λ_N − λ\| bei N = 10², 10³, 10⁴, 10⁵ (3000 Starts) | r = 4: 3,26·10⁻³; 3,22·10⁻⁴; 3,33·10⁻⁵; 3,42·10⁻⁶, Steigung im log-log-Bild **−0,992**; r = 3,9: 2,68·10⁻²; 7,92·10⁻³; 2,55·10⁻³; 8,23·10⁻⁴, Steigung **−0,503** | `test_claim_estimation_error_medians_and_slopes` (Band 0,5 %), `test_claim_estimation_errors_recomputed_up_to_n_10000` (exakt, neu gerechnet bis N = 10⁴) |
| Quadratisches Mittel | r = 3,9: Steigung −0,500; r = 4: **−0,194**, weil **eine** der 3000 Float64-Bahnen bei r = 4 in 10⁵ Schritten exakt auf 0 fällt (Start 770, Schritt 62305); deshalb Median statt quadratisches Mittel | `test_claim_over_1e3_counts_and_the_one_orbit_that_falls_on_zero` (exakt, die Bahn neu gerechnet) |
| Starts mit \|Fehler\| > 10⁻³ bei N = 10², 10³, 10⁴, 10⁵ | r = 4: 2307, 572, 1, 1; r = 3,9: 2951, 2788, 2380, 1245 (von 3000) | wie oben (exakt) |
| Referenz r = 3,9 und λ(3,7) | Mittel der 3000 Schätzungen bei N = 10⁵: 0,49602 (Standardfehler des Mittels 2,3·10⁻⁵); **λ(3,7) = 0,35495** (Standardfehler 1,8·10⁻⁵) | `test_claim_reference_for_r_3_9_and_lambda_37` (Band) |
| Teleskop-Identität bei r = 4 | x₀ = 0,3, N = 10: Σ ln\|f′\| = 6,12120307 gleich der Formel (Handbeispiel); N = 10⁶: λ_N − ln 2 = −1,49·10⁻⁶ gleich dem Randterm auf 10⁻⁸; beides gilt für die berechnete Float64-Folge, für die exakte Bahn von x₀ = 0,3 wären die Zahlen andere | `test_h2_telescope_identity_by_hand`, `test_claim_lambda_4_after_a_million_steps_equals_ln_2_plus_the_boundary_term` (Band) |
| Vorhersagehorizont, Decimal, Toleranz 0,1, 100 Starts, δ₀ = 10⁻³ / 10⁻⁶ / 10⁻⁹ / 10⁻¹² / 10⁻¹⁵ / 10⁻³⁰ | r = 4: 7,6 / 17,5 / 27,5 / 37,3 / 47,4 / 97,4; r = 3,9: 9,9 / 23,7 / 37,5 / 51,2 / 66,1 / 136,4; r = 3,7: 12,8 / 32,9 / 51,8 / 70,8 / 90,4 / 188,9 (Spannweite bei r = 4: 26 bis 32 bei 10⁻⁹, 46 bis 51 bei 10⁻¹⁵) | `test_claim_horizon_table_matches_the_file_and_recomputation` (Band, vier Zellen neu gerechnet) |
| Horizont gegen die Formel ln(Toleranz/δ₀)/λ | Das Mittel liegt über der Formel: r = 4 (λ = ln 2) um 0,76 bis 1,05 Schritte über die sechs δ₀; r = 3,9 (λ = 0,49602) um 0,10 bis 1,80 | `test_claim_horizon_against_the_formula_for_r_4_and_3_9` (Band) |
| Schritte je Zehnerpotenz kleineres δ₀ (Mittel zwischen 10⁻⁶ und 10⁻¹²) | r = 4: **3,31** gegen ln(10)/ln 2 = 3,32; r = 3,9: **4,57** gegen ln(10)/0,49602 = 4,64 | `test_claim_steps_per_decade` (Band) |
| Toleranz, r = 4, δ₀ = 10⁻⁹ | Toleranz 0,01 / 0,1 / 0,5: 24,2 / 27,5 / 30,3 Schritte (log₂-Formel 23,3 / 26,6 / 28,9) | `test_claim_tolerance_dependence_at_r_4` (Band) |
| Float64-Bahn gegen Decimal-Bahn, r = 4, Toleranz 0,1, 100 Starts | mittlerer Horizont **52,66** (44 bis 59); nach der Formel einem Anfangsfehler von **1,4·10⁻¹⁷** gleichwertig, nach log-linearer Interpolation in der Tabelle 2,6·10⁻¹⁷; beide unter 2⁻⁵³ = 1,1·10⁻¹⁶. **Gerechnete Größe, nicht direkt gemessen.** | `test_claim_float64_orbit_horizon_and_effective_initial_error` (Band) |
| Zweibahn-Verfahren, 20 Starts, N = 10⁵ | r = 3,9: endliche Ergebnisse bei d₀ = 10⁻³ / 10⁻⁶ / 10⁻⁹ / 10⁻¹²: 20 / 20 / 20 / **6**; größte Abweichung je Start gegen die Ableitungsmethode 3,1·10⁻⁴ / 9,0·10⁻⁶ / 2,0·10⁻⁷; r = 4: 20 / 20 / **8** bei 10⁻⁶ / 10⁻⁹ / 10⁻¹², Abweichung gegen ln 2 bei 10⁻⁹ 2,8·10⁻⁵ | `test_claim_two_orbit_counts_and_deviations`, `test_claim_two_orbit_recomputed_for_d0_1e9_and_1e12` (exakt, neu gerechnet) |

Die 89,9 % gelten für dieses Raster, diese Schwelle und diese Einstellungen, nicht für „fast alle r“.

## Befunde und Korrekturen gegenüber dem Plan

- **Erwartung 1/√N für r = 4 war falsch (Korrektur der Gliederung):** Bei r = 4 fällt der Schätzfehler wie 1/N (Teleskopsumme, Experiment 4), bei r = 3,9 gemessen wie 1/√N. Gemessen mit SplitMix64-Starts: Steigung −0,992 (r = 4) und −0,503 (r = 3,9); die Vormessung (numpy-Zufall) gab −1,006 und −0,499.
- **Zahlen der Vormessung verschoben sich mit den neuen Startwerten,** wie bei jeder Messung: Median bei N = 10⁵ und r = 4 3,42·10⁻⁶ (Vormessung 3,17·10⁻⁶); Horizont r = 4, δ₀ = 10⁻⁹ 27,5 (Vormessung 27,4); bei r = 3,9 und δ₀ = 10⁻¹⁵ 66,1 gegen 65,3. Die Zählung des Rasters (860, 773, 3, 84, 44 von 57) stimmt mit der Vormessung überein, weil sie von den Zufalls-Seeds nicht abhängt.
- **Das Zweibahn-Verfahren versagt bei 10⁻¹² früher als geplant:** Der Plan nannte „16 von 20 nicht endlich“ (4 endlich); mit den Startwerten des Baus sind es bei r = 3,9 14 von 20 nicht endlich (6 endlich) und bei r = 4 12 von 20 (8 endlich). Der Grund (Abstand 0 nach Rundung, Division durch 0) ist eine Vermutung und wurde nicht einzeln nachgeprüft; die App zählt die Starts.
- **Wirksamer Anfangsfehler der Float64-Bahn:** Der Plan nannte rund 3·10⁻¹⁷. Gerechnet sind es 1,4·10⁻¹⁷ nach der Formel (tol·e^(−λ·n̄)) und 2,6·10⁻¹⁷ nach Interpolation in der Horizont-Tabelle; die Formel unterschätzt den Horizont bei r = 4 um etwa einen Schritt (bei r = 3,9 um 0,10 bis 1,80 Schritte), die Interpolation nutzt diese Korrektur. Beides sind gerechnete Größen. Die Aussage „unter 2⁻⁵³“ gilt für beide.
- **Formel gegen Messung:** Der Plan nannte größte Abweichungen von 1,2 (r = 4) und 1,4 (r = 3,9) Schritten; gemessen sind 1,05 und 1,80 (bei δ₀ = 10⁻³⁰, mit λ = 0,49602 als Referenz). Die Formel unterschätzt den Mittelwert in allen zwölf Zellen.
- **λ(3,7) gemessen:** 0,35495 (Plan: nicht gemessen); es steht auf dem Schnellstart-Szenario r = 3,7.
- **„Gleichmäßige Starts“ bei der Messung der Fehler sind SplitMix64-Zufallsstarts** in (0,05; 0,95), nicht ein Gitter.

## Ehrliche Grenzen

- **Nur die logistische Abbildung** bei den gemessenen r und Startwerten; Float64 oder Decimal mit 130 Stellen. Nichts davon gilt für „Chaos allgemein“.
- **„Chaotisch“ heißt hier λ über 0,003 auf dem genannten Raster;** die Zählung hängt an Schwelle, Rasterdichte, N und Einschwingen (89,9 % gegen 89,5 % bei doppelter Dichte). Werte nahe null und Fenster sind Teil der Zählung.
- **Die Referenz bei r = 3,9 ist selbst eine Schätzung** (Mittel der Schätzungen bei N = 10⁵, Standardfehler 2,3·10⁻⁵); der Wert 0,4966 aus der Vormessung der Linie liegt rund 6·10⁻⁴ darüber (hier nicht neu gemessen). Die Steigung 1/√N bei r = 3,9 ist eine Messung, keine Herleitung.
- **Median statt quadratisches Mittel** für den Fehler, weil bei r = 4 eine Bahn von 3000 exakt auf 0 fällt; die Aussage über den Median gilt für diese Starts.
- **Mittelwert des Horizonts hängt an der Verteilung der Startwerte** (hier gleichverteilt in (0,05; 0,95)); die Spannweite über die Starts ist groß (r = 3,9, δ₀ = 10⁻⁹: 29 bis 57).
- **Das Zweibahn-Verfahren braucht ein passendes d₀;** die Aussage „10⁻⁹ ist passend“ gilt für diese Abbildung und N = 10⁵, nicht allgemein. Float16 und Float32, verrauschte Daten und das Verfahren nach Wolf sind nicht behandelt.
- **Quellen:** Lyapunov-Exponent, Benettin und Wolf stehen hier aus dem Gedächtnis (Lehrbuchwissen) und sind nicht anhand einer Quelle geprüft; ln 2 und die Zelt-Konjugation sind hergeleitet und numerisch bestätigt. Es gibt keine Literaturwerte, deshalb kein `test_literatur_*`.
- **Bei r = 3 exakt** konvergiert die Bahn nur langsam, λ_N ist dort nicht entschieden (Betrag unter 0,02 bei N = 30000).

## Verifikation

Die Suite hat 110 Tests (`pytest --collect-only -q tests`).

- **Kern:** Handbeispiele H1 bis H4 des Plans von Hand nachgerechnet (`tests/test_algorithm.py`: 3-Zyklus der Zeltabbildung mit x = 0,18826 / 0,61126 / 0,95048 und Multiplikator +8, Teleskop-Identität 6,12120307, Abstände 0,001 bis 0,106574 bei k = 7, Exponenten −0,223144 und −0,916291).
- **Orakel (`tests/test_oracle_lyapunov_chaos.py`):** λ bei stabilen Zyklen gegen den Multiplikator eines unabhängig per Newton-Verfahren (komplexe Schritt-Ableitung) gelösten Zyklus (r = 2,8; 3,2; 3,83); λ(4) gegen ln 2 und gegen die Teleskop-Formel an 1000 Zufallsstarts; Teleskop-Identität an 50 Zufallsbahnen; Decimal-Horizont mit 130 gegen 200 Stellen; Float64-Bahn gegen Decimal bei Anfangsabstand 10⁻³; Zweibahn gegen Ableitungsmethode (Abweichung unter 10⁻⁵ bei d₀ = 10⁻⁹); λ der konjugierten Abbildung z → z² + (r/2 − r²/4) Schritt für Schritt.
- **Eigenschaften (`tests/test_eigenschaften_lyapunov_chaos.py`):** λ hängt nicht vom Start ab (30 Starts, Band), Teleskop-Identität, Konjugation Logistik zu Zelt, Horizont wächst mit fallendem Anfangsfehler und mit der Toleranz (im Mittel), gleicher Zufalls-Seed gleiche Ausgabe, λ eines Zyklus aus dem Multiplikator an 30 Zufalls-r.
- **Randfälle (`tests/test_edge_cases.py`):** x₀ ∈ {0, 1/2, 1} bei r = 4 (der Nullwächter wird bei 1/2 gezielt ausgelöst), r = 3 exakt, r < 3, N = 1, d₀ nahe der Maschinengenauigkeit, Schwelle an Randwerten (exklusiv für „positiv“, inklusiv für „nahe null“), Horizont ohne Wachstum, Reglerränder.
- **Claims und Verdikte:** jede Zahl der Tabelle oben (`tests/test_claims.py`); alle Verdikte als reine Funktionen an handgesetzten Werten (`tests/test_verdicts.py`).
- **Oberfläche:** AppTest-Rauchtest (jeder Button, jeder Regler an Minimum und Maximum), Schnellstart in der Seitenleiste und aktives Szenario hervorgehoben, jede Regleränderung hebt die Markierung auf (`tests/test_layout.py`), Footer, Laufzeit-Importe.
- **Zweierpotenz-Falle:** Keine Voreinstellung ist eine Zweierpotenz, kein Startwert 0, 1/2 oder 1 (`tests/test_presets.py`); der Zusatzfall x₀ = 1/2 ist ein gewollter Kollaps im Randfall-Test.

## Dateistruktur

| Datei | Zweck |
|---|---|
| `.github/workflows/tests.yml` | CI: Push, Pull Request, wöchentlicher Canary |
| `.gitignore` | venv, Caches, Streamlit-Secrets |
| `.streamlit/config.toml` | Streamlit-Einstellungen (keine Nutzungsstatistik) |
| `app.py` | Streamlit-App: fünf Experimente, Kernfrage, Annahmen, Formeln |
| `lyc_algorithm.py` | Kern: Ableitungsmethode, Zweibahn-Verfahren, Konjugation und Teleskop-Identität, Decimal-Horizont |
| `lyc_constants.py` | Regler-Grenzen, feste Annahmen, Farben, Schnellstart-Szenarien |
| `lyc_evaluation.py` | Kennzahlen und Verdikte (reine Funktionen), Laden der vorgerechneten Messungen |
| `lyc_layout.py` | Fester Kopf (Stylesheet) |
| `lyc_presets.py` | Permalink (Begrenzen), Schnellstart (aktives Szenario hervorgehoben), Zufalls-Seed-Knopf |
| `lyc_rng.py` | SplitMix64 für die Startwerte (Kopie aus `logistische-abbildung-demo`) |
| `lyc_visualization.py` | Plotly-Abbildungen (Achsen gesperrt) |
| `tools/messung_schaetzfehler.py` | Skript der vorgerechneten Messungen (`--check` rechnet Teile neu) |
| `data/messungen.json` | Ergebnis dieser Messungen (eingefroren) |
| `requirements-dev.txt` | Laufzeit plus pytest |
| `requirements.txt` | Laufzeit-Bibliotheken (ungepinnt) |
| `tests/` | Handbeispiele, Orakel, Eigenschaften, Randfälle, Claims, Verdikte, Schnellstart, Layout, AppTest-Rauchtest, Footer, Laufzeit-Importe |

## Bewusst nicht umgesetzt

- **Rauschen und Schätzung aus Daten** (Rosenstein-Schätzer, 0-1-Test): [exponent-aus-daten-demo](https://github.com/sebastian-hanisch/exponent-aus-daten-demo).
- **Float16 und Float32, periodische Rechnerbahnen, Verdopplungsabbildung:** [gleitkomma-chaos-demo](https://github.com/sebastian-hanisch/gleitkomma-chaos-demo).
- **Das Lager-Modell mit dem Zweibahn-Verfahren als Messmethode:** [lager-chaos-demo](https://github.com/sebastian-hanisch/lager-chaos-demo).
- **Die Verfahren von Wolf und Benettin im Wortlaut** und mehrdimensionale Systeme (Lyapunov-Spektrum).
- **PDF-Export:** nicht nötig, die App hat kein Ergebnis zum Mitnehmen.

## Lokal ausführen

```bash
python3 -m venv venv
source venv/bin/activate          # Windows: venv\Scripts\activate

pip install -r requirements.txt
streamlit run app.py
```

## Tests ausführen

```bash
pip install -r requirements-dev.txt
pytest tests/ -v
```

Die vorgerechneten Messungen lassen sich mit `python tools/messung_schaetzfehler.py` neu erzeugen (etwa 25 Sekunden) bzw. mit `--check` teilweise prüfen.

Gebaut mit Streamlit, Plotly und NumPy.

---

Diese Demo ist Teil des Portfolios von [Sebastian Hanisch](https://sebastianhanisch.net) – Operations Research und Machine Learning ([Über mich](https://sebastianhanisch.net/ueber-mich.html)). Mehr zu den Demos: [Interaktive Demos](https://sebastianhanisch.net/demos.html).
