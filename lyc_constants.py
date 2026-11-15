"""Konstanten: Regler-Grenzen, feste Annahmen, Farben und Schnellstart-Szenarien.

r = 4 ist hier erlaubt (Zielfall: λ = ln 2 exakt), aber nur mit Startwerten, die keine Dualbrüche sind (kein x₀ = 0, 1/2, 1); alle anderen Voreinstellungen sind keine Zweierpotenzen
(Rechner-Artefakt der Linie, siehe das Stück zum Gleitkomma-Chaos). Der Zufalls-Seed heißt „Zufalls-Seed“."""

import math

# ---- allgemeine Einstellungen (Seitenleiste) ----------------------------------------------------------------------------------------------
R_MIN, R_MAX, DEFAULT_R = 2.5, 4.0, 3.9
SEED_MAX, DEFAULT_SEED = 999999, 2027

# ---- Experiment 1: λ(r) --------------------------------------------------------------------------------------------------------------
E1_LO_DEFAULT, E1_HI_DEFAULT = 2.5, 3.9999
E1_NCOLS_MIN, E1_NCOLS_MAX, DEFAULT_E1_NCOLS = 200, 1500, 700
E1_N_MIN, E1_N_MAX, DEFAULT_E1_N = 1000, 20000, 6000
E1_THR_MIN, E1_THR_MAX, DEFAULT_E1_THR = 0.0005, 0.05, 0.003      # Schwelle für „chaotisch“ (λ > Schwelle) und „nahe null“ (|λ| ≤ Schwelle)
E1_TRANSIENT = 3000
E1_X0 = 0.3141592653589793
ONSET = 3.5699456                                                   # Häufungspunkt der Verdopplungen (Zählgrenze der Vormessung; Messung aus dem Stück davor, nicht aus einer Quelle)
WINDOW_R = 1.0 + math.sqrt(8.0)                                     # Beginn des Periode-3-Fensters
WINDOW_END = 3.857                                                  # Ende des Zählbereichs „Fenster“ der Vormessung

# ---- Experiment 2: Vorhersagehorizont ------------------------------------------------------------------------------------------------
E2_EXP_MIN, E2_EXP_MAX, DEFAULT_E2_EXP = 3, 30, 9                   # Anfangsfehler δ₀ = 10^(−exp)
E2_TOL_MIN, E2_TOL_MAX, DEFAULT_E2_TOL = 0.01, 0.9, 0.1
E2_STARTS = 100
E2_START_LO, E2_START_HI = 0.05, 0.95
DECIMAL_DIGITS = 130
HORIZON_NMAX = 400
LAMBDA_STARTS = 20
LAMBDA_N = 20000

# ---- Experiment 3: Schätzfehler gegen N ------------------------------------------------------------------------------------------------
E3_STARTS_LIVE = 200
E3_CHECKS_LIVE = (100, 1000, 10000)
E3_CHECKS_FROZEN = (100, 1000, 10000, 100000)
E3_STARTS_FROZEN = 3000

# ---- Experiment 4: Konjugation und Teleskop-Identität --------------------------------------------------------------------------------
E4_X0_MIN, E4_X0_MAX, DEFAULT_E4_X0 = 0.01, 0.99, 0.3
E4_N_MIN, E4_N_MAX, DEFAULT_E4_N = 5, 2000, 10

# ---- Experiment 5: zwei Methoden --------------------------------------------------------------------------------------------------------
E5_EXP_MIN, E5_EXP_MAX, DEFAULT_E5_EXP = 1, 12, 9
E5_N_MIN, E5_N_MAX, DEFAULT_E5_N = 2000, 100000, 20000
E5_STARTS = 20

# ---- Farben --------------------------------------------------------------------------------------------------------------------------
COLOR_MAIN = "#2F6B65"
COLOR_REF = "#c0562a"
COLOR_GREY = "rgba(150,150,150,0.75)"
COLOR_POS = "#c0562a"
COLOR_ZERO = "#8f8f8f"
COLOR_NEG = "#2F6B65"

# ---- Schnellstart: je Szenario die Werte der Regler, die es setzt; alle übrigen behalten ihre Voreinstellung ------------------------------
PRESETS = {
    "r = 3,9": {},
    "r = 4: ln 2": {"r": 4.0},
    "r = 3,83: Fenster": {"r": 3.83, "e1_lo": 3.8, "e1_hi": 3.87},
    "r = 3,2: stabil": {"r": 3.2},
    "r = 3,7": {"r": 3.7},
}
PRESET_HELP = {
    "r = 3,9": "Alle Experimente bei r = 3,9 (λ ≈ 0,4960): Horizont, Zweibahn-Verfahren; Experiment 1 zeigt das ganze λ(r)-Diagramm. Alle anderen Regler stehen auf ihrer Voreinstellung.",
    "r = 4: ln 2": "r = 4, der Zielfall: λ = ln 2 = 0,6931 exakt (Konjugation mit der Zeltabbildung), der Schätzfehler fällt wie 1/N, ein zehnmal kleinerer Anfangsfehler kostet ln(10)/ln 2 = 3,3 Schritte.",
    "r = 3,83: Fenster": "r = 3,83 im Periode-3-Fenster (λ = −0,3697 aus dem Zyklus-Multiplikator); Experiment 1 zoomt auf r von 3,80 bis 3,87.",
    "r = 3,2: stabil": "r = 3,2 mit dem stabilen 2-Zyklus: λ = ½·ln(4 + 2r − r²) = −0,9163; der Horizont ist unendlich (kleine Fehler schrumpfen).",
    "r = 3,7": "r = 3,7: chaotisch; Horizont und Zweibahn-Verfahren hier gerechnet (λ(3,7) wird in der App aus der Bahn geschätzt).",
}
DEFAULT_PRESET = "r = 3,9"
