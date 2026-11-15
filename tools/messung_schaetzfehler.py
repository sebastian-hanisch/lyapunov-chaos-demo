"""Vorgerechnete Messungen für Stück „Chaos messen“ (zu schwer für die Live-Oberfläche), Ergebnis in data/messungen.json.

Aufruf (aus dem Wurzelordner der Demo, einige Minuten, ein Prozess):

    python tools/messung_schaetzfehler.py            # rechnet alles neu und schreibt data/messungen.json
    python tools/messung_schaetzfehler.py --check    # rechnet kleine Teile neu (300 Starts, N bis 10^4; 100 r-Werte) und vergleicht mit der Datei

Inhalt (alle Startwerte aus SplitMix64 mit festem Zufalls-Seed, Float64, Ableitungsmethode wo nicht anders genannt):
  fehler    Fehler der Schätzung λ_N über 3000 Starts gleichmäßig in [0,05; 0,95) (Zufalls-Seed 31), N = 10², 10³, 10⁴, 10⁵, für r = 4 (Referenz ln 2) und r = 3,9
            (Referenz: Mittel der 3000 Schätzungen bei N = 10⁵, selbst eine Schätzung); Median |Fehler|, quadratisches Mittel, Steigungen im log-log-Bild;
            Zahl der Starts, deren Float64-Bahn bei r = 4 in 10⁵ Schritten exakt auf 0 fällt; λ(3,7) als Mittel derselben 3000 Schätzungen.
  kurve6000 Gegenprobe der Zählung mit 6000 r-Werten (sonst gleich).
  kurve     λ(r) auf 3000 r-Werten von 2,5 bis 3,9999 (Einschwingen 3000, N = 30000): Zählung oberhalb 3,5699456, Fenster 3,8284 bis 3,857, exakte Gegenproben bei stabilen Zyklen.
  zweibahn  Zweibahn-Verfahren gegen Ableitungsmethode, r = 3,9 und r = 4, 20 Starts (Zufalls-Seed 41), N = 10⁵, d0 = 10⁻³ … 10⁻¹².
  horizont  Vorhersagehorizont (Decimal, 130 Stellen, tol 0,1, 100 Starts, Zufalls-Seed 4) für r = 4; 3,9; 3,7 und Anfangsfehler 10⁻³ … 10⁻³⁰."""

import json
import math
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import lyc_algorithm as A  # noqa: E402

DATA = ROOT / "data" / "messungen.json"
SEED_ERR, SEED_HOR, SEED_TWO = 31, 4, 41


def messung_fehler(n_starts=3000, checks=(100, 1000, 10000, 100000)) -> dict:
    x0 = A.start_values(SEED_ERR, n_starts)
    est = {r: A.lyapunov_estimates(r, x0, checks) for r in (4.0, 3.9, 3.7)}
    nmax = max(checks)
    ref39 = float(est[3.9][nmax].mean())
    out = {"n_starts": n_starts, "checks": list(checks), "seed": SEED_ERR, "ref_39": ref39, "ref_39_stderr": float(est[3.9][nmax].std() / math.sqrt(n_starts)),
           "lambda_37": float(est[3.7][nmax].mean()), "lambda_37_stderr": float(est[3.7][nmax].std() / math.sqrt(n_starts))}
    for key, r, ref in (("r4", 4.0, A.LN2), ("r39", 3.9, ref39)):
        rows = A.estimation_errors(est[r], ref)
        out[key] = {"median_abs": [w.median_abs for w in rows], "rms": [w.rms for w in rows], "over_1e3": [w.over_1e3 for w in rows],
                    "slope_median": A.loglog_slope(checks, [w.median_abs for w in rows]), "slope_rms": A.loglog_slope(checks, [w.rms for w in rows])}
    out["zero_hits_r4"] = A.first_zero_hit(4.0, x0, nmax)
    return out


def messung_kurve(n_r=3000, n=30000, transient=3000) -> dict:
    rs = np.linspace(2.5, 3.9999, n_r)
    lam = A.lambda_curve(rs, n, transient)
    cc = A.count_curve(rs, lam, 0.003)
    w_all, w_neg = A.window_counts(rs, lam)
    exact = {}
    for r in (2.8, 3.2, 3.83):
        p, lam_exact, mult = A.cycle_exponent(r)
        est = float(A.lambda_curve(np.array([r]), n, transient)[0])
        exact[str(r)] = {"periode": p, "multiplikator": mult, "lambda_exakt": lam_exact, "lambda_geschaetzt": est}
    return {"n_r": n_r, "n": n, "transient": transient, "oberhalb": cc.n_above, "pos": cc.pos, "nahe_null": cc.zero, "neg": cc.neg, "max_unterhalb": cc.max_below,
            "pos_unterhalb": cc.n_below_pos, "lambda_max": float(lam.max()), "r_bei_lambda_max": float(rs[int(np.argmax(lam))]), "fenster_n": w_all, "fenster_neg": w_neg, "exakt": exact}


def messung_kurve_6000() -> dict:
    """Gegenprobe zum Raster: 6000 r-Werte (doppelt so dicht), sonst gleiche Einstellungen."""
    rs = np.linspace(2.5, 3.9999, 6000)
    lam = A.lambda_curve(rs, 30000, 3000)
    cc = A.count_curve(rs, lam, 0.003)
    return {"n_r": 6000, "oberhalb": cc.n_above, "pos": cc.pos, "nahe_null": cc.zero, "neg": cc.neg}


def messung_zweibahn(n=100000) -> dict:
    x0 = A.start_values(SEED_TWO, 20)
    out = {"seed": SEED_TWO, "n": n}
    for r, ds in ((3.9, (1e-3, 1e-6, 1e-9, 1e-12)), (4.0, (1e-6, 1e-9, 1e-12))):
        ref = A.lyapunov_estimates(r, x0, (n,))[n]
        rows = {}
        for d0 in ds:
            zw = A.lyapunov_two_orbit(r, x0, d0, n)
            ok = np.isfinite(zw)
            target = ref if r != 4.0 else np.full_like(ref, A.LN2)
            rows[repr(d0)] = {"endlich": int(ok.sum()), "mittel": float(zw[ok].mean()) if ok.any() else None,
                              "max_abw": float(np.abs(zw[ok] - target[ok]).max()) if ok.any() else None}
        out[str(r)] = {"ableitung_mittel": float(ref.mean()), "ableitung_streuung": float(ref.std()), "zeilen": rows}
    return out


def messung_horizont() -> dict:
    x0 = A.start_values(SEED_HOR, 100)
    out = {"seed": SEED_HOR, "tol": 0.1, "starts": 100}
    for r in (4.0, 3.9, 3.7):
        rows = {}
        for e in (3, 6, 9, 12, 15, 30):
            ns = A.horizon_decimal(r, x0, 10.0 ** -e, 0.1)
            rows[str(e)] = {"mittel": float(ns.mean()), "min": int(ns.min()), "max": int(ns.max())}
        out[str(r)] = rows
    tolrows = {}
    for tol in (0.01, 0.1, 0.5):
        ns = A.horizon_decimal(4.0, x0, 1e-9, tol)
        tolrows[str(tol)] = {"mittel": float(ns.mean()), "min": int(ns.min()), "max": int(ns.max())}
    out["tol_r4_d09"] = tolrows
    return out


def run_all() -> dict:
    t0 = time.time()
    res = {"beschreibung": __doc__.split("\n\n")[0]}
    for name, f in (("fehler", messung_fehler), ("kurve", messung_kurve), ("kurve6000", messung_kurve_6000), ("zweibahn", messung_zweibahn), ("horizont", messung_horizont)):
        res[name] = f()
        print(f"  {name} fertig ({time.time() - t0:.0f} s)", file=sys.stderr, flush=True)
    res["rechenzeit_s"] = round(time.time() - t0, 1)
    return res


def main(argv) -> int:
    if "--check" in argv:
        stored = json.loads(DATA.read_text(encoding="utf-8"))
        x0 = A.start_values(SEED_ERR, 300)
        est = A.lyapunov_estimates(4.0, x0, (100, 1000, 10000))
        rows = A.estimation_errors(est, A.LN2)
        fresh = [w.median_abs for w in rows]
        print("Median |Fehler| r = 4, 300 Starts:", fresh)
        rs = np.linspace(2.5, 3.9999, 3000)[::30]
        lam = A.lambda_curve(rs, 30000, 3000)
        old = stored["kurve"]
        print("λ(r) auf jedem 30. r-Wert neu gerechnet:", len(rs), "Werte, größtes", float(lam.max()), "gespeichert insgesamt", old["lambda_max"])
        return 0
    res = run_all()
    DATA.parent.mkdir(exist_ok=True)
    DATA.write_text(json.dumps(res, ensure_ascii=False, indent=1), encoding="utf-8")
    print("fertig", res["rechenzeit_s"], "s")
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
