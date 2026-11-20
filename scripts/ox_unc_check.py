"""Reported uncertainty against empirical scatter, flat spectrum on a tilted slit.
Columns: sigma ratios reported/empirical (1 = correct) for uncertainty='residual' and
'propagated' on the detector grid and on the out_x=2 grid, the lag-1 correlation
(empirical, predicted from the returned covariance) and the mean chi2.
Usage: ox_unc_check.py [read_noise] [nreal]"""
import sys
from pathlib import Path
import numpy as np
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "tests"))
import charslit
from synth import make_swath

NC, NR = 400, 136
RN = float(sys.argv[1]) if len(sys.argv) > 1 else 0.0
NREAL = int(sys.argv[2]) if len(sys.argv) > 2 else 60
TR = slice(30, NC - 30)
T2 = slice(60, 2 * NC - 60)
rng = np.random.default_rng(3)
noises = [rng.standard_normal((NR, NC)) for _ in range(NREAL)]
SET = (("linear (master)", dict(osample_x=1, mu_x=[])),
       ("ox 8 default", dict(osample_x=8)),
       ("ox 8 mu3=1e-6", dict(osample_x=8, mu_x=[0, 0, 1e-6])))

print(f"read noise {RN:g} e-, {NREAL} realisations")
print(f"{'shear':>5} {'setting':16} {'weights':7} {'resid':>6} {'prop':>6} {'prop x2':>8} "
      f"{'r1 emp':>7} {'r1 pred':>8} {'r1 x2 emp':>10} {'pred':>6} {'chi2':>6}")
for shear in (0.0, 2.0, 8.0):
    m, yc, sc, _ = make_swath(dict(kind="flat"), NC, NR, c1=shear / NR)
    V = m + RN ** 2
    for name, kw in SET:
        for w in ("none", "model", "unc", "flux"):
            S, S2, Ur, Up, Cv, C2, chi = [], [], [], [], [], [], []
            for z in noises:
                d = m + np.sqrt(V) * z
                a = (d, np.sqrt(V), np.ones(d.shape, np.uint8), yc, sc, np.zeros(NR))
                o = 2 if kw["osample_x"] > 1 else None
                r0 = charslit.slitdec(*a, osample=8, kappa=0.0, weights=w, **kw)
                r1 = charslit.slitdec(*a, osample=8, kappa=0.0, weights=w, uncertainty="propagated", **kw)
                S.append(r1["spectrum"][TR]); Ur.append(r0["uncertainty"][TR]); Up.append(r1["uncertainty"][TR])
                Cv.append(r1["covariance"][TR]); chi.append(r1["chi2"])
                if o:
                    r2 = charslit.slitdec(*a, osample=8, kappa=0.0, weights=w, uncertainty="propagated", out_x=o, **kw)
                    S2.append(r2["spectrum_out"][T2]); C2.append(r2["covariance"][T2])
            S = np.array(S); D = S - S.mean(0); emp = S.std(0).mean(); Cv = np.mean(Cv, 0)
            r1e = (D[:, 1:] * D[:, :-1]).mean() / D.var(); r1p = Cv[:, 1].mean() / Cv[:, 0].mean()
            x2 = ""
            if S2:
                S2 = np.array(S2); D2 = S2 - S2.mean(0); C2 = np.mean(C2, 0)
                x2 = (f"{np.sqrt(C2[:, 0].mean()) / S2.std(0).mean():>8.2f} {r1e:>+7.2f} {r1p:>+8.2f} "
                      f"{(D2[:, 1:] * D2[:, :-1]).mean() / D2.var():>+10.2f} {C2[:, 1].mean() / C2[:, 0].mean():>+6.2f}")
            else:
                x2 = f"{'':>8} {r1e:>+7.2f} {r1p:>+8.2f} {'':>10} {'':>6}"
            print(f"{shear:>5g} {name:16} {w:7} {np.mean(Ur) / emp:>6.2f} {np.mean(Up) / emp:>6.2f} {x2} {np.mean(chi):>6.2f}")
