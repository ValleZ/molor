"""Which essential oils are worth sourcing GC data for, decided by measurement.

Building an oil costs a literature search for a real composition, so the
question of which to do next deserves better than a guess. This screens
candidates against the three things that actually decide it.

*Does the library hold the molecules that matter?* Sandalwood is santalol and
vetiver is khusimol, and neither is in the library, so no amount of care makes
them buildable -- the same wall patchouli and oakmoss hit.

*Is it a real mixture?* An oil that is 85% of one thing lands on that thing.
Sweet orange already demonstrated this, sitting on D-Limonene at a cosine of
1.0000, and eucalyptus, lemongrass and palmarosa will do the same on their own
dominant constituents. Those builds are not wrong so much as empty.

*Can it be checked?* The model's 138-word vocabulary contains some natural
names and not others. An oil whose own name is in there can be scored blind;
one whose is not has to be judged by eye, which is how a build that omits the
signature molecule passes inspection -- the reason ylang-ylang was left out.

The compositions here are approximate, from general knowledge rather than a
cited analysis, and they exist only to rank candidates. Anything this screen
recommends gets a sourced composition before it is built for real, which is why
nothing here is written to data/naturals/compositions.json.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))

from build_naturals import self_word  # noqa: E402
from eval_mixtures import combine  # noqa: E402

#: Candidate oils and their principal constituents, as library names where one
#: exists and a bare chemical name where it does not. Approximate shares.
CANDIDATES = {
    "Neroli": {"Linalool": 35, "Linalyl Acetate": 15, "Nerolidol": 8,
               "D-Limonene": 12, "Terpineol Alpha": 5, "Geraniol 95": 3,
               "Methyl Anthranilate": 0.5, "Indolene 50": 0.2,
               "Beta Pinene": 8},
    "Jasmine absolute": {"Benzyl Acetate": 24, "Benzyl Benzoate": 14,
                         "Linalool": 8, "Indolene 50": 2, "Cis Jasmone": 3,
                         "Hedione": 3, "Phenethyl Alcohol": 1,
                         "methyl jasmonate": 2, "Farnesol": 2},
    "Lavandin Grosso": {"Linalool": 30, "Linalyl Acetate": 28, "Camphor": 7,
                        "Eucalyptol": 6, "Borneol": 3, "Terpineol Alpha": 2,
                        "Caryophyllene Beta": 2},
    "Rosemary (cineole type)": {"Eucalyptol": 45, "Camphor": 12,
                                "Alpha Pinene": 12, "Borneol": 4,
                                "camphene": 5, "D-Limonene": 3,
                                "verbenone": 2},
    "Citronella (Java)": {"Citronellal": 38, "Geraniol 95": 22,
                          "Citronellol": 11, "D-Limonene": 4,
                          "Geranyl Acetate": 4},
    "Cardamom": {"Terpinyl Acetate": 40, "Eucalyptol": 28, "Linalool": 4,
                 "D-Limonene": 3, "Terpineol Alpha": 3, "sabinene": 4},
    "Basil (linalool type)": {"Linalool": 50, "Methyl Chavicol": 12,
                              "Eucalyptol": 6, "Eugenol ex Clove": 4,
                              "Caryophyllene Beta": 3},
    "Thyme (thymol type)": {"Thymol": 42, "p-cymene": 20,
                            "Terpinene Gamma": 9, "Carvacrol": 4,
                            "Linalool": 4, "Caryophyllene Beta": 2},
    "Black pepper": {"Caryophyllene Beta": 25, "D-Limonene": 20,
                     "Beta Pinene": 10, "Alpha Pinene": 9,
                     "delta-3-carene": 15, "Myrcene": 2},
    "Juniper berry": {"Alpha Pinene": 40, "Myrcene": 10, "D-Limonene": 5,
                      "sabinene": 15, "Terpinene Gamma": 2,
                      "Caryophyllene Beta": 3},
    "Clary sage": {"Linalyl Acetate": 60, "Linalool": 18,
                   "Geranyl Acetate": 5, "germacrene D": 5,
                   "Caryophyllene Beta": 2, "sclareol": 1},
    "Cypress": {"Alpha Pinene": 45, "delta-3-carene": 20, "Cedrol": 5,
                "D-Limonene": 4, "Terpinolene, Rectified": 3},
    "Spearmint": {"l-Carvone": 60, "D-Limonene": 15, "Eucalyptol": 2,
                  "Myrcene": 2, "Beta Pinene": 1},
    "Eucalyptus globulus": {"Eucalyptol": 80, "Alpha Pinene": 8,
                            "D-Limonene": 5},
    "Lemongrass": {"Citral": 75, "Myrcene": 8, "Geraniol 95": 4,
                   "D-Limonene": 2},
    "Palmarosa": {"Geraniol 95": 80, "Geranyl Acetate": 8, "Linalool": 3,
                  "Nerol": 2},
    "Star anise": {"Anethole ex Star Anise": 88, "D-Limonene": 3,
                   "Linalool": 1},
    "Coriander seed": {"Linalool": 72, "Alpha Pinene": 6, "Camphor": 5,
                       "D-Limonene": 3, "Geraniol 95": 2},
    "Ho wood": {"Linalool": 92, "Eucalyptol": 2, "Camphor": 1},
    "Sandalwood": {"alpha-santalol": 48, "beta-santalol": 22,
                   "santalene": 10, "Farnesol": 1},
    "Tea tree": {"terpinen-4-ol": 40, "Terpinene Gamma": 20,
                 "p-cymene": 4, "Eucalyptol": 4, "Alpha Pinene": 3},
    "Chamomile (Roman)": {"isobutyl angelate": 35, "chamazulene": 5,
                          "pinocarvone": 8, "Alpha Pinene": 2},
    "Ginger": {"zingiberene": 30, "camphene": 8, "ar-curcumene": 10,
               "Citral": 3, "D-Limonene": 3},
}


def main() -> int:
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    except Exception:  # noqa: BLE001
        pass

    pj = json.loads((ROOT / "artifacts" / "percepts_fp3.json")
                    .read_text(encoding="utf-8"))
    desc = pj["descriptors"]
    names = list(pj["molecules"])
    P = {n: np.asarray(r, float) for n, r in zip(names, pj["probs"])}
    LM = np.stack([P[n] for n in names])
    X = LM - LM.mean(0)
    Xn = X / np.maximum(np.linalg.norm(X, axis=1, keepdims=True), 1e-12)
    rng = np.random.default_rng(0)

    rows = []
    for oil, comp in CANDIDATES.items():
        stated = sum(comp.values())
        res = [(n, v) for n, v in comp.items() if n in P]
        cov = sum(v for _, v in res) / stated
        word = self_word(oil)
        word = word if (word and word in desc) else None

        verdict, nearest, zself = "", "", None
        if not res or cov < 0.6:
            verdict = "cannot build"
        else:
            w = np.array([v for _, v in res], float)
            w = w / w.sum()
            mix = combine("noisy-or", np.stack([P[n] for n, _ in res]), w)
            y = mix - LM.mean(0)
            y = y / max(float(np.linalg.norm(y)), 1e-12)
            sim = Xn @ y
            k = int(np.argmax(sim))
            nearest = names[k]
            if sim[k] >= 0.995:
                verdict = f"collapses onto {nearest}"
            else:
                verdict = "real mixture"
            if word:
                null = np.stack([
                    combine("noisy-or",
                            LM[rng.choice(len(LM), len(res), replace=False)], w)
                    for _ in range(1500)])
                z = (mix - null.mean(0)) / np.maximum(null.std(0), 1e-9)
                i = desc.index(word)
                zself = (int(list(np.argsort(-z)).index(i)) + 1, float(z[i]))
        rows.append({"oil": oil, "coverage": cov, "word": word,
                     "verdict": verdict, "nearest": nearest, "self": zself,
                     "cos": float(sim[k]) if res and cov >= 0.6 else None,
                     "missing": [n for n in comp if n not in P]})

    def rank(r):
        good = r["verdict"] == "real mixture"
        return (not good, -(r["self"][1] if r["self"] else -9), -r["coverage"])

    print("[screen] candidates, best first. Compositions are approximate and "
          "exist only to rank what to source.\n")
    # The cosine to the nearest single material is printed rather than left
    # implicit: several oils sit just under the 0.995 collapse threshold, and
    # "real mixture at 0.991" deserves less confidence than one at 0.80.
    print(f"  {'oil':26s} {'cov':>5s} {'nearest':>7s} {'verdict':30s} "
          f"{'self-check':>22s}")
    print("  " + "-" * 96)
    for r in sorted(rows, key=rank):
        sc = (f"{r['word']} {r['self'][0]}/138 z{r['self'][1]:+.1f}"
              if r["self"] else (f"no word ({r['word']})" if r["word"]
                                 else "no word"))
        cs = f"{r['cos']:.3f}" if r["cos"] is not None else "-"
        print(f"  {r['oil'][:26]:26s} {r['coverage']:5.0%} {cs:>7s} "
              f"{r['verdict'][:30]:30s} {sc:>22s}")
        if r["missing"] and r["verdict"] == "cannot build":
            print(f"        missing: {', '.join(r['missing'])}")

    best = [r for r in sorted(rows, key=rank)
            if r["verdict"] == "real mixture" and r["coverage"] >= 0.8]
    print(f"\n[screen] {len(best)} worth sourcing: "
          + ", ".join(r["oil"] for r in best))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
