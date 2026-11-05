"""Build essential oils from their GC composition and score them as mixtures.

Naturals were being dropped from every formula, which cost more coverage than
anything else: across the published patent formulas they are 6316 g of the
13362 g the library could not resolve. But a natural is a known mixture of
mostly common materials, so it can be built rather than dropped.

Each oil is composed from its constituents by ONE rule, in both spaces, and
--compose selects it. The default is the composition-weighted mean: an oil is a
thing you can buy, so its vector should describe the thing, and it then sits
among its own constituents on the map instead of outside them.

That is a change, and the reason is worth keeping. The rule used to be noisy-OR
-- a union, `1 - prod(1 - w_i p_i)` -- justified as "a receptor fires if any
constituent fires it", which is right at low occupancy and wrong once ligands
compete for one pocket. It also pushes a mixture outward by construction, and
an oil of five to eight constituents is a mixture: Cedarwood EO came out the
single most extreme point on the published atlas (percept sum 9.84 against a
material median near 5) with Myrrh beside it, reading as a "super wood" that no
perfumer recognises. Recomposed by the mean, cedarwood reads 4.18 and sits with
cedrol and cedrene.

Nothing in the data picks between the rules -- rule_bakeoff.py cannot separate
the mean from the union or from the dominant ingredient -- so the rendered map
decided it. The union remains reachable for the comparison, and the atlas's
white point is unaffected either way: it is the mean of logits, because it is
the map's origin rather than a material.

The validation is built in and is close to free: the model's own vocabulary
contains the words bergamot, lavender, geranium, mint, clove, cinnamon, rose,
lemon, orange and a dozen more. So an oil can be scored without being told its
name, and the question is where its own name lands by lift against a null of
random blends of the same size. A method that cannot put "lavender" near the
top for lavender oil is not worth wiring into anything.

Two things this cannot fix. A composition is one published analysis of one lot
or chemotype, so an oil built here is a plausible member of a family rather
than the bottle on the shelf; the four lavenders are kept partly to make that
visible. And where a constituent is missing from the library the weights are
renormalised over what remains, which quietly rewrites the oil -- geranium
loses isomenthone, peppermint loses menthofuran, and both of those carry
character out of proportion to their percentage.
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))

from eval_mixtures import combine  # noqa: E402
from orvector.store import Library  # noqa: E402

COMPOSITIONS = ROOT / "data" / "naturals" / "compositions.json"
NULL_DRAWS = 3000

#: The descriptor each oil's own name claims, for the blind self-check. Only
#: where the vocabulary has the word; an oil with no matching word is built but
#: not scored.
SELF_WORD = {
    "lavender": "lavender", "spike lavender": "lavender",
    "bergamot": "bergamot", "geranium": "geranium", "peppermint": "mint",
    "spearmint": "mint", "clove bud": "clove", "cinnamon bark": "cinnamon",
    "rose": "rose", "lemon": "lemon", "neroli": "orangeflower",
    "vetiver": "vetiver", "sandalwood": "sandalwood", "cedarwood": "cedar",
    "chamomile": "chamomile", "grapefruit": "grapefruit", "jasmine": "jasmin",
    "orange": "orange", "pine": "pine", "tea": "tea",
    # The vocabulary has no word for either of these. Tangerine is checked
    # against orange, which is the honest comparison anyway since the build
    # omits the only things that distinguish them. Lime is checked against
    # lemon, the nearest available word, so a pass there is weaker evidence
    # than a pass on an oil whose own name is in the vocabulary.
    "tangerine": "orange", "lime": "lemon",
    # These two name a smell rather than a plant, and the claim is testable:
    # lemongrass claims lemon and palmarosa claims rose. Whole-word matching
    # does not reach inside either name, so they are listed.
    "lemongrass": "lemon", "palmarosa": "rose",
    # A lavandin is a lavender hybrid, sold and used as a lavender, so the
    # lavender descriptor is a fair claim to hold it to.
    "lavandin": "lavender",
}


def self_word(name: str) -> str | None:
    """The descriptor an oil's name claims, matched on whole words only.

    Substring matching is wrong here and quietly so: "rosemary" contains
    "rose", so a rosemary oil was being scored against the rose descriptor and
    landing 134th of 138, which reads as a failed build rather than a mismatched
    question. The vocabulary has no word for rosemary, and saying so is the
    correct answer.
    """
    low = re.sub(r"[^a-z]+", " ", name.lower())
    words = set(low.split())
    for key in sorted(SELF_WORD, key=len, reverse=True):
        if all(part in words for part in key.split()):
            return SELF_WORD[key]
    return None


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--percepts", default="artifacts/percepts_fp3.json")
    ap.add_argument("--store", default="library", choices=("library",),
                    help="the receptor store to compose in. One store since "
                         "the MPNN was removed; kept as a flag because --out "
                         "is derived from it")
    ap.add_argument("--out", default="artifacts/naturals.json")
    ap.add_argument("--force-out", action="store_true",
                    help="write --out even when it is not the file this "
                         "--store feeds; for one-off analysis only")
    ap.add_argument("--min-coverage", type=float, default=0.6,
                    help="fraction of the stated composition that must resolve")
    # Trimming is OFF by default. It hid 13 of 47 oils from the atlas because
    # each collapsed onto one constituent (Orange EO 0.9998 to D-Limonene,
    # clove onto eugenol, ...), and the perfumer's ruling is that both the oil
    # and the molecule must be visible: an oil is a real shelf material, not a
    # duplicate point. Pass an explicit threshold to re-enable the trim.
    ap.add_argument("--max-nearest-cosine", type=float, default=1.01,
                    help="drop an oil this close to a single library material; "
                         "default 1.01 keeps every oil (no trimming)")
    ap.add_argument("--compose", default="weighted-mean",
                    choices=("weighted-mean", "logit-mean", "noisy-or"),
                    help="how an oil's vectors are composed from its "
                         "constituents, in BOTH spaces. weighted-mean is the "
                         "rule chosen on the perfumer's judgement of the "
                         "published atlas: oils placed among their own "
                         "constituents instead of outside them, cedarwood's "
                         "percept sum 9.84 -> 4.18. noisy-or keeps an oil equal "
                         "to the union of its parts, which read as the union's "
                         "super-wood extreme. logit-mean is the atlas's "
                         "placement rule for injected blends")
    args = ap.parse_args()

    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    except Exception:  # noqa: BLE001
        pass

    pj = json.loads((ROOT / args.percepts).read_text(encoding="utf-8"))
    desc = pj["descriptors"]
    names = list(pj["molecules"])
    P = {n: np.asarray(r, float) for n, r in zip(names, pj["probs"])}
    LM = np.stack([P[n] for n in names])

    # --store and --out are independent, and getting them out of step writes
    # one encoder's receptor vectors into another encoder's naturals file.
    # That happened: the GCN atlas spent a deploy plotting its oils with MPNN
    # coordinates, which drew them as a tight clump apart from everything
    # else, correctly ordered inside the clump and meaningless beside it.
    expect = "artifacts/naturals.json"
    # Compared as paths, not strings: on Windows "artifacts\\x.json" and
    # "artifacts/x.json" are the same file, and a string comparison rejected
    # the correct target with the message telling you to pass it.
    if Path(args.out) != Path(expect) and not args.force_out:
        raise SystemExit(
            f"--store {args.store} writes {expect}, not {args.out}.\n"
            f"export_viewer_data.py reads one naturals file per store by that "
            f"name, so a mismatch puts one encoder's vectors in another's "
            f"space. Pass --out {expect}, or --force-out to override.")

    entries, _, scores = Library(name=args.store).load()
    S = {e.name: s for e, s in zip(entries, scores)}

    spec = json.loads(COMPOSITIONS.read_text(encoding="utf-8"))["naturals"]
    rng = np.random.default_rng(0)
    X = LM - LM.mean(0)
    Xn = X / np.maximum(np.linalg.norm(X, axis=1, keepdims=True), 1e-12)

    out = {"descriptors": desc, "naturals": {}}
    trimmed: list[tuple[str, float, str]] = []
    print(f"[nat] building {len(spec)} essential oils from GC composition\n")
    for oil, meta in spec.items():
        comp = meta["composition"]
        stated = sum(comp.values())
        res = [(n, v) for n, v in comp.items() if n in P and n in S]
        cov = sum(v for _, v in res) / stated if stated else 0.0
        missing = [n for n in comp if n not in P or n not in S]
        if not res or cov < args.min_coverage:
            print(f"  SKIP {oil}: only {cov:.0%} of the stated composition "
                  f"resolves ({', '.join(missing)})")
            continue

        w = np.array([v for _, v in res], float)
        w = w / w.sum()
        V = np.stack([P[n] for n, _ in res])
        # One rule for both spaces, and it is the same rule the null uses:
        # --compose weighted-mean by default. The union was measured to work
        # against itself here -- an oil composed by union is pushed outside its
        # own constituents, and cedarwood became the most extreme point on the
        # published atlas (percept sum 9.84 against a material median ~5). The
        # perfumer's judgement on the rendered maps was unambiguous, so the
        # composition changed; the union stays reachable by flag for the
        # comparison, and the white point is unaffected either way (it is the
        # mean of logits, because it is the map's origin, not a material).
        mix = combine(args.compose, V, w)
        rec = combine(args.compose, np.stack([S[n] for n, _ in res]), w)

        null = np.stack([
            combine(args.compose, LM[rng.choice(len(LM), len(res), replace=False)], w)
            for _ in range(NULL_DRAWS)])
        z = (mix - null.mean(0)) / np.maximum(null.std(0), 1e-9)
        order = list(np.argsort(-z))

        y = mix - LM.mean(0)
        y = y / max(float(np.linalg.norm(y)), 1e-12)
        sim = Xn @ y
        near = [names[k] for k in np.argsort(-sim)[:5]]

        # An oil dominated by one constituent is not a mixture in any useful
        # sense: sweet orange at 91% limonene lands on D-Limonene at cosine
        # 1.0000, and so does tangerine, because what separates them is a
        # material the library does not hold. Three diamonds stacked on one
        # disc would imply three oils the model can tell apart, so the collapse
        # is recorded and shown on the card instead of being left to look like
        # a result.
        top = int(np.argmax(sim))
        collapsed = names[top] if sim[top] >= 0.995 else None

        # An oil this close to one of its own constituents earns no place on
        # the atlas: the point repeats what the molecule already says.
        # Lemongrass and clove pass their own self-check and are still
        # dropped, because passing a name test is not the same as carrying
        # information -- lemongrass at 0.992 from Citral tells you what Citral
        # tells you. The composition stays in the data file with its source,
        # so raising the threshold brings it back without redoing the work.
        atlas = sim[top] < args.max_nearest_cosine
        if not atlas:
            trimmed.append((oil, float(sim[top]), names[top]))
            print(f"  TRIM {oil}: cosine {sim[top]:.3f} to {names[top]}, "
                  f"carries nothing the material does not -- off-atlas, "
                  f"still available to a formula")

        word = self_word(oil)
        check = None
        if word and word in desc:
            i = desc.index(word)
            check = {"descriptor": word, "rank": order.index(i) + 1,
                     "of": len(desc), "z": round(float(z[i]), 2),
                     "p": round(float(mix[i]), 3)}

        out["naturals"][oil] = {
            "composition_used": {n: round(float(x), 4) for (n, _), x in zip(res, w)},
            # The spec may carry both of these about itself, and its word wins:
            # `omitted` is the prose about what the reconstruction cannot voice
            # (the resins and gums left in the still), and a `coverage` given
            # there is a citation statistic -- the fraction of one published
            # chromatogram the resolved constituents account for -- where the
            # derived figure only says how much of the stated composition the
            # library happened to hold. Myrrh's nine constituents all resolve
            # and cover 64% of the analysis; 1.0 would be a different claim.
            "coverage": round(float(meta.get("coverage", cov)), 4),
            "omitted_from_library": meta.get("omitted") or missing,
            "source": meta.get("source", ""),
            "note": meta.get("note", ""),
            "percept": [round(float(v), 4) for v in mix],
            "receptors": [round(float(v), 4) for v in rec],
            "lift_top": [{"name": desc[i], "z": round(float(z[i]), 2)}
                         for i in order[:10]],
            "nearest_materials": near,
            "nearest_cosine": round(float(sim[int(np.argmax(sim))]), 4),
            "collapsed_onto": collapsed,
            "self_check": check,
            # Whether this oil earns a point on the atlas. A trimmed oil is
            # still built and still written here, because those are different
            # questions: lemon oil is useless as a point beside D-Limonene and
            # necessary as an ingredient when a formula calls for lemon oil.
            # Leaving it out of the file made any formula that says "Lemon EO"
            # lose that weight, which is the worse error of the two.
            "atlas": bool(atlas),
        }

        tag = (f"{check['descriptor']} rank {check['rank']}/{check['of']}, "
               f"z {check['z']:+.2f}" if check else "no vocabulary word")
        print(f"  {oil}")
        print(f"    {len(res)} constituents, {cov:.0%} of composition   "
              f"SELF-CHECK: {tag}")
        print(f"    top lift: " + ", ".join(
            f"{desc[i]}({z[i]:+.1f})" for i in order[:6]))
        print(f"    nearest:  " + ", ".join(near))
        if missing:
            print(f"    omitted:  {', '.join(missing)}")
        if collapsed:
            print(f"    COLLAPSED onto {collapsed} at cosine "
                  f"{sim[int(np.argmax(sim))]:.4f}: not a distinguishable oil")

    out["trimmed"] = [{"name": n, "nearest_cosine": round(c, 4),
                       "nearest": m} for n, c, m in trimmed]
    (ROOT / args.out).write_text(json.dumps(out, indent=2), encoding="utf-8")
    passed = [k for k, v in out["naturals"].items()
              if v["self_check"] and v["self_check"]["rank"] <= 5]
    scored = [k for k, v in out["naturals"].items() if v["self_check"]]
    on_atlas = [k for k, v in out["naturals"].items() if v["atlas"]]
    print(f"\n[nat] {len(out['naturals'])} built, {len(on_atlas)} on the atlas, {len(trimmed)} trimmed from it but still available to formulas, "
          f"{len(scored)} self-checkable, "
          f"{len(passed)} put their own name in the top 5 of {len(desc)}")
    print(f"[nat] wrote {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
