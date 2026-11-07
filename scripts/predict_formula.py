"""Predict what a formula smells like, and place it among the raw materials.

This answers two different questions, and they deserve different amounts of
trust.

*What does it smell of?* Combine the ingredients' predicted percept vectors.
The combination rule has been tested once, in ``eval_mixtures.py``, against the
perfumer's own written verdicts on 30 of their formulas. Noisy-OR beat the
dominant-ingredient baseline there, but the margin was significant only with
four closely related formulas included, so the rule is better than nothing and
not much more than that. Read the descriptor list as a ranking, not as
intensities.

*What is it near in the library?* This is on firmer ground, because it asks
only for a distance between vectors the model produced directly. Two clouds are
reported. Percept space is the space of predicted smell, so a neighbour there
is a material predicted to smell similar. Receptor space is the model's own
output space, and a mixture has no receptor vector of its own -- the one used
here assumes a receptor fires if any ingredient fires it, which is the right
shape at low occupancy and wrong once ligands compete for the same pocket. It
is reported because it places the blend on the published atlas, not because it
is more reliable than the percept neighbours.

The model cannot name a perfume. Nothing in its training data pairs a
composition with a name, so any name here would be a human gloss on the
descriptor ranking, not a prediction.

Input is a CSV with columns ``material,dilution,weight``: the stock strength as
a percentage and the weight of that stock, exactly as a formula sheet records
them. Neat parts are computed from the two, so diluted ingredients are weighted
by how much material they actually contribute.
"""

from __future__ import annotations

import argparse
import csv
import io
import json
import re
import sys
from pathlib import Path

import numpy as np
from scipy.spatial.distance import pdist, squareform

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))

from eval_mixtures import combine  # noqa: E402
from orvector.project import preprocess  # noqa: E402
from orvector.store import Library  # noqa: E402
from parse_formulas import classify  # noqa: E402
from strength import load_strength  # noqa: E402


def read_formula(path: Path):
    rows = []
    for r in csv.DictReader(io.open(path, encoding="utf-8")):
        name = (r.get("material") or "").strip()
        if not name:
            continue
        dil = float(r.get("dilution") or 100.0)
        wt = float(r.get("weight") or 0.0)
        rows.append({"name": name, "dilution": dil, "weight": wt,
                     "neat": wt * dil / 100.0})
    return rows


def centred_cosine_to(P: np.ndarray, q: np.ndarray) -> np.ndarray:
    """Cosine of q to each row of P, after centring both on P's mean."""
    mu = P.mean(0)
    X = P - mu
    X /= np.maximum(np.linalg.norm(X, axis=1, keepdims=True), 1e-12)
    y = q - mu
    y /= max(float(np.linalg.norm(y)), 1e-12)
    return X @ y


def thresholds() -> dict[str, float]:
    """Water-phase detection thresholds in ppb, by InChIKey block.

    These were described as "measured" throughout this project and that was
    wrong. The perfumer compiled them from published and web sources and
    approximated some from air-phase figures, so an individual value may be out
    by a factor of ten or more. At that uncertainty the doses derived from them
    are not calibrated: perturbing one threshold by 10x moves a five-material
    blend by 0.4 to 0.7 in percept space, where the just-noticeable yardstick
    is 0.525. Use them for ordering, not for dosing precision -- except where
    a blotter session has measured the material's relative strength, which is
    applied on top of this figure through scripts/strength.py.
    """
    path = ROOT / "data" / "inventory" / "inventory.csv"
    out: dict[str, float] = {}
    if not path.is_file():
        return out
    for r in csv.DictReader(io.open(path, encoding="utf-8")):
        v = (r.get("water_threshold_ppb") or "").strip()
        k = (r.get("inchikey") or "").strip()
        if v and k:
            try:
                out[k[:14]] = float(v)
            except ValueError:
                pass
    return out


#: Words in an ingredient name that describe the stock rather than the plant.
#: "Bergamot EO", "bergamot oil" and "bergamot essential oil" are one
#: ingredient, and "Neroli accord" is a perfumer's reconstruction of one, which
#: is matched to the oil deliberately: that is the assumption, and the report
#: prints it so it can be disagreed with.
OIL_NOISE = {"eo", "oil", "essential", "absolute", "abs", "accord", "resinoid",
             "concrete", "extract", "co2", "root", "seed", "leaf", "leaves",
             "bark", "wood", "berry", "berries", "flower", "peel", "fruit",
             "expressed", "distilled", "natural", "pure"}


def oil_tokens(name: str) -> list[str]:
    """Lowercase words of a name, with the stock words and punctuation gone."""
    low = re.sub(r"[^a-z0-9]+", " ", name.lower())
    return [w for w in low.split() if w and w not in OIL_NOISE]


#: How a natural was made. The words are noise for finding the plant, but two
#: forms of one plant are different materials: "Labdanum Resinoid" on the
#: perfumer's shelf was matched to the library's "Labdanum Absolute", so the
#: engine offered an absolute the perfumer does not have (2026-11-07).
OIL_FORMS = {"absolute": "absolute", "abs": "absolute", "resinoid": "resinoid",
             "concrete": "concrete", "co2": "co2", "eo": "oil", "essential": "oil"}


def oil_form(name: str) -> str | None:
    """The form a name states, or None when it states none."""
    low = re.sub(r"[^a-z0-9]+", " ", name.lower()).split()
    forms = {OIL_FORMS[w] for w in low if w in OIL_FORMS}
    return forms.pop() if len(forms) == 1 else None


def owned_names() -> set[str]:
    """Material names the perfumer has on the shelf, from the inventory.

    A neighbour nobody owns cannot be smelled against the blend, so it cannot
    settle anything. utf-8-sig because the inventory file carries a BOM, and
    reading it as plain utf-8 corrupts the first column name and silently
    returns an empty set.
    """
    path = ROOT / "data" / "inventory" / "inventory.csv"
    if not path.is_file():
        return set()
    out = set()
    for r in csv.DictReader(io.open(path, encoding="utf-8-sig")):
        n = (r.get("name") or r.get("material") or "").strip()
        if n:
            out.add(n)
    return out


def load_oils(store: str):
    """Essential oils built from GC composition, for one receptor space.

    Oils are built per store because each encoder has its own receptor space.
    The percept vector does not depend on the store -- the percept model reads
    fingerprints -- but it is read from the same file, so a missing file means
    no oils rather than oils placed in the wrong space.
    """
    path = (ROOT / "artifacts" / "naturals.json" if store == "library"
            else ROOT / "artifacts" / f"naturals_{store}.json")
    if not path.is_file():
        return {}
    blob = json.loads(path.read_text(encoding="utf-8"))
    return {k: v for k, v in blob["naturals"].items()}


def oil_defaults() -> dict:
    """Which oil to take when several chemotypes of one plant tie.

    "Lavender Maillette France" matched three lavenders equally well and was
    dropped, which is the right behaviour for a genuine ambiguity and the
    wrong one here: the French angustifolia entry has said in prose from the
    start that it is the default for an unqualified lavender. The data file
    now says so in a field, so the code can act on it.
    """
    path = ROOT / "data" / "naturals" / "compositions.json"
    out: dict[str, str] = {}
    if not path.is_file():
        return out
    blob = json.loads(path.read_text(encoding="utf-8"))
    for oil, meta in blob.get("naturals", {}).items():
        for word in meta.get("default_for", []):
            out[word.lower()] = oil
    return out


def match_oil(name: str, oils: list[str]):
    """Resolve "Bergamot EO" to "Bergamot EO (Turkish)", or report ambiguity.

    Matching is on whole words after the stock words are stripped, and the
    oil's head word has to be one of the ingredient's words -- so "Cedar
    Virginia" reaches "Cedarwood EO (Virginia)" through a prefix on the head
    and the shared qualifier, while "Lavandin" cannot reach a lavender.

    An ingredient that matches several oils equally well is not resolved. Four
    lavender chemotypes are in the file precisely because they are different
    materials, so picking one for a formula that says only "lavender" would be
    inventing the answer; the caller is told to name which, and the weight is
    dropped until it does.
    """
    want = oil_tokens(name)
    if not want:
        return None, []
    want_form = oil_form(name)
    # Two passes, and the order matters. A prefix match is what carries
    # "Cedar Virginia" to "Cedarwood EO (Virginia)", and it is also what
    # carried "Lemon EO" to "Lemongrass EO" and then declared the pair
    # ambiguous, losing the ingredient. An exact head word beats a prefix of
    # one, so exact matches are taken alone whenever any exist.
    #
    # Both sides of a prefix comparison must be at least three characters.
    # Without that, "Aldehyde C-16" tokenises to include a bare "c", which is
    # a prefix of cedarwood, clary, clove and cinnamon, so an aldehyde was
    # reported as matching four essential oils equally well and dropped.
    for exact in (True, False):
        scored = []
        for oil in oils:
            toks = oil_tokens(oil)
            if not toks:
                continue
            # Both name a form and they differ: a different material.
            have_form = oil_form(oil)
            if want_form and have_form and want_form != have_form:
                continue
            head = toks[0]
            if exact:
                if head not in want:
                    continue
            # A prefix match is for inflections -- mandarin/mandarine,
            # petitgrain/petitgrains -- so the two words may differ by a
            # couple of characters and no more. Without the length bound,
            # "pepper" matched "peppermint" and black pepper oil silently
            # became peppermint: a swap rather than a drop, and a swap is
            # invisible to whoever reads the result.
            elif not any(len(w) >= 3 and len(head) >= 3
                         and abs(len(w) - len(head)) <= 2
                         and (w.startswith(head) or head.startswith(w))
                         for w in want):
                continue
            overlap = sum(1 for w in toks[1:] if w in want)
            scored.append((overlap, oil))
        if scored:
            best = max(s for s, _ in scored)
            hits = [o for s, o in scored if s == best]
            if len(hits) > 1:
                dflt = oil_defaults()
                for w in want:
                    if w in dflt and dflt[w] in hits:
                        return dflt[w], hits
            return (hits[0] if len(hits) == 1 else None), hits
    return None, []


def logit(p: np.ndarray) -> np.ndarray:
    q = np.clip(p, 1e-6, 1 - 1e-6)
    return np.log(q / (1 - q))


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("formula", help="CSV with material,dilution,weight")
    ap.add_argument("--percepts", default="artifacts/percepts_fp3.json")
    ap.add_argument("--store", default="library")
    ap.add_argument("--rule", default="logit-mean",
                    choices=["logit-mean", "noisy-or", "weighted-mean", "max",
                             "dominant"],
                    help="how the ingredients' percept vectors are combined. "
                         "logit-mean is the project's rule for a blend's "
                         "character: the simplifier mixes by it, the atlas "
                         "places a blend by it, and the engine mixes its "
                         "proposals by it. noisy-or (the union) and "
                         "weighted-mean are kept for comparison -- the union "
                         "preserves a one-sided character but saturates and "
                         "measures ingredient count; the probability mean is "
                         "not what any published surface uses")
    ap.add_argument("--weight", default="mass", choices=["mass", "oav"],
                    help="mass: neat share. oav: neat share / measured "
                         "detection threshold, so traces of strong materials "
                         "count for what they do")
    ap.add_argument("--top", type=int, default=12)
    ap.add_argument("--null", type=int, default=3000,
                    help="random blends to score the descriptor lift "
                         "against; the raw values are base-rate dominated")
    ap.add_argument("--neighbours", type=int, default=8)
    ap.add_argument("--out", default=None)
    ap.add_argument("--no-oils", action="store_true",
                    help="drop essential oils instead of building them from "
                         "their GC composition")
    ap.add_argument("--viewer-point", default=None,
                    help="write a query file that export_viewer_data.py "
                         "--query injects, to see where the blend lands")
    ap.add_argument("--stores", default="library",
                    help="receptor spaces to write a vector for with "
                         "--viewer-point; one per encoder on the atlas, and "
                         "there is one encoder since the MPNN was removed")
    ap.add_argument("--owned", action="store_true",
                    help="also list the nearest materials the perfumer "
                         "actually stocks, which are the only ones a bench "
                         "comparison can use")
    ap.add_argument("--compare", default=None,
                    help="a point file from an earlier --viewer-point run: "
                         "reports how far this blend is from that one")
    ap.add_argument("--place", default="logit-mean",
                    choices=["logit-mean", "union", "mean"],
                    help="how the injected point is positioned. logit-mean "
                         "averages in the space the atlas is built from, so "
                         "the blend sits among the materials. union is "
                         "noisy-OR, which fires every receptor any ingredient "
                         "fires and therefore lands on the rim regardless of "
                         "what the blend smells of")
    ap.add_argument("--label", default=None,
                    help="name for the injected point (default: the filename)")
    args = ap.parse_args()

    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    except Exception:  # noqa: BLE001
        pass

    pj = json.loads((ROOT / args.percepts).read_text(encoding="utf-8"))
    desc = pj["descriptors"]
    P = {n: np.asarray(r, float) for n, r in zip(pj["molecules"], pj["probs"])}

    entries, logits, scores = Library(name=args.store).load()
    lib_names = [e.name for e in entries]
    lib_set = set(lib_names)
    S = {n: s for n, s in zip(lib_names, scores)}

    # Oils join the material set here, with a percept vector and a receptor
    # vector of their own, so an ingredient that names one is built rather
    # than dropped. On a formula written in naturals that is the difference
    # between reading a quarter of the weight and reading nearly all of it.
    oils = {} if args.no_oils else load_oils(args.store)
    oil_names = list(oils)
    for k, v in oils.items():
        P[k] = np.asarray(v["percept"], float)
        S[k] = np.asarray(v["receptors"], float)

    fpath = Path(args.formula)
    rows = read_formula(fpath if fpath.is_absolute() else ROOT / fpath)
    total_neat = sum(r["neat"] for r in rows)

    thr = thresholds()
    strength_meas = load_strength()
    strength_used = []
    block = {e.name: e.inchikey[:14] for e in entries}
    resolved, missing, ambiguous = [], [], []
    for r in rows:
        kind, name = classify(r["name"], lib_set)
        r["is_oil"] = False
        if kind == "solvent":
            r["resolved"] = None
            r["threshold"] = None
            r["why"] = "solvent"
            missing.append(r)
            continue
        if kind == "material" and name in P:
            r["resolved"] = name
            r["threshold"] = thr.get(block.get(name, ""))
            # A measured strength corrects the compiled figure: more smell per
            # gram is the same as a lower effective threshold. Materials with
            # no measurement keep their compiled value untouched.
            st = strength_meas.get(name)
            if st and r["threshold"]:
                r["threshold"] /= st["mult"]
                strength_used.append((name, st["mult"], st["n"]))
            resolved.append(r)
            continue
        hit, cands = match_oil(r["name"], oil_names)
        if hit:
            r["resolved"] = hit
            r["is_oil"] = True
            # An oil has no single detection threshold, so odour-activity
            # weighting cannot see one; mass weighting is what it gets.
            r["threshold"] = None
            resolved.append(r)
            continue
        if cands:
            ambiguous.append((r["name"], cands))
        r["resolved"] = None
        r["threshold"] = None
        missing.append(r)

    # Coverage of the odorous weight, not of the bottle: a formula for skin
    # application is mostly carrier, and one was 88% MCT oil, which made a
    # fully resolved formula read as 11% covered.
    solvent_neat = sum(r["neat"] for r in missing
                       if r.get("why") == "solvent")
    odorous = max(total_neat - solvent_neat, 1e-9)
    cov = sum(r["neat"] for r in resolved) / odorous
    print(f"[formula] {fpath.name}: {len(rows)} ingredients, "
          f"{total_neat:.2f} g neat material")
    print(f"  resolved {len(resolved)}, covering {cov:.1%} of the neat weight\n")
    print(f"  {'ingredient':28s} {'stock':>6s} {'g':>8s} {'parts/1000':>11s}  "
          f"resolved to")
    print("  " + "-" * 88)
    for r in sorted(rows, key=lambda x: -x["neat"]):
        tag = r["resolved"] or "-- not in the library, dropped --"
        if r.get("is_oil"):
            tag += "   [EO built from GC]"
        print(f"  {r['name'][:28]:28s} {r['dilution']:5.0f}% {r['weight']:8.3f} "
              f"{1000 * r['neat'] / total_neat:11.2f}  {tag}")
    if missing:
        lost = sum(r["neat"] for r in missing) / total_neat
        print(f"\n  dropped {len(missing)} ingredient(s), {lost:.1%} of the "
              f"neat weight")
    for nm, cands in ambiguous:
        print(f"  [oil] {nm!r} matches {len(cands)} oils equally well, so it "
              f"was dropped rather than guessed:")
        for c in cands:
            print(f"        {c}")
        print(f"        name the one you mean in the formula to resolve it")

    if len(resolved) < 3:
        raise SystemExit("too few resolvable ingredients to combine")

    w = np.array([r["neat"] for r in resolved], float)
    if args.weight == "oav":
        # Odour activity: neat share divided by the measured detection
        # threshold, so a trace of something enormously strong counts for what
        # it does rather than for what it weighs. The thresholds are water-phase
        # and assembled from mixed sources, which is the wrong phase for
        # sniffing a blend and not very accurate either; and detection
        # threshold is not perceived contribution at the dose used, which is
        # what the rule actually wants. Ethyl Maltol shows the gap: a weak
        # threshold, yet a few percent of it is plainly sweet on the bench.
        if strength_used:
            print(f"\n  [strength] measured on a blotter, so the compiled "
                  f"threshold is corrected for {len(strength_used)} "
                  f"ingredient(s):")
            for nm, mult, cnt in sorted(strength_used, key=lambda t: -t[1]):
                print(f"    {nm:28s} x{mult:5.2f} strength "
                      f"({cnt} judgement{'s' if cnt != 1 else ''})"
                      f"{'  dose less' if mult > 1 else '  dose more'}")
        have = [r for r in resolved if r["threshold"] is not None]
        if len(have) < len(resolved):
            lost = [r["resolved"] for r in resolved if r["threshold"] is None]
            print(f"\n  [oav] no measured threshold for {', '.join(lost)}; "
                  f"dropped from the odour-activity weighting")
        if len(have) < 3:
            raise SystemExit("too few ingredients have a measured threshold")
        resolved = have
        w = np.array([r["neat"] / r["threshold"] for r in resolved], float)
    w = w / w.sum()
    V = np.stack([P[r["resolved"]] for r in resolved])
    mix = combine(args.rule, V, w)

    order = np.argsort(-mix)[: args.top]
    print(f"\n  predicted descriptors ({args.rule}, ranking only):")
    for i in order:
        bar = "#" * int(round(28 * mix[i] / max(mix.max(), 1e-9)))
        print(f"    {desc[i]:16s} {mix[i]:.3f}  {bar}")

    # Lift against a null of random blends of the same size at the same
    # weights. This is the only informative reading of the list above: the raw
    # values are dominated by how common each label is in GS-LF, so a random
    # ten-material blend scores floral 0.90 and sweet 0.95, and a formula that
    # scores floral 0.99 has therefore said almost nothing. What the null
    # cannot fix is that formula and null are combined by the same rule, so a
    # bias in the rule survives the subtraction.
    pool = np.stack([P[n] for n in list(pj["molecules"]) + oil_names])
    rng = np.random.default_rng(0)
    null = np.stack([
        combine(args.rule,
                pool[rng.choice(len(pool), len(resolved), replace=False)], w)
        for _ in range(args.null)])
    z = (mix - null.mean(0)) / np.maximum(null.std(0), 1e-9)
    print(f"\n  lift against {args.null} random blends of {len(resolved)} "
          f"materials at these weights:")
    for i in np.argsort(-z)[: args.top]:
        bar = "#" * int(round(min(14, max(0.0, float(z[i])))))
        print(f"    {desc[i]:16s} z {z[i]:+6.2f}   p {mix[i]:.3f}  {bar}")

    print("\n  what drives the top descriptors:")
    for i in order[:5]:
        share = w * V[:, i]
        j = np.argsort(-share)[:3]
        who = ", ".join(f"{resolved[k]['resolved']} ({V[k, i]:.2f})" for k in j)
        print(f"    {desc[i]:16s} <- {who}")

    # Oils are candidate neighbours too: an oil is a point on the atlas, and
    # "this blend is nearest bergamot oil" is a more useful sentence than the
    # nearest single molecule when the blend is mostly naturals.
    pnames = list(pj["molecules"]) + oil_names
    PM = np.stack([P[n] for n in pnames])
    sim = centred_cosine_to(PM, mix)
    own = {r["resolved"] for r in resolved}
    print("\n  nearest raw materials in percept space (predicted smell):")
    for k in np.argsort(-sim)[: args.neighbours]:
        mark = " *" if pnames[k] in own else "  "
        print(f"   {mark} {sim[k]:+.3f}  {pnames[k]}")
    print("    * = an ingredient of this formula")

    if args.owned:
        # The bench test needs a bottle. An oil counts as owned when the
        # inventory names it, since the inventory lists Bergamot BF and Lemon
        # FCF alongside the single molecules.
        own_shelf = owned_names()
        shelf = [n for n in pnames if n in own_shelf
                 or any(o.lower().startswith(n.split(" EO")[0].lower())
                        for o in own_shelf)]
        if shelf:
            idx = {n: i for i, n in enumerate(pnames)}
            ss = sorted(shelf, key=lambda n: -sim[idx[n]])
            print("")
            print(f"  nearest of the {len(shelf)} materials on the shelf "
                  f"(what a blotter comparison can use):")
            for n in ss[: args.neighbours]:
                mark = " *" if n in own else "  "
                print(f"   {mark} {sim[idx[n]]:+.3f}  {n}")

    def receptor_mix(score_of) -> np.ndarray:
        """Noisy-OR over the ingredients' receptor vectors in one space.

        A receptor fires if any ingredient fires it, scaled by weight. Right
        at low occupancy, wrong once two ligands compete for one pocket -- the
        caveat that applies to every mixture here. It places the blend; it is
        not an argument about it.
        """
        pr = np.stack([score_of(r["resolved"]) for r in resolved])
        ww = np.clip(w / w.max(), 0, 1)[:, None]
        return 1.0 - np.prod(1.0 - ww * np.clip(pr, 0, 1 - 1e-6), axis=0)

    def placement(score_of) -> np.ndarray:
        """The receptor vector the blend is drawn at, in one space.

        Three rules, three different answers, and the difference is geometry
        rather than smell. A 28-ingredient formula under noisy-OR sits 62 units
        from the centre of a cloud whose materials sit 14.6 out on average,
        because a union fires every receptor any ingredient fires: that
        distance measures ingredient count, not character. Averaging the raw
        probabilities is no better -- it produces a flat vector no single
        molecule resembles. Averaging the logits puts the blend 14.5 from the
        centre, which is exactly where a material sits, and next to French
        lavender and bergamot, which is what percept space says too.

        So logit-mean is the default for a blend. The atlas is built by taking
        logits and reducing them, so averaging there is averaging in the
        space the picture is actually drawn in. The oils keep the union rule
        and are mostly unharmed by it -- with five to eight constituents the
        inflation is small -- but Cedarwood EO, at 152.8 total activation
        against a library median of 5.5, shows the same artefact starting.
        """
        pr = np.stack([score_of(r["resolved"]) for r in resolved])
        if args.place == "union":
            ww = np.clip(w / w.max(), 0, 1)[:, None]
            return 1.0 - np.prod(1.0 - ww * np.clip(pr, 0, 1 - 1e-6), axis=0)
        if args.place == "mean":
            return (w[:, None] * pr).sum(0)
        z = (w[:, None] * logit(pr)).sum(0)
        return 1.0 / (1.0 + np.exp(-z))

    rec_mix = placement(lambda n: S[n])
    rec_names = lib_names + oil_names
    RS = np.vstack([scores] + ([np.stack([S[k] for k in oil_names])]
                               if oil_names else []))
    red, _ = preprocess(logit(np.vstack([RS, rec_mix[None, :]])),
                        n_components=20)
    D = squareform(pdist(red))[-1, :-1]
    med = float(np.median(pdist(red[:-1])))
    print(f"\n  nearest raw materials in receptor space, placed by {args.place} (median library "
          f"separation {med:.2f}):")
    for k in np.argsort(D)[: args.neighbours]:
        mark = " *" if rec_names[k] in own else "  "
        print(f"   {mark} {D[k]:6.2f}  {rec_names[k]}")

    if args.compare:
        other = json.loads((ROOT / args.compare).read_text(encoding="utf-8"))
        q = other["queries"][0]
        o_percept = np.asarray(q["percept"], float)
        # Two blends combined by different rules are not comparable at all:
        # noisy-OR puts a large blend 4.8 from the average point and a mean
        # puts the same blend 0.3 from it, so the comparison would be between
        # two rules rather than two formulas.
        if q.get("rule") and q["rule"] != args.rule:
            print(f"  [compare] {q['name']} was combined with "
                  f"{q['rule']} and this run uses {args.rule}; the numbers "
                  f"below compare the rules, not the formulas. Re-run with "
                  f"--rule {q['rule']}.")
        # Centred cosine in percept space, against the same library mean the
        # neighbour lists use, so the number is comparable to them.
        mu = PM.mean(0)
        a, b = mix - mu, o_percept - mu
        cos = float(a @ b / max(np.linalg.norm(a) * np.linalg.norm(b), 1e-12))
        print("")
        print(f"  against {q['name']}:")
        print(f"    percept-space cosine {cos:+.3f}  "
              f"(this formula's nearest library material is "
              f"{float(sim.max()):+.3f})")
        # Cosine alone is misleading for blends of different sizes, in both
        # directions. Under noisy-OR a thirty-material blend saturates: nearly
        # every descriptor is pinned near 1, so two unrelated thirty-material
        # blends reach cosine +0.94 by both sitting against the ceiling. Under
        # a mean the opposite happens -- the blend collapses towards the
        # library's average point, and cosine normalises away exactly that
        # shrinkage. So the distance from the average point is printed too: it
        # is what says whether a blend still has a character of its own. A
        # single material sits about 1.2 out; three independent random
        # thirty-material blends sit 0.20 to 0.31 out under a mean, which is
        # this model's version of olfactory white.
        mu_d = float(np.linalg.norm(mix - mu))
        o_d = float(np.linalg.norm(o_percept - mu))
        print(f"    distance from the library's average point: this blend "
              f"{mu_d:.2f}, that one {o_d:.2f}, median single material "
              f"{float(np.median(np.linalg.norm(PM - mu, axis=1))):.2f}")
        if args.store in q.get("receptors", {}):
            o_rec = np.asarray(q["receptors"][args.store], float)
            red2, _ = preprocess(
                logit(np.vstack([RS, rec_mix[None, :], o_rec[None, :]])),
                n_components=20)
            d2 = float(np.linalg.norm(red2[-1] - red2[-2]))
            print(f"    receptor-space distance {d2:.2f}  "
                  f"(median library separation {med:.2f}, and the nearest "
                  f"library material to this blend is {float(D.min()):.2f})")

    if args.viewer_point:
        label = args.label or f"BLEND: {fpath.stem}"
        recs = {}
        for store in [s.strip() for s in args.stores.split(",") if s.strip()]:
            ents_s, _, sc_s = Library(name=store).load()
            m = {e.name: s for e, s in zip(ents_s, sc_s)}
            m.update({k: np.asarray(v["receptors"], float)
                      for k, v in load_oils(store).items()})
            lost = [r["resolved"] for r in resolved if r["resolved"] not in m]
            if lost:
                print(f"  [point] {store} cannot place "
                      f"{', '.join(sorted(set(lost)))}; skipped")
                continue
            recs[store] = [round(float(v), 4)
                           for v in placement(lambda n: m[n])]
        built = ", ".join(
            f"{r['resolved']} {100 * r['neat'] / total_neat:.1f}%"
            for r in sorted(resolved, key=lambda x: -x["neat"])[:6])
        blob = {"queries": [{
            "name": label,
            "built_from": f"{len(resolved)} of {len(rows)} ingredients, "
                          f"{cov:.0%} of the neat weight: {built}, ...",
            "coverage": round(cov, 4),
            "rule": args.rule,
            "placement": args.place,
            "weighting": args.weight,
            "ingredients": [
                {"name": r["resolved"],
                 "pct": round(100 * r["neat"] / total_neat, 2),
                 "oil": bool(r.get("is_oil"))}
                for r in sorted(resolved, key=lambda x: -x["neat"])],
            "dropped": [r["name"] for r in
                        sorted(missing, key=lambda x: -x["neat"])],
            "percept": [round(float(v), 4) for v in mix],
            "receptors": recs,
        }]}
        qp = ROOT / args.viewer_point
        qp.parent.mkdir(parents=True, exist_ok=True)
        qp.write_text(json.dumps(blob, indent=2), encoding="utf-8")
        print(f"[point] wrote {args.viewer_point} for {', '.join(recs)}")

    if args.out:
        rep = {"formula": fpath.name, "rule": args.rule,
               "coverage": round(cov, 4),
               "ingredients": [{"name": r["name"], "resolved": r["resolved"],
                                "parts_per_1000": round(
                                    1000 * r["neat"] / total_neat, 2)}
                               for r in rows],
               "descriptors": [{"name": desc[i], "p": round(float(mix[i]), 4)}
                               for i in np.argsort(-mix)[:30]],
               "percept_neighbours": [
                   {"name": pnames[k], "cos": round(float(sim[k]), 4)}
                   for k in np.argsort(-sim)[:15]],
               "receptor_neighbours": [
                   {"name": lib_names[k], "d": round(float(D[k]), 3)}
                   for k in np.argsort(D)[:15]]}
        (ROOT / args.out).write_text(json.dumps(rep, indent=2), encoding="utf-8")
        print(f"\n[formula] wrote {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
