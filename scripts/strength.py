"""Measured relative strength, and how it corrects a compiled threshold.

The dose conversion turns mass into odour activity by dividing by a detection
threshold. Those thresholds were compiled from published and web sources, some
converted from air-phase figures, and they correlate with the perfumer's own
maximum use levels at only Spearman +0.38 -- against +0.76 for the 28 that
were really measured by Abraham. An individual figure may be out by a factor
of ten, which is larger than the thing the dose optimiser is optimising.

A blotter session measured the quantity that actually matters instead: how
much smell a material delivers at its own use level, relative to the other
materials on the bench. One bin is a factor of three, and
scripts/apply_strength.py turns the bins into a multiplier, shrunk toward no
correction in proportion to its standard error.

A multiplier above 1 means the material delivers more smell per gram than its
use level implies. Odour activity is mass over threshold, so more smell per
gram is the same as a lower effective threshold:

    effective threshold = compiled threshold / strength multiplier

That is the only place the measurement enters. It does not replace the
compiled figure in the inventory, it corrects it where a nose has checked, and
materials with no measurement keep their compiled value unchanged. Both the
CLI and the browser page apply it identically, so a formula dosed in one and
dosed in the other get the same answer.
"""

from __future__ import annotations

import csv
import io
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

DEFAULT_PATH = ROOT / "data" / "inventory" / "strength.csv"


def load_strength(path: Path | str | None = None) -> dict[str, dict]:
    """{material name -> {mult, bin, se, n}} from a filled strength sheet.

    Absent file is normal and means no material has been measured yet, so
    every compiled threshold stands as it is.
    """
    p = Path(path) if path else DEFAULT_PATH
    if not p.is_absolute():
        p = ROOT / p
    if not p.is_file():
        return {}
    out: dict[str, dict] = {}
    for r in csv.DictReader(io.open(p, encoding="utf-8-sig")):
        n = (r.get("name") or "").strip()
        if not n:
            continue

        def num(key, default=None):
            v = (r.get(key) or "").strip()
            try:
                return float(v)
            except ValueError:
                return default

        mult = num("strength_multiplier")
        if not mult or mult <= 0:
            continue
        out[n] = {
            "mult": mult,
            "bin": num("bin_mean", 0.0),
            "bin_raw": num("bin_raw", 0.0),
            "se": num("standard_error_bins"),
            "n": int(num("judgements", 0) or 0),
        }
    return out


def corrected_threshold(name: str, thr: float | None,
                        strength: dict[str, dict]) -> tuple[float | None, bool]:
    """(effective threshold, whether a measurement was applied)."""
    if thr is None:
        return None, False
    s = strength.get(name)
    if not s:
        return thr, False
    return thr / s["mult"], True
