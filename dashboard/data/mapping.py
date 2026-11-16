"""Column-name normalisation and alias mapping (read-only; never touches files)."""
from __future__ import annotations

import re
from collections import Counter

import pandas as pd

from dashboard.config.schema import ALIASES


def normalise_name(name: object) -> str:
    s = str(name).strip().lower()
    s = s.replace("₹", "inr").replace("%", "pct")
    s = re.sub(r"[^0-9a-z]+", "_", s)
    return s.strip("_") or "unnamed"


def map_columns(df: pd.DataFrame, alias_group: str, expected: tuple[str, ...] = ()) -> tuple[pd.DataFrame, dict, list]:
    """Return (renamed copy, {original: canonical}, missing expected columns).

    Matching order: exact canonical name -> listed alias -> questionnaire prefix
    (e.g. ``q11_checkout_gap``) when exactly one column carries that prefix.
    Columns that do not match keep their normalised name (so encoded binary
    multi-select columns such as ``acceptable_venue_types__hotels`` survive).
    """
    aliases = ALIASES.get(alias_group, {})
    canon_names = set(expected) | set(aliases)
    lookup: dict[str, str] = {}
    for canon, alts in aliases.items():
        for a in alts:
            lookup.setdefault(normalise_name(a), canon)

    norm = {c: normalise_name(c) for c in df.columns}
    # de-duplicate normalised names
    seen: Counter = Counter()
    for c, n in list(norm.items()):
        seen[n] += 1
        if seen[n] > 1:
            norm[c] = f"{n}_{seen[n]}"

    taken = {n for n in norm.values() if n in canon_names}
    mapping: dict[str, str] = {}
    for c, n in norm.items():
        if n in canon_names:
            mapping[c] = n
        elif n in lookup and lookup[n] not in taken:
            mapping[c] = lookup[n]
            taken.add(lookup[n])
        else:
            mapping[c] = n

    # questionnaire-number prefixes: q11_xxx -> checkout_gap_hours when unique
    prefix_counts = Counter(m.group(1) for n in mapping.values() if (m := re.match(r"^(q\d+)_", n)))
    for c, n in list(mapping.items()):
        m = re.match(r"^(q\d+)_", n)
        if m and prefix_counts[m.group(1)] == 1 and m.group(1) in lookup:
            canon = lookup[m.group(1)]
            if canon not in taken:
                mapping[c] = canon
                taken.add(canon)

    out = df.rename(columns=mapping)
    missing = [c for c in expected if c not in out.columns]
    return out, {k: v for k, v in mapping.items() if str(k) != v}, missing


def find_encoded_columns(df: pd.DataFrame, field: str, alias_group: str = "consumer") -> list[str]:
    """Columns that look like binary-encoded options of a multi-select question."""
    prefixes = [field] + [normalise_name(a) for a in ALIASES.get(alias_group, {}).get(field, ())]
    cols = []
    for c in df.columns:
        if c == field:
            continue
        for p in prefixes:
            if c.startswith(p + "__") or (c.startswith(p + "_") and len(c) > len(p) + 1):
                if _is_binary(df[c]):
                    cols.append(c)
                break
    return cols


_TRUE = {"1", "1.0", "true", "yes", "y", "t", "selected"}
_FALSE = {"0", "0.0", "false", "no", "n", "f", "", "nan", "none"}


def _is_binary(s: pd.Series) -> bool:
    vals = {str(v).strip().lower() for v in s.dropna().unique()[:50]}
    return bool(vals) and vals <= (_TRUE | _FALSE)


def to_bool(s: pd.Series) -> pd.Series:
    """Map yes/no/1/0/true/false style values to booleans; anything else -> <NA>."""
    def conv(v):
        if pd.isna(v):
            return pd.NA
        t = str(v).strip().lower()
        if t in _TRUE or t.startswith("yes"):
            return True
        if t in _FALSE or t.startswith("no"):
            return False
        return pd.NA
    return s.map(conv).astype("boolean")
