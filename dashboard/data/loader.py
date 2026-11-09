"""Read-only dataset discovery and loading.

Files are looked up (recursively) in the data directory, which is ``data/`` at
the repository root unless the ``LUGGAGE_DATA_DIR`` environment variable says
otherwise. Users can also upload CSVs for the current browser session from the
"Data sources" page; uploads are held in memory only and never written to disk.
"""
from __future__ import annotations

import hashlib
import io
import os
import tomllib
from dataclasses import dataclass, field
from pathlib import Path

import pandas as pd
import streamlit as st

from dashboard.config.schema import DATASETS, PROVENANCE, DatasetSpec
from dashboard.data.mapping import map_columns

REPO_ROOT = Path(__file__).resolve().parents[2]
ENCODINGS = ("utf-8", "utf-8-sig", "cp1252", "latin-1")
UPLOAD_STATE_KEY = "uploaded_datasets"


def data_dir() -> Path:
    env = os.environ.get("LUGGAGE_DATA_DIR")
    return Path(env).expanduser().resolve() if env else REPO_ROOT / "data"


@dataclass
class DatasetResult:
    key: str
    spec: DatasetSpec
    status: str = "missing"  # loaded | missing | empty | error
    source: str = ""
    path: str = ""
    message: str = ""
    raw: pd.DataFrame | None = None  # exactly as read
    df: pd.DataFrame | None = None  # canonical column names (copy)
    renamed: dict = field(default_factory=dict)
    missing_columns: list = field(default_factory=list)
    provenance: str = "unclassified"
    provenance_note: str = ""

    @property
    def ok(self) -> bool:
        return self.status == "loaded" and self.df is not None and len(self.df) > 0

    @property
    def n(self) -> int:
        return 0 if self.df is None else len(self.df)


# ------------------------------------------------------------------ reading
def _read_bytes(content: bytes) -> pd.DataFrame:
    last_err: Exception | None = None
    for enc in ENCODINGS:
        try:
            df = pd.read_csv(io.BytesIO(content), encoding=enc, skipinitialspace=True, low_memory=False)
            # Comma is the contract; fall back to tab / semicolon only for single-column reads
            # (semicolons inside values are multi-select separators, so never sniff blindly).
            if df.shape[1] == 1:
                header = str(df.columns[0])
                for sep in ("\t", ";"):
                    if sep in header:
                        df = pd.read_csv(io.BytesIO(content), encoding=enc, sep=sep, skipinitialspace=True,
                                         low_memory=False)
                        break
            return df
        except UnicodeDecodeError as e:
            last_err = e
        except pd.errors.EmptyDataError:
            return pd.DataFrame()
    raise ValueError(f"Could not decode file with {', '.join(ENCODINGS)}: {last_err}")


@st.cache_data(show_spinner=False, max_entries=64)
def _read_cached(digest: str, content: bytes) -> pd.DataFrame:  # digest keys the cache
    return _read_bytes(content)


def _discover(base: Path) -> dict[str, Path]:
    found: dict[str, Path] = {}
    if not base.exists():
        return found
    for root, dirs, files in os.walk(base):
        dirs[:] = sorted(d for d in dirs if not d.startswith((".", "__")))
        for f in sorted(files):
            if f.lower().endswith(".csv"):
                found.setdefault(f.lower(), Path(root) / f)
    return found


def _manifest(base: Path) -> dict:
    p = base / "sources.toml"
    if not p.exists():
        return {}
    try:
        with open(p, "rb") as fh:
            return tomllib.load(fh)
    except Exception:  # malformed manifest must not break the app
        return {"_error": f"Could not parse {p.name}"}


def _record_level_provenance(df: pd.DataFrame) -> set[str]:
    if "data_source" not in df.columns:
        return set()
    out = set()
    for v in df["data_source"].dropna().astype(str).str.lower().unique():
        if "synth" in v:
            out.add("synthetic")
        elif "transaction" in v or "pilot" in v or "booking" in v:
            out.add("transaction")
        elif "secondary" in v:
            out.add("secondary")
        elif "illustr" in v or "scenario" in v:
            out.add("illustrative")
        elif "primary" in v or "actual" in v or "real" in v or "survey" in v:
            out.add("primary")
    return out


def _resolve_provenance(results: dict[str, DatasetResult], manifest: dict) -> None:
    declared = manifest.get("provenance", {}) if isinstance(manifest, dict) else {}
    for key, r in results.items():
        if key in declared and declared[key] in PROVENANCE:
            r.provenance, r.provenance_note = declared[key], "declared in data/sources.toml"
        elif r.df is not None and (rec := _record_level_provenance(r.df)):
            if len(rec) == 1:
                r.provenance, r.provenance_note = rec.pop(), "from the data_source column"
            else:
                r.provenance = "unclassified"
                r.provenance_note = "MIXED sources in data_source column: " + ", ".join(sorted(rec))
        elif not r.spec.default_provenance.startswith("inherit:"):
            r.provenance = r.spec.default_provenance
            r.provenance_note = ("project default (consumer data described as synthetic in the brief)"
                                 if r.provenance == "synthetic" else
                                 "project default" if r.provenance != "unclassified" else
                                 "not declared — set it in data/sources.toml")
    for key, r in results.items():
        if r.provenance_note == "" and r.spec.default_provenance.startswith("inherit:"):
            parent = results.get(r.spec.default_provenance.split(":", 1)[1])
            r.provenance = parent.provenance if parent else "unclassified"
            r.provenance_note = "model output — inherits the provenance of the consumer dataset"


def load_all() -> dict[str, DatasetResult]:
    """Discover, read and map every dataset in the contract. Never raises."""
    base = data_dir()
    found = _discover(base)
    uploads: dict = st.session_state.get(UPLOAD_STATE_KEY, {})
    results: dict[str, DatasetResult] = {}
    for key, spec in DATASETS.items():
        r = DatasetResult(key=key, spec=spec)
        content: bytes | None = None
        if key in uploads:
            name, content = uploads[key]
            r.source, r.path = f"uploaded this session ({name})", name
        else:
            for fname in spec.filenames:
                if fname.lower() in found:
                    p = found[fname.lower()]
                    try:
                        content = p.read_bytes()
                    except OSError as e:
                        r.status, r.message = "error", f"Cannot read {p.name}: {e}"
                    try:
                        rel = p.relative_to(REPO_ROOT)
                    except ValueError:
                        rel = p
                    r.source, r.path = f"repository file {rel}", str(rel)
                    break
        if content is None:
            if r.status != "error":
                r.status = "missing"
                r.message = f"Expected file not found: {' or '.join(spec.filenames)} in {base.name}/"
            results[key] = r
            continue
        try:
            raw = _read_cached(hashlib.sha1(content).hexdigest(), content)
        except Exception as e:  # noqa: BLE001 — surface any parse failure as a status
            r.status, r.message = "error", f"Could not parse CSV: {e}"
            results[key] = r
            continue
        if raw.empty or len(raw.columns) == 0:
            r.status, r.message, r.raw = "empty", "File was found but contains no records.", raw
            results[key] = r
            continue
        mapped, renamed, missing = map_columns(raw, spec.alias_group or "model", spec.expected_columns)
        r.raw, r.df, r.renamed, r.missing_columns = raw, mapped, renamed, missing
        req_missing = [c for c in spec.required_columns if c not in mapped.columns]
        if req_missing:
            r.status = "error"
            r.message = "Required column(s) missing: " + ", ".join(req_missing)
        else:
            r.status = "loaded"
            r.message = f"{len(mapped):,} records × {len(raw.columns)} columns"
        results[key] = r
    manifest = _manifest(base)
    _resolve_provenance(results, manifest)
    if "_error" in manifest:
        for r in results.values():
            if r.provenance_note.startswith("project default") or "not declared" in r.provenance_note:
                r.provenance_note += f" ({manifest['_error']})"
    return results


def register_upload(key: str, name: str, content: bytes) -> bool:
    """Store an uploaded CSV for this session. Returns True when something changed."""
    uploads = dict(st.session_state.get(UPLOAD_STATE_KEY, {}))
    if uploads.get(key) == (name, content):
        return False
    uploads[key] = (name, content)
    st.session_state[UPLOAD_STATE_KEY] = uploads
    return True


def clear_uploads() -> None:
    st.session_state[UPLOAD_STATE_KEY] = {}
