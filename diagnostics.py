from __future__ import annotations
"""parquet_auditor_plus.py

A rich, safety-first inspection utility for Parquet datasets.

Key features
------------
* **Zero-touch discovery** – point it at files *or* directories; it crawls for
  ``*.parquet`` underneath.
* **Metadata-only fast path** – relies on Parquet footer metadata so even very
  large tables are inspected in milliseconds.
* **Optional deep scan (--deep)** – lazily streams only the columns needed for
  heavy diagnostics (timestamp range, distinct ticker count) so memory usage
  stays low.
* **Human-friendly numbers** courtesy of *humanize*.
* **Compact row-group overview** – quickly reveals writer partitioning.
* **Graceful degradation** – corrupt or missing files are reported and the
  audit continues instead of crashing.

Installation
~~~~~~~~~~~~
```
pip install pyarrow humanize
```

Example
~~~~~~~
```
python parquet_auditor_plus.py data/ --stats --deep -n 5
```
"""

import argparse
import sys
import textwrap
from pathlib import Path
from typing import Iterable, Mapping, Sequence

import humanize  # pretty-print numbers & sizes
import pyarrow as pa
import pyarrow.compute as pc
import pyarrow.parquet as pq

# ---------------------------------------------------------------------------
# Default dataset mapping (override via CLI)
# ---------------------------------------------------------------------------
DEFAULT_DATA_DIR = Path("data")
FILES: Mapping[str, Path] = {
    "primary": DEFAULT_DATA_DIR / "features.parquet",
    "fundamental": DEFAULT_DATA_DIR / "fundamental_features.parquet",
    "tca": DEFAULT_DATA_DIR / "tca_features.parquet",
}

# ---------------------------------------------------------------------------
# Pretty helpers
# ---------------------------------------------------------------------------

def fmt_num(x: int | None) -> str:
    return "-" if x is None else humanize.intcomma(x)


def fmt_size(bytes_: int | None) -> str:
    return "-" if bytes_ is None else humanize.naturalsize(bytes_, binary=True)


def hline(width: int = 80, char: str = "─") -> str:
    return char * width


def print_header(title: str, width: int = 80) -> None:
    print("\n" + hline(width))
    print(title.center(width))
    print(hline(width))


# ---------------------------------------------------------------------------
# Safe IO helpers
# ---------------------------------------------------------------------------

def safe_open_parquet(path: Path) -> pq.ParquetFile | None:
    """Return *ParquetFile* or *None*; emit readable diagnostics on failure."""
    try:
        return pq.ParquetFile(path)
    except (FileNotFoundError, pa.lib.ArrowInvalid, pa.lib.ArrowIOError) as e:
        print(f"⚠️  {path}: {e}")
        return None


# ---------------------------------------------------------------------------
# Metadata-only statistics helpers
# ---------------------------------------------------------------------------

def parquet_column_stats(pfile: pq.ParquetFile) -> dict[str, dict[str, str]]:
    """Collect null-count, min, max for each column using ONLY metadata.

    Statistics are optional in the Parquet spec, so guard accesses.
    """

    out: dict[str, dict[str, str]] = {}
    meta = pfile.metadata
    if meta is None:
        return out

    schema = pfile.schema_arrow
    for i, field in enumerate(schema):
        col_meta = meta.row_group(0).column(i) if meta.num_row_groups else None
        stats = col_meta.statistics if (col_meta and col_meta.statistics) else None

        # Portable null-count: PyArrow ≥ 12 stores it on *statistics*; earlier
        # versions may omit it entirely.
        nulls: int | None
        if stats and hasattr(stats, "null_count") and stats.null_count is not None:  # type: ignore[attr-defined]
            nulls = stats.null_count  # type: ignore[attr-defined]
        else:
            nulls = None

        out[field.name] = {
            "type": str(field.type),
            "nulls": fmt_num(nulls),
            "min": str(stats.min) if (stats and stats.has_min_max) else "-",
            "max": str(stats.max) if (stats and stats.has_min_max) else "-",
        }
    return out


def row_group_overview(pfile: pq.ParquetFile) -> list[str]:
    """Return a per-row-group row count list formatted with commas."""
    meta = pfile.metadata
    if meta is None:
        return []
    return [fmt_num(meta.row_group(i).num_rows) for i in range(meta.num_row_groups)]


# ---------------------------------------------------------------------------
# Deep diagnostics (requires reading data)
# ---------------------------------------------------------------------------

def deep_timestamp_range(
    pfile: pq.ParquetFile, column: str = "event_timestamp"
) -> tuple[str, str] | None:
    schema = pfile.schema_arrow
    if schema.get_field_index(column) == -1:
        return None
    col = pfile.read([column])[column]
    if col.length() == 0:
        return None
    return str(pc.min(col).as_py()), str(pc.max(col).as_py())


def deep_distinct_count(pfile: pq.ParquetFile, column: str = "ticker_id") -> str | None:
    """Return an *approximate* or exact distinct count, depending on PyArrow version.

    Falls back gracefully when neither ``approx_unique`` nor ``count_distinct`` is
    available (old PyArrow releases). In that case the result is computed via a
    Python ``set`` on the *first chunk only* to avoid OOM, so treat it as a hint.
    """

    schema = pfile.schema_arrow
    if schema.get_field_index(column) == -1:
        return None

    col = pfile.read([column])[column]
    if col.length() == 0:
        return "-"

    # Try the modern approximate algorithm first
    if hasattr(pc, "approx_unique"):
        return fmt_num(pc.approx_unique(col).as_py())  # type: ignore[attr-defined]

    # Fallback: exact but potentially heavy – available in many versions
    if hasattr(pc, "count_distinct"):
        return fmt_num(pc.count_distinct(col).as_py())  # type: ignore[attr-defined]

    # Final fallback for *very* old PyArrow: use Python set on first chunk
    first_chunk = col.chunk(0).to_pylist()
    return fmt_num(len(set(first_chunk))) + " (first chunk only)"


# ---------------------------------------------------------------------------
# Table preview helper
# ---------------------------------------------------------------------------

def preview_rows(
    pfile: pq.ParquetFile, cols: Sequence[str] | None, n: int = 3
) -> pa.Table:
    return pfile.read(columns=cols or None, use_threads=True).slice(0, n)


# ---------------------------------------------------------------------------
# File discovery
# ---------------------------------------------------------------------------

def discover_parquet(paths: Iterable[Path], recursive: bool = False) -> list[Path]:
    files: list[Path] = []
    for p in paths:
        if p.is_dir():
            pattern = "**/*.parquet" if recursive else "*.parquet"
            files.extend(sorted(p.glob(pattern)))
        else:
            files.append(p)
    return files


# ---------------------------------------------------------------------------
# Primary audit routine
# ---------------------------------------------------------------------------

def audit_one(
    path: Path,
    sample_n: int = 3,
    show_stats: bool = False,
    deep: bool = False,
):
    print_header(f"▶ {path}")

    pfile = safe_open_parquet(path)
    if pfile is None:
        return

    meta = pfile.metadata
    size = fmt_size(path.stat().st_size) if path.exists() else "-"
    rows = fmt_num(meta.num_rows) if meta else "-"
    row_groups = fmt_num(meta.num_row_groups) if meta else "-"

    print(textwrap.dedent(
        f"""
        File size     : {size}
        Rows          : {rows}
        Row groups    : {row_groups} ({', '.join(row_group_overview(pfile))})
        Schema        :
        {pfile.schema}\n"""
    ))

    # Metadata column stats
    if show_stats:
        stats = parquet_column_stats(pfile)
        if stats:
            print("Column statistics (metadata):")
            for col, s in stats.items():
                print(
                    f"  • {col:<20} nulls={s['nulls']:<10} min={s['min']:<20} max={s['max']:<20} ({s['type']})"
                )
            print()
        else:
            print("(no per-column statistics written by producer)\n")

    # Data-reading diagnostics
    if deep:
        ts_range = deep_timestamp_range(pfile)
        uniq_ticker = deep_distinct_count(pfile)
        if ts_range:
            print(f"event_timestamp range  : {ts_range[0]}  →  {ts_range[1]}")
        if uniq_ticker:
            print(f"Distinct ticker_id     : {uniq_ticker}")
        print()

    # Sample rows
    schema = pfile.schema_arrow
    core_cols = [c for c in ("ticker_id", "event_timestamp", "created_timestamp") if schema.get_field_index(c) != -1]
    if meta and meta.num_rows:
        sample = preview_rows(pfile, core_cols, sample_n)
        print("Sample rows:")
        print(sample.to_pandas())
    else:
        print("(no rows to display)")


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------

def main(argv: Sequence[str] | None = None):
    parser = argparse.ArgumentParser(
        description="Audit Parquet files quickly and safely.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("paths", nargs="*", type=Path, help="Files or directories to inspect")
    parser.add_argument("--sample", "-n", type=int, default=3, help="Rows to preview")
    parser.add_argument("--stats", action="store_true", help="Show per-column metadata statistics")
    parser.add_argument("--deep", action="store_true", help="Compute expensive statistics (min/max, distinct counts)")
    parser.add_argument("--recursive", "-r", action="store_true", help="Recurse into sub-directories")
    args = parser.parse_args(argv)

    targets = discover_parquet(args.paths or FILES.values(), recursive=args.recursive)
    if not targets:
        print("No Parquet files found – specify paths or adjust FILES mapping.")
        sys.exit(1)

    for p in targets:
        audit_one(p, sample_n=args.sample, show_stats=args.stats, deep=args.deep)


if __name__ == "__main__":
    main()