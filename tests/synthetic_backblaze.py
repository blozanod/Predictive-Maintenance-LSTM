"""Tiny synthetic Backblaze Drive Stats generator, in the REAL shipped on-disk format.

Lets the CPU tests exercise ``src.datasets.backblaze`` without downloading the quarterly
archives (~10^2 files x ~10^5 rows per quarter, tens of GB across the record). What
"real format" means here, and why each detail matters:

  * ONE CSV **per day**, named ``YYYY-MM-DD.csv``, written into a quarter-archive
    subdirectory (``data_Q1_2024/``) rather than flat -- the shipped ZIPs are NOT flat
    and NOT consistently nested (pre-2016 archives put the days under a year folder),
    so the loader globs ``**/????-??-??.csv`` recursively. ``junk=True`` also drops the
    ``__MACOSX/`` shadow tree and a ``.DS_Store`` the archives commonly carry.
  * **CRLF** line endings and a header row, optionally with a **UTF-8 BOM**
    (``bom=True``) -- some redistributions carry one, and it lands in the first column
    NAME, which is exactly where a by-name column selector breaks.
  * A **configurable metadata prefix width** (``meta_width`` 5 / 8 / 11): the 5 columns
    that were always there, plus the Q2-2023 additions (``vault_id, pod_id,
    is_legacy_format``) and the Q3-2023 ones (``datacenter, cluster_id, pod_slot_num``).
    This is the schema-drift trap: the column COUNT changes across quarters, so a
    positional reader silently reads the wrong attribute.
  * SMART columns as ``smart_N_normalized`` / ``smart_N_raw`` PAIRS in ASCENDING
    attribute order -- new attributes are INSERTED in that order in the real files, not
    appended. ``smart_attrs`` sets which exist; ``omit_columns`` deletes named columns
    outright, which is how the "requested SMART column is absent" fail-loud branch is
    exercised.
  * **Model-conditional SMART availability**: ``empty_attrs`` names, per model, the
    attributes that model never populates -- written as EMPTY STRINGS (the real
    encoding), not zeros. In the shipped corpus a model populates ~17-22 of the 93
    attributes and 187/188/193 are absent on several HGST models.
  * A ``capacity_bytes = -1`` row (``bad_capacity``): Backblaze's own guidance is that
    such a row is unreliable in full, so the loader drops it -- which punches a
    one-day hole in that drive's run.
  * A fleet mixing **failed** drives (a terminal ``failure=1`` row, then the drive is
    gone) with **right-censored** survivors, including one that simply VANISHES
    mid-window (retired/migrated: last row ``failure=0``), one with a multi-day
    collection GAP, and one too short to window. Telling those apart is the entire
    point of this milestone.

Sizes are down-scaled (a few tens of drives over a few tens of days) but every
structural property above is preserved verbatim.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np

from src.config import BACKBLAZE_META_COLUMNS

# The three metadata prefixes a loader meets across quarters, keyed by WIDTH. The five
# base columns were always there; Q2 2023 added vault/pod/legacy and Q3 2023 the
# datacenter trio. Only the first five are load-bearing for the loader (it counts the
# leading non-SMART columns and checks the width against the documented set), so the
# exact order of the extras is a fixture detail.
BACKBLAZE_META_LAYOUTS = {
    5: tuple(BACKBLAZE_META_COLUMNS),
    8: tuple(BACKBLAZE_META_COLUMNS) + ("vault_id", "pod_id", "is_legacy_format"),
    11: tuple(BACKBLAZE_META_COLUMNS) + ("vault_id", "pod_id", "is_legacy_format",
                                         "datacenter", "cluster_id", "pod_slot_num"),
}

# Attributes the fixture emits, ASCENDING (the real insertion order). Covers every
# member of ``config.BACKBLAZE_DEFAULT_SMART`` plus 1 (a huge vendor-encoded raw
# counter, ~1.2e11 -- it does not survive float32 storage, which is why the parsed-frame
# cache keeps float64) and 193 (load cycle count, absent on some models).
SYNTHETIC_SMART_ATTRS = (1, 5, 9, 187, 188, 193, 194, 197, 198)

# Two high-volume models, one Seagate + one HGST, matching config.BACKBLAZE_DEFAULT_MODELS.
SYNTHETIC_MODELS = ("ST12000NM0008", "HGST HMS5C4040ALE640")

# Attributes each model NEVER populates (written as empty strings). Availability is
# model-conditional in the real corpus -- 187/188 in particular are absent on several
# HGST models, which is why model scoping has to come before channel selection.
SYNTHETIC_EMPTY_ATTRS = {"HGST HMS5C4040ALE640": (187, 188, 193)}

# Per-model advertised capacity (bytes), verbatim shapes from the real files.
_CAPACITY = {"ST12000NM0008": "12000138625024", "HGST HMS5C4040ALE640": "4000787030016"}
_DEFAULT_CAPACITY = "8001563222016"

# Values for the optional metadata columns; never read by the loader, but they must be
# present and plausible so the width check sees a real 8- or 11-column prefix.
_META_EXTRAS = {"vault_id": "1234", "pod_id": "5", "is_legacy_format": "0",
                "datacenter": "sac0", "cluster_id": "7", "pod_slot_num": "3"}

# (serial suffix, day offset) whose row carries capacity_bytes = -1. Placed on a
# long-lived survivor so the drop leaves a ONE-DAY hole rather than shortening a run.
DEFAULT_BAD_CAPACITY = ("S0V3", 2)

# How many days before the end of a failed drive's life its SMART counters start moving.
_RAMP_DAYS = 10


def synthetic_smart_columns(attrs=SYNTHETIC_SMART_ATTRS) -> list:
    """``smart_N_normalized`` / ``smart_N_raw`` pairs in ascending attribute order --
    the real header layout, into which new attributes are INSERTED, not appended."""
    return [f"smart_{attr}_{kind}" for attr in attrs
            for kind in ("normalized", "raw")]


def drive_specs(models=SYNTHETIC_MODELS, n_days: int = 40, n_failed: int = 2,
                n_survivors: int = 5) -> list:
    """The synthetic fleet: one dict per drive with its observed day OFFSETS and whether
    it ends in a real failure.

    Per model: ``n_failed`` drives that die on different days (terminal ``failure=1``),
    then ``n_survivors`` right-censored ones -- of which the first VANISHES halfway
    through (retired/migrated, last row ``failure=0``), the second carries an 8-day
    collection GAP, the third is only 3 days long (too short to window), and the rest run
    to the end of the window. Deterministic: no RNG."""
    specs = []
    for model_index, model in enumerate(models):
        for i in range(n_failed):
            # Different death days, so failures are not all on the window's last day.
            last = n_days - 1 - 4 * i
            specs.append({"serial": f"S{model_index}F{i}", "model": model,
                          "days": np.arange(0, last + 1), "failed": True,
                          "kind": "failed"})
        for i in range(n_survivors):
            days, kind = np.arange(0, n_days), "full"
            if i == 0:
                days, kind = np.arange(0, n_days // 2), "vanished"
            elif i == 1:
                days = np.setdiff1d(np.arange(0, n_days), np.arange(5, 13))
                kind = "gapped"
            elif i == 2:
                days, kind = np.arange(0, 3), "short"
            specs.append({"serial": f"S{model_index}V{i}", "model": model,
                          "days": days, "failed": False, "kind": kind})
    return specs


def _smart_raw(attr: int, index: int, to_end: int, failed: bool) -> str:
    """One drive-day's RAW value for one attribute, as the file's own decimal string.

    ``index`` is the day's position in the drive's life and ``to_end`` the days left in
    it. Failed drives ramp their error counters over the last ``_RAMP_DAYS`` days, so
    there is genuine, learnable pre-failure signal; survivors stay flat.

    Deliberately RNG-free: a drive-day's value must depend only on that drive-day, so
    two archives written with DIFFERENT attribute sets (the schema-drift tests) carry
    byte-identical values for the attributes they share."""
    ramp = max(0, _RAMP_DAYS - to_end) if failed else 0
    if attr == 1:      # vendor-encoded read error rate: ~1.2e11, beyond float32
        return str(117_000_000_000 + 7 * index)
    if attr == 5:      # reallocated sectors
        return str(8 * ramp)
    if attr == 9:      # power-on hours
        return str(24 * (index + 1))
    if attr == 187:    # reported uncorrectable errors
        return str(3 * ramp)
    if attr == 188:    # command timeout
        return "0"
    if attr == 193:    # load cycle count
        return str(1000 + 3 * index)
    if attr == 194:    # temperature (degC), with a deterministic wobble
        return str(28 + ramp + (7 * index) % 3)
    if attr == 197:    # current pending sectors
        return str(2 * ramp)
    return str(ramp)   # 198: offline uncorrectable


def write_synthetic_backblaze(
    data_dir,
    start: str = "2024-01-01",
    n_days: int = 40,
    models=SYNTHETIC_MODELS,
    n_failed: int = 2,
    n_survivors: int = 5,
    meta_width: int = 5,
    smart_attrs=SYNTHETIC_SMART_ATTRS,
    empty_attrs=SYNTHETIC_EMPTY_ATTRS,
    omit_columns=(),
    subdir: str = "data_Q1_2024",
    junk: bool = True,
    bom: bool = False,
    bad_capacity=DEFAULT_BAD_CAPACITY,
    all_bad_capacity: bool = False,
    seed: int = 0,
) -> Path:
    """Write one real-format daily-CSV archive under ``data_dir`` and return the folder
    the day files landed in.

    ``meta_width`` (5/8/11) picks the quarter's metadata prefix, ``smart_attrs`` which
    SMART attributes exist, ``omit_columns`` deletes named columns from the header
    entirely (the "requested SMART column is absent" case), ``empty_attrs`` maps a model
    to the attributes it leaves as EMPTY STRINGS, and ``bad_capacity`` is the
    ``(serial, day offset)`` whose row gets ``capacity_bytes = -1`` (``None`` to write
    none; ``all_bad_capacity=True`` marks EVERY row, i.e. an archive nothing survives).
    Rows within a day are shuffled, because the shipped files are in no drive order the
    loader may rely on.
    """
    # ``bad_capacity`` accepts a single (serial, day_offset) pair OR a sequence of them,
    # so a test can punch a MULTI-DAY hole -- the case that distinguishes "the loader
    # dropped these rows" from "the drive left the fleet" in gap segmentation.
    if bad_capacity is None:
        _bad_rows = set()
    elif len(bad_capacity) and isinstance(bad_capacity[0], (tuple, list)):
        _bad_rows = {(serial, int(offset)) for serial, offset in bad_capacity}
    else:
        _bad_rows = {(bad_capacity[0], int(bad_capacity[1]))}

    rng = np.random.default_rng(seed)
    out_dir = Path(data_dir) / subdir if subdir else Path(data_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    smart_columns = synthetic_smart_columns(smart_attrs)
    header = [c for c in list(BACKBLAZE_META_LAYOUTS[meta_width]) + smart_columns
              if c not in set(omit_columns)]

    specs = drive_specs(models, n_days, n_failed, n_survivors)
    for spec in specs:                       # day offset -> position in the drive's life
        spec["index"] = {int(d): i for i, d in enumerate(spec["days"])}

    day0 = np.datetime64(start, "D")
    for offset in range(n_days):
        date = str(day0 + offset)
        rows = []
        for spec in specs:
            if offset not in spec["index"]:
                continue                     # the drive is not in the fleet that day
            index = spec["index"][offset]
            n_obs = len(spec["days"])
            to_end = n_obs - 1 - index
            model = spec["model"]
            blank = set(empty_attrs.get(model, ()))
            capacity = _CAPACITY.get(model, _DEFAULT_CAPACITY)
            if all_bad_capacity or (spec["serial"], offset) in _bad_rows:
                capacity = "-1"              # Backblaze: the whole row is unreliable
            values = {"date": date, "serial_number": spec["serial"], "model": model,
                      "capacity_bytes": capacity,
                      # failure == 1 marks the LAST day the drive was operational.
                      "failure": "1" if (spec["failed"] and to_end == 0) else "0"}
            values.update(_META_EXTRAS)
            for attr in smart_attrs:
                raw = _smart_raw(attr, index, to_end, spec["failed"])
                # An attribute the model does not populate is an EMPTY STRING in both
                # the normalized and the raw column -- never a zero.
                normalized = "" if attr in blank else str(100 - min(99, index // 10))
                values[f"smart_{attr}_normalized"] = normalized
                values[f"smart_{attr}_raw"] = "" if attr in blank else raw
            rows.append([values[column] for column in header])
        rng.shuffle(rows)
        text = "\r\n".join([",".join(header)] + [",".join(r) for r in rows]) + "\r\n"
        (out_dir / f"{date}.csv").write_text(("\ufeff" if bom else "") + text,
                                             encoding="utf-8", newline="")

    if junk:
        # The two artefacts the shipped archives carry: a __MACOSX shadow tree of
        # AppleDouble files and a .DS_Store. Neither may reach the parser.
        macos = out_dir.parent / "__MACOSX" / out_dir.name
        macos.mkdir(parents=True, exist_ok=True)
        (macos / f"._{str(day0)}.csv").write_text("Mac OS X resource fork\n")
        (macos / f"{str(day0)}.csv").write_text("not a Drive Stats file\n")
        (out_dir / ".DS_Store").write_text("junk\n")
    return out_dir
