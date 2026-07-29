"""The FD001 stable-key guard (repo invariant §1.2).

Cache keys are pure functions of `Config`. Every recorded FD001 artifact on Drive was
written under the keys pinned below, so a key change silently orphans months of cached
embeddings and invalidates the recorded §12 winner. New key fields must therefore be
added CONDITIONALLY (only when non-default), which is exactly what the second half of
this module asserts. If a schema genuinely must change, bump `CACHE_SCHEMA_VERSION` and
update the expectations here in the same commit, with a `CHANGES.md` note.
"""

from __future__ import annotations

from src.config import (CACHE_SCHEMA_VERSION, Config, FD001_NONCONSTANT_SENSORS)

# The recorded FD001 keys (CHANGES.md §40 records the window key verbatim).
FD001_WINDOW_KEY = "windows_FD001_1da313c871251cec"
FD001_EMBEDDING_KEY = "emb_FD001_chronos-2_forecast_token_w30_c30_v2_c778b501f4cc8f6e"


def test_fd001_default_keys_are_byte_identical_to_the_recorded_ones():
    cfg = Config(dataset="FD001")
    assert cfg.window_cache_key() == FD001_WINDOW_KEY
    assert cfg.embedding_cache_key() == FD001_EMBEDDING_KEY
    assert CACHE_SCHEMA_VERSION == 2
    # the resolved sensor default is the recorded explicit list (no silent drift)
    assert cfg.sensor_columns == list(FD001_NONCONSTANT_SENSORS)


def test_every_v2_key_field_is_conditional_on_being_non_default():
    """The v2 build added `channel_aggregation` (RQ-M) and `noise_injection` +
    `noise_seed` (RQ-H) to the keys. At their defaults they must be ABSENT from the key
    dicts, so an FD001 key computed today equals one computed before they existed."""
    cfg = Config(dataset="FD001")
    win, emb = cfg._window_key_fields(), cfg._embedding_key_fields()
    for absent in ("channel_aggregation", "noise_injection", "noise_seed", "seed"):
        assert absent not in win and absent not in emb
    # ... and present the moment they are set (so a real change re-keys)
    noisy = cfg.replace(noise_injection={"kind": "gaussian", "snr_db": 20})
    assert {"noise_injection", "noise_seed"} <= set(noisy._window_key_fields())
    assert "channel_aggregation" in cfg.replace(
        channel_aggregation="mean")._embedding_key_fields()


def test_dataset_scoped_key_fields_never_leak_into_fd001():
    """XJTU-SY / N-CMAPSS split-protocol knobs are keyed only for their own family, so
    changing one can never re-key an FD001 cache."""
    cfg = Config(dataset="FD001")
    for field, value in (("xjtu_test_truncation", 0.5), ("ncmapss_test_truncation", 0.5),
                         ("dsall_datasets", ["DS01", "DS02"])):
        assert cfg.replace(**{field: value}).window_cache_key() == FD001_WINDOW_KEY
    # paths and the experiment namespace are never in a key (§23)
    assert cfg.replace(cache_dir="/somewhere/else", results_dir="/elsewhere",
                       data_root="/another/root", data_dir="/explicit",
                       experiment_name="run-1").embedding_cache_key() == FD001_EMBEDDING_KEY
    # ... nor is any Stage-B-only head knob (§9)
    assert cfg.replace(head_features="emb+locscale+raw",
                       head_hidden_dim=512).embedding_cache_key() == FD001_EMBEDDING_KEY
