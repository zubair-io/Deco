from pathlib import Path

from deco.config import load_config


def test_load_config_from_repo_default():
    cfg = load_config()
    assert cfg.gate.min_cosine == 0.55
    assert cfg.gate.min_inliers == 30
    assert cfg.retrieval.top_k == 50
    assert cfg.embedding.model_version.startswith("dinov2")


def test_load_config_explicit_path(tmp_path: Path):
    p = tmp_path / "deco.toml"
    p.write_text(
        "[gate]\nmin_cosine = 0.9\nmin_inliers = 5\n"
        "[retrieval]\ntop_k = 10\nrerank_top_n = 3\n"
        "[ingest]\nrate_limit_rps = 1.0\nmax_backoff_s = 10.0\n"
        '[embedding]\nmodel_version = "test-v0"\n'
        '[paths]\nimage_cache_dir = "/tmp/imgs"\n'
    )
    cfg = load_config(p)
    assert cfg.gate.min_cosine == 0.9
    assert cfg.paths.image_cache_dir == Path("/tmp/imgs")
