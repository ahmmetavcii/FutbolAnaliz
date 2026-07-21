"""Regression tests for SoccerNet blocker remediation artifacts."""
from __future__ import annotations

import json
from pathlib import Path

import pytest

W = Path("/home/ahmet/workspace/soccernet_blocker_remediation")
A = W / "artifacts"
DOCS = Path(__file__).resolve().parents[2] / "docs/setup/soccernet_blocker_remediation"
STATUS = json.loads((DOCS / "status.json").read_text())
REPOS = {r["repository"]: r for r in STATUS["repositories"]}

PASS_INFER = {"ORIGINAL_FULL_PASS", "ORIGINAL_INFERENCE_PASS"}
PASS_SMOKE = {"ORIGINAL_SMOKE_PASS"}
PASS_DEVKIT = {"PASS_DEVKIT"}
VALID_ORIG = PASS_INFER | PASS_SMOKE | PASS_DEVKIT | {
    "SOURCE_ONLY_NO_MODEL", "N/A_README_ONLY", "BLOCKED_DEPENDENCY_CONFLICT",
    "BLOCKED_BUILD_NO_NVCC", "BLOCKED_RUNTIME_OOM", "BLOCKED_RUNTIME_ERROR",
    "CHECKPOINT_NOT_PUBLISHED", "CHECKPOINT_LINK_BROKEN",
    "EXTERNAL_BLOCKER_DATA_ACCESS", "EXTERNAL_BLOCKER_OFFICIAL_GT", "FAIL",
}
VALID_COMPAT = {
    "N/A", "COMPATIBLE_IMPLEMENTATION_PASS", "COMPATIBLE_REPLACEMENT_PASS",
    "COMPATIBLE_CLEANROOM_PASS", "EXISTING_COMPATIBLE_PASS", "NOT_IMPLEMENTED",
}


def test_ai_dev_torch_unchanged():
    before = (W / "system_before.txt").read_text()
    after = (W / "ai_dev_torch_after.txt").read_text().strip().splitlines()
    assert "torch 2.11.0+cu128" in before
    assert after[0] == "2.11.0+cu128"
    assert after[1] == "12.8"
    assert STATUS["ai_dev_torch_unchanged"] is True


@pytest.mark.parametrize("repo,env", [
    ("sn-banner", "sn-banner-runtime"),
    ("sn-caption", "sn-caption-eval"),
    ("sn-nvs", "sn-nvs-build"),
    ("sn-depth", "sn-depth-runtime"),
])
def test_env_isolation(repo, env):
    assert REPOS[repo]["env"] == env
    assert Path(f"/home/ahmet/miniconda3/envs/{env}").is_dir()
    freeze = A / repo / "pip_freeze.txt"
    assert freeze.is_file() and freeze.stat().st_size > 0


def test_banner_checkpoint_sha_and_real_inference():
    r = REPOS["sn-banner"]
    assert r["new_original"] == "ORIGINAL_INFERENCE_PASS"
    assert r["real_inference"] is True
    pred = json.loads((A / "sn-banner/predictions.json").read_text())
    assert pred["status"] == "ORIGINAL_INFERENCE_PASS"
    assert pred["checkpoint_sha256"] == "0133213b6b72273a03bce6961911386a61dbf04ad6de8cd747044f11f8b6c8e8"
    assert len(pred["frames"]) == 5
    for f, v in pred["frames"].items():
        assert v["finite"] is True
        assert Path(v["mask_png"]).is_file() and Path(v["mask_png"]).stat().st_size > 0
        assert Path(v["overlay_jpg"]).is_file()


def test_caption_evaluator_java_meteor_not_model():
    r = REPOS["sn-caption"]
    assert r["new_original"] == "PASS_DEVKIT"
    assert r["real_inference"] is False  # evaluator ≠ model
    m = json.loads((A / "sn-caption/metrics.json").read_text())
    assert m["status"] == "PASS_DEVKIT"
    assert m["checks"]["METEOR"]["direction_ok"] is True
    assert m["checks"]["Bleu_4"]["direction_ok"] is True
    java = (A / "sn-caption/java_info.txt").read_text()
    assert "openjdk" in java.lower()


def test_nvs_build_smoke_not_scene_inference():
    r = REPOS["sn-nvs"]
    assert r["new_original"] == "ORIGINAL_SMOKE_PASS"
    assert r["real_inference"] is False
    assert r["build_fixed"] is True
    f = json.loads((A / "sn-nvs/forward_test.json").read_text())
    assert f["status"] == "ORIGINAL_SMOKE_PASS"
    assert f["finite"] is True
    assert Path(A / "sn-nvs/synthetic_render.png").stat().st_size > 0


def test_gamestate_oom_fixed_original_and_compatible_separate():
    r = REPOS["sn-gamestate"]
    assert r["prev_original"] == "BLOCKED_RUNTIME_OOM"
    assert r["new_original"] == "ORIGINAL_INFERENCE_PASS"
    assert r["compatible"] == "COMPATIBLE_IMPLEMENTATION_PASS"
    p50 = json.loads((A / "sn-gamestate/original_tracklab_predictions_50f.json").read_text())
    assert p50["frames"] >= 20 and p50["detections"] > 0
    assert "ORIGINAL TrackLab" in p50["source"]
    assert r["peak_vram_gb"] and r["peak_vram_gb"] < 8


def test_pts_and_teamspotting_and_depth_real_inference():
    for repo in ("PTS-baseline", "sn-teamspotting", "sn-depth"):
        r = REPOS[repo]
        assert r["new_original"] == "ORIGINAL_INFERENCE_PASS"
        assert r["real_inference"] is True
        pred = json.loads((A / repo / "predictions.json").read_text())
        assert pred["status"] == "ORIGINAL_INFERENCE_PASS"
        assert "ORIGINAL" in pred["implementation"]


def test_checkpoint_manifest_shas():
    man = json.loads((DOCS / "checkpoint_manifest.json").read_text())
    assert len(man["checkpoints"]) >= 4
    for c in man["checkpoints"]:
        assert Path(c["file"]).is_file()
        assert len(c["sha256"]) == 64
        assert c["bytes"] > 0


@pytest.mark.parametrize("repo", list(REPOS))
def test_status_schema_and_evidence_files(repo):
    r = REPOS[repo]
    assert r["new_original"] in VALID_ORIG
    assert r["compatible"] in VALID_COMPAT
    d = A / repo
    assert (d / "test_results.json").is_file()
    assert (d / "artifact_manifest.json").is_file()
    man = json.loads((d / "artifact_manifest.json").read_text())
    assert man
    for name, meta in man.items():
        assert meta["bytes"] > 0


def test_no_synthetic_marked_as_real_inference():
    """NVS synthetic forward must stay ORIGINAL_SMOKE_PASS with real_inference=False."""
    assert REPOS["sn-nvs"]["real_inference"] is False
    assert REPOS["sn-nvs"]["new_original"] == "ORIGINAL_SMOKE_PASS"


def test_evaluator_not_promoted_to_model_pass():
    assert REPOS["sn-caption"]["new_original"] == "PASS_DEVKIT"
    assert REPOS["sn-caption"]["real_inference"] is False


def test_external_blockers_classified():
    assert "EXTERNAL_BLOCKER" in str(REPOS["sn-mvfoul"]["blocker"])
    assert "EXTERNAL_BLOCKER" in str(REPOS["sn-trackeval"]["blocker"])
    assert REPOS["sn-jersey"]["new_original"] == "N/A_README_ONLY"
    assert REPOS["sn-jersey"]["compatible"] == "COMPATIBLE_CLEANROOM_PASS"


def test_docs_exist():
    for name in ("status.json", "final_matrix.md", "full_remediation_report.md",
                 "unresolved_external_blockers.md", "environment_changes.md",
                 "checkpoint_manifest.json", "integration_decision.md"):
        assert (DOCS / name).is_file() and (DOCS / name).stat().st_size > 0
