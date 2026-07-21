"""Automated verification tests for the 13 remaining SoccerNet repositories.

These tests validate the audit artifacts produced under
``/home/ahmet/workspace/soccernet_remaining_repo_tests`` and the master
``status.json`` under ``docs/setup/soccernet_remaining_repo_tests``. They assert:
- repository inventory + git metadata exist and are consistent
- source code is present and compiles
- artifact manifests are non-empty and parseable
- every PASS-class status is backed by real evidence artifacts
"""
from __future__ import annotations

import json
import math
from pathlib import Path

import pytest

BASE = Path("/home/ahmet/workspace/soccernet_remaining_repo_tests")
ART = BASE / "artifacts"
DOCS = Path(__file__).resolve().parents[2] / "docs/setup/soccernet_remaining_repo_tests"
STATUS = json.loads((DOCS / "status.json").read_text())

REPOS = [
    "sn-banner", "sn-nvs", "sn-teamspotting", "sn-depth", "sn-mvfoul",
    "sn-caption", "sn-spotting", "sn-reid", "sn-tracking", "ActiveSpotting",
    "PTS-baseline", "sn-grounding", "SoccerNet-v3",
]
PASS_CLASS = {"ORIGINAL_FULL_PASS", "ORIGINAL_INFERENCE_PASS", "ORIGINAL_SMOKE_PASS", "PASS_DEVKIT"}
VALID_ORIGINAL = PASS_CLASS | {
    "SOURCE_ONLY_NO_MODEL", "N/A_README_ONLY", "BLOCKED_MISSING_CHECKPOINT",
    "BLOCKED_CHECKPOINT_LINK_BROKEN", "BLOCKED_DATA_ACCESS", "BLOCKED_DEPENDENCY_CONFLICT",
    "BLOCKED_BUILD_NO_NVCC", "BLOCKED_RUNTIME_OOM", "BLOCKED_RUNTIME_ERROR", "FAIL",
}
VALID_COMPAT = {"N/A", "EXISTING_COMPATIBLE_PASS", "EXISTING_COMPATIBLE_PARTIAL", "NOT_IMPLEMENTED"}


def _rec(repo):
    return next(r for r in STATUS["repositories"] if r["repository"] == repo)


@pytest.mark.parametrize("repo", REPOS)
def test_repository_inventory_exists(repo):
    inv = ART / repo / "repository_inventory.json"
    assert inv.is_file() and inv.stat().st_size > 0
    data = json.loads(inv.read_text())
    assert data["exists"] and len(data["head_commit"]) == 40


@pytest.mark.parametrize("repo", REPOS)
def test_git_metadata_exists(repo):
    gm = ART / repo / "git_metadata.txt"
    assert gm.is_file() and "HEAD:" in gm.read_text()


@pytest.mark.parametrize("repo", REPOS)
def test_source_code_present(repo):
    inv = json.loads((ART / repo / "repository_inventory.json").read_text())
    assert inv["py_file_count"] + inv["notebook_count"] > 0
    assert inv["tracked_file_count"] > 1


@pytest.mark.parametrize("repo", REPOS)
def test_compile_pass(repo):
    comp = json.loads((ART / repo / "compile.json").read_text())
    assert comp["compile_status"] == "COMPILE_PASS"
    assert comp["py_compiled_ok"] == comp["py_total"] and comp["py_total"] > 0


@pytest.mark.parametrize("repo", REPOS)
def test_artifact_manifest(repo):
    man = ART / repo / "artifact_manifest.json"
    assert man.is_file()
    data = json.loads(man.read_text())
    assert data, "manifest empty"
    for name, meta in data.items():
        assert meta["bytes"] > 0 and len(meta["sha256"]) == 64


@pytest.mark.parametrize("repo", REPOS)
def test_status_schema(repo):
    r = _rec(repo)
    assert r["original_implementation_status"] in VALID_ORIGINAL
    assert r["compatible_implementation_status"] in VALID_COMPAT
    assert r["compile_status"] == "COMPILE_PASS"
    assert r["source_code_present"] is True


def test_status_counts_consistent():
    assert STATUS["total_repos"] == len(REPOS)
    assert STATUS["with_source_code"] == len(REPOS)
    assert STATUS["compile_pass_count"] == len(REPOS)


def test_devkit_evaluator_evidence():
    """PASS_DEVKIT repos must be backed by real evaluator perfect>imperfect evidence."""
    ev = json.loads((BASE / "phase6_evaluators.json").read_text())
    tv = json.loads((BASE / "phase6_tracking_v3.json").read_text())
    assert ev["action_spotting"]["status"] == "PASS_DEVKIT"
    assert ev["action_spotting"]["perfect_a_mAP"] > ev["action_spotting"]["imperfect_a_mAP"]
    assert ev["replay_grounding"]["status"] == "PASS_DEVKIT"
    assert ev["mvfoul"]["status"] == "PASS_DEVKIT"
    assert ev["reid"]["status"] == "PASS_DEVKIT"
    assert ev["reid"]["perfect"]["mAP"] > ev["reid"]["imperfect"]["mAP"]
    assert tv["sn-tracking"]["status"] == "PASS_DEVKIT"
    assert tv["sn-tracking"]["perfect"]["HOTA"] > tv["sn-tracking"]["idswitch"]["HOTA"]
    assert tv["sn-tracking"]["idswitch"]["IDSW"] >= 2


def test_activespotting_real_forward_evidence():
    """ORIGINAL_SMOKE_PASS must be backed by a finite real forward."""
    md = json.loads((BASE / "phase6_models.json").read_text())
    fwd = md["ActiveSpotting"]["forwards"]
    assert md["ActiveSpotting"]["status"] == "ORIGINAL_SMOKE_PASS"
    for name, f in fwd.items():
        assert f["finite"] is True
        assert f["out_shape"][0] == 2 and f["out_shape"][1] == 32768


def test_banner_checkpoint_downloaded_with_sha():
    r = _rec("sn-banner")
    ck = r["checkpoint"]
    assert ck["available_official"] is True
    assert len(ck["sha256"]) == 64 and ck["size_bytes"] > 10_000_000
    assert Path(ck["path"]).is_file()
    # documented as not loadable without OpenMMLab
    assert ck["loads"] is False


@pytest.mark.parametrize("repo,expected", [
    ("sn-nvs", "BLOCKED_BUILD_NO_NVCC"),
    ("sn-banner", "BLOCKED_DEPENDENCY_CONFLICT"),
    ("ActiveSpotting", "ORIGINAL_SMOKE_PASS"),
    ("sn-reid", "PASS_DEVKIT"),
])
def test_expected_statuses(repo, expected):
    assert _rec(repo)["original_implementation_status"] == expected


def test_no_pass_without_evidence():
    """Any PASS-class original status must have non-empty evidence text."""
    for r in STATUS["repositories"]:
        if r["original_implementation_status"] in PASS_CLASS:
            assert r["evidence"] and len(r["evidence"]) > 20
