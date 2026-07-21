from __future__ import annotations

import pytest

from football_analytics.multicamera import (
    AssignmentStatus,
    AuditEventKind,
    GlobalIdentityRegistry,
    PlayerRole,
    apply_manual_corrections,
    score_candidate,
)

EMBEDDING = tuple(float(i % 5) for i in range(16))


def test_strong_multi_cue_match_merges(make_obs):
    registry = GlobalIdentityRegistry()
    first = registry.assign(
        make_obs(
            camera_id="cam1",
            local_track_id=1,
            reference_time_seconds=0.0,
            pitch_xy_m=(30.0, 30.0),
            team_id=0,
            team_confidence=0.9,
            jersey_number=9,
            jersey_confidence=0.9,
            role=PlayerRole.OUTFIELD,
            reid_embedding=EMBEDDING,
        )
    )
    assert first.status is AssignmentStatus.NEW_IDENTITY

    second = registry.assign(
        make_obs(
            camera_id="cam2",
            local_track_id=7,
            reference_time_seconds=1.0,
            pitch_xy_m=(31.0, 30.0),
            team_id=0,
            team_confidence=0.9,
            jersey_number=9,
            jersey_confidence=0.9,
            role=PlayerRole.OUTFIELD,
            reid_embedding=EMBEDDING,
        )
    )
    assert second.status is AssignmentStatus.MATCHED
    assert second.global_id == first.global_id
    assert registry.global_id_for(("cam2", 7)) == first.global_id


def test_global_id_survives_shot_cut_and_local_id_churn(make_obs):
    registry = GlobalIdentityRegistry()
    cues = dict(
        team_id=0,
        team_confidence=0.9,
        jersey_number=9,
        jersey_confidence=0.9,
        role=PlayerRole.OUTFIELD,
        reid_embedding=EMBEDDING,
    )
    first = registry.assign(
        make_obs(
            camera_id="cam1",
            local_track_id=1,
            reference_time_seconds=0.0,
            pitch_xy_m=(30.0, 30.0),
            **cues,
        )
    )
    # After a shot cut the same camera restarts local track ids.
    after_cut = registry.assign(
        make_obs(
            camera_id="cam1",
            local_track_id=42,
            reference_time_seconds=4.0,
            pitch_xy_m=(33.0, 30.0),
            **cues,
        )
    )
    # A different camera picks the player up moments later.
    other_camera = registry.assign(
        make_obs(
            camera_id="cam3",
            local_track_id=5,
            reference_time_seconds=4.5,
            pitch_xy_m=(33.5, 30.2),
            **cues,
        )
    )
    assert after_cut.status is AssignmentStatus.MATCHED
    assert other_camera.status is AssignmentStatus.MATCHED
    assert after_cut.global_id == first.global_id
    assert other_camera.global_id == first.global_id
    assert registry.global_id_for(("cam1", 42)) == first.global_id
    assert registry.global_id_for(("cam3", 5)) == first.global_id


def test_jersey_alone_is_never_sufficient(make_obs):
    registry = GlobalIdentityRegistry()
    first = registry.assign(
        make_obs(
            camera_id="cam1",
            local_track_id=1,
            reference_time_seconds=0.0,
            pitch_xy_m=(10.0, 10.0),
            jersey_number=9,
            jersey_confidence=0.95,
        )
    )
    # Same jersey, no team / embedding / position evidence at all.
    second = registry.assign(
        make_obs(
            camera_id="cam2",
            local_track_id=2,
            reference_time_seconds=600.0,
            pitch_xy_m=None,
            jersey_number=9,
            jersey_confidence=0.95,
        )
    )
    assert second.status is not AssignmentStatus.MATCHED
    assert second.global_id != first.global_id


def test_impossible_simultaneous_distant_duplicate_rejected(make_obs):
    registry = GlobalIdentityRegistry()
    first = registry.assign(
        make_obs(
            camera_id="cam1",
            local_track_id=1,
            reference_time_seconds=10.0,
            pitch_xy_m=(0.0, 0.0),
            team_id=0,
            team_confidence=0.9,
            jersey_number=9,
            jersey_confidence=0.9,
            reid_embedding=EMBEDDING,
        )
    )
    # Identical evidence but 60 m away half a second later: unreachable.
    second = registry.assign(
        make_obs(
            camera_id="cam2",
            local_track_id=2,
            reference_time_seconds=10.5,
            pitch_xy_m=(60.0, 0.0),
            team_id=0,
            team_confidence=0.9,
            jersey_number=9,
            jersey_confidence=0.9,
            reid_embedding=EMBEDDING,
        )
    )
    assert second.status is AssignmentStatus.NEW_IDENTITY
    assert second.global_id != first.global_id

    snapshot = registry.identities[first.global_id].snapshot()
    scored = score_candidate(
        make_obs(
            camera_id="cam2",
            local_track_id=2,
            reference_time_seconds=10.5,
            pitch_xy_m=(60.0, 0.0),
            reid_embedding=EMBEDDING,
        ),
        snapshot,
    )
    assert scored.hard_reject
    assert "impossible position" in scored.reject_reason


def test_ambiguous_match_becomes_unresolved(make_obs):
    registry = GlobalIdentityRegistry()
    first = registry.assign(
        make_obs(
            camera_id="cam1",
            local_track_id=1,
            reference_time_seconds=0.0,
            pitch_xy_m=(10.0, 10.0),
            jersey_number=9,
            jersey_confidence=0.9,
        )
    )
    second = registry.assign(
        make_obs(
            camera_id="cam2",
            local_track_id=2,
            reference_time_seconds=600.0,
            jersey_number=9,
            jersey_confidence=0.9,
        )
    )
    assert second.status is AssignmentStatus.UNRESOLVED
    assert second.candidate_global_id == first.global_id
    unresolved = registry.unresolved_identities()
    assert [identity.global_id for identity in unresolved] == [second.global_id]
    kinds = [event.kind for event in registry.audit.events]
    assert AuditEventKind.UNRESOLVED in kinds


def test_merge_is_reversible_with_audit_history(make_obs):
    registry = GlobalIdentityRegistry()
    first = registry.assign(
        make_obs(
            camera_id="cam1",
            local_track_id=1,
            reference_time_seconds=0.0,
            pitch_xy_m=(10.0, 10.0),
            team_id=0,
            team_confidence=0.8,
        )
    )
    second = registry.assign(
        make_obs(
            camera_id="cam2",
            local_track_id=2,
            reference_time_seconds=300.0,
            pitch_xy_m=(90.0, 50.0),
            team_id=1,
            team_confidence=0.9,
        )
    )
    assert first.global_id != second.global_id

    sequence = registry.merge_identities(
        second.global_id, first.global_id, reason="review_confirmed"
    )
    assert second.global_id not in registry.identities
    merged = registry.identities[first.global_id]
    assert ("cam2", 2) in merged.track_keys
    assert registry.global_id_for(("cam2", 2)) == first.global_id

    restored = registry.undo_merge(sequence)
    assert restored == second.global_id
    assert registry.global_id_for(("cam2", 2)) == second.global_id
    assert registry.identities[first.global_id].track_keys == {("cam1", 1)}
    # The identity's team attribute is restored exactly.
    assert registry.identities[first.global_id].team_id == 0

    with pytest.raises(ValueError):
        registry.undo_merge(sequence)

    history_kinds = [event.kind for event in registry.history_for(first.global_id)]
    assert AuditEventKind.MERGE in history_kinds
    assert AuditEventKind.SPLIT in history_kinds


def test_split_moves_tracks_to_new_identity(make_obs):
    registry = GlobalIdentityRegistry()
    assignment = registry.assign(
        make_obs(camera_id="cam1", local_track_id=1, reference_time_seconds=0.0)
    )
    registry.assign(
        make_obs(camera_id="cam1", local_track_id=1, reference_time_seconds=1.0)
    )
    other = registry.assign(
        make_obs(
            camera_id="cam2",
            local_track_id=5,
            reference_time_seconds=100.0,
            pitch_xy_m=(50.0, 50.0),
        )
    )
    registry.merge_identities(other.global_id, assignment.global_id, reason="mistake")

    new_gid = registry.split_identity(
        assignment.global_id, [("cam2", 5)], reason="review_split"
    )
    assert registry.global_id_for(("cam2", 5)) == new_gid
    assert registry.identities[new_gid].unresolved
    assert ("cam2", 5) not in registry.identities[assignment.global_id].track_keys


def test_apply_manual_corrections(make_obs):
    registry = GlobalIdentityRegistry()
    a = registry.assign(
        make_obs(camera_id="cam1", local_track_id=1, reference_time_seconds=0.0)
    )
    b = registry.assign(
        make_obs(
            camera_id="cam2",
            local_track_id=2,
            reference_time_seconds=500.0,
            pitch_xy_m=(80.0, 60.0),
        )
    )
    apply_manual_corrections(
        registry, {"merges": [{"source": b.global_id, "target": a.global_id}]}
    )
    assert registry.global_id_for(("cam2", 2)) == a.global_id
    apply_manual_corrections(
        registry,
        {"splits": [{"global_id": a.global_id, "track_keys": [["cam2", 2]]}]},
    )
    assert registry.global_id_for(("cam2", 2)) != a.global_id
