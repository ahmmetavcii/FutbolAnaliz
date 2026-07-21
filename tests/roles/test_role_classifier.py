from __future__ import annotations

import pytest

from football_analytics.roles import (
    COUNTABLE_ROLES,
    OFFICIAL_ROLES,
    PersonFrameFeatures,
    PersonRole,
    RoleClassifier,
    counts_toward_team_totals,
    is_official,
    is_team_member,
)


class TestTaxonomy:
    def test_all_required_roles_exist(self):
        values = {role.value for role in PersonRole}
        assert values == {
            "outfield_player",
            "goalkeeper",
            "referee",
            "assistant_referee",
            "fourth_official",
            "substitute",
            "staff",
            "unknown_person",
        }

    def test_officials_are_never_team_members(self):
        for role in OFFICIAL_ROLES:
            assert is_official(role)
            assert not is_team_member(role)
            assert not counts_toward_team_totals(role)

    def test_goalkeeper_is_team_member_and_countable(self):
        assert is_team_member(PersonRole.GOALKEEPER)
        assert counts_toward_team_totals(PersonRole.GOALKEEPER)
        assert PersonRole.GOALKEEPER in COUNTABLE_ROLES

    def test_substitute_is_team_member_but_not_countable(self):
        assert is_team_member(PersonRole.SUBSTITUTE)
        assert not counts_toward_team_totals(PersonRole.SUBSTITUTE)

    def test_staff_and_unknown_never_count(self):
        assert not counts_toward_team_totals(PersonRole.STAFF)
        assert not counts_toward_team_totals(PersonRole.UNKNOWN_PERSON)


class TestRoleClassifier:
    @pytest.fixture
    def classifier(self) -> RoleClassifier:
        return RoleClassifier()

    def test_official_kit_on_pitch_scores_referee(self, classifier):
        observation = classifier.classify_frame(
            PersonFrameFeatures(
                track_id=1,
                timestamp_ms=0.0,
                on_pitch=True,
                kit_similarity_officials=0.9,
                kit_similarity_team0=0.05,
                kit_similarity_team1=0.05,
            )
        )
        assert observation.best_role is PersonRole.REFEREE

    def test_official_kit_on_touchline_scores_assistant(self, classifier):
        observation = classifier.classify_frame(
            PersonFrameFeatures(
                track_id=2,
                timestamp_ms=0.0,
                on_pitch=False,
                near_touchline=True,
                kit_similarity_officials=0.9,
            )
        )
        assert observation.best_role is PersonRole.ASSISTANT_REFEREE

    def test_official_kit_in_technical_area_scores_fourth_official(self, classifier):
        observation = classifier.classify_frame(
            PersonFrameFeatures(
                track_id=3,
                timestamp_ms=0.0,
                on_pitch=False,
                in_technical_area=True,
                kit_similarity_officials=0.9,
            )
        )
        assert observation.best_role is PersonRole.FOURTH_OFFICIAL

    def test_team_kit_on_pitch_scores_outfield_player(self, classifier):
        observation = classifier.classify_frame(
            PersonFrameFeatures(
                track_id=4,
                timestamp_ms=0.0,
                on_pitch=True,
                kit_similarity_team0=0.9,
                kit_similarity_officials=0.05,
            )
        )
        assert observation.best_role is PersonRole.OUTFIELD_PLAYER

    def test_team_kit_in_bench_area_scores_substitute(self, classifier):
        observation = classifier.classify_frame(
            PersonFrameFeatures(
                track_id=5,
                timestamp_ms=0.0,
                on_pitch=False,
                in_bench_area=True,
                kit_similarity_team1=0.85,
            )
        )
        assert observation.best_role is PersonRole.SUBSTITUTE

    def test_goalkeeper_evidence_beats_outfield(self, classifier):
        observation = classifier.classify_frame(
            PersonFrameFeatures(
                track_id=6,
                timestamp_ms=0.0,
                on_pitch=True,
                kit_similarity_team0=0.55,
                goalkeeper_kit_distinctiveness=0.95,
                own_penalty_area_occupancy=0.95,
                is_deepest_teammate=True,
                team_id=0,
            )
        )
        assert observation.best_role is PersonRole.GOALKEEPER

    def test_no_evidence_yields_unknown(self, classifier):
        observation = classifier.classify_frame(
            PersonFrameFeatures(track_id=7, timestamp_ms=0.0)
        )
        assert observation.best_role is PersonRole.UNKNOWN_PERSON
        assert observation.best_score == 0.0

    def test_weak_conflicting_evidence_yields_unknown(self, classifier):
        observation = classifier.classify_frame(
            PersonFrameFeatures(
                track_id=8,
                timestamp_ms=0.0,
                kit_similarity_officials=0.36,
                kit_similarity_team0=0.4,
            )
        )
        assert observation.best_role is PersonRole.UNKNOWN_PERSON

    def test_replay_flag_propagates(self, classifier):
        observation = classifier.classify_frame(
            PersonFrameFeatures(
                track_id=9,
                timestamp_ms=0.0,
                replay=True,
                kit_similarity_team0=0.9,
            )
        )
        assert observation.replay is True
