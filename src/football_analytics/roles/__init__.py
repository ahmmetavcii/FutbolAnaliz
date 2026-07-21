"""Person-role classification: taxonomy, evidence scoring, temporal voting.

Design contracts:

- Roles: outfield_player, goalkeeper, referee, assistant_referee,
  fourth_official, substitute, staff, unknown_person.
- Officials are never included in team or player totals.
- The goalkeeper is a team member, and the role is sticky over time.
- Per-frame classifiers emit evidence; temporal voting makes decisions.
- Replay frames never contribute role evidence.
"""

from football_analytics.roles.active_player_state import (
    ActivePlayerRecord,
    ActivePlayerStateConfig,
    ActivePlayerStateTracker,
    ParticipationState,
)
from football_analytics.roles.goalkeeper_classifier import (
    GoalkeeperClassifier,
    GoalkeeperClassifierConfig,
    GoalkeeperFeatures,
)
from football_analytics.roles.referee_classifier import (
    OfficialScores,
    RefereeClassifier,
    RefereeClassifierConfig,
    RefereeFeatures,
)
from football_analytics.roles.role_classifier import (
    COUNTABLE_ROLES,
    OFFICIAL_ROLES,
    TEAM_ROLES,
    PersonFrameFeatures,
    PersonRole,
    RoleClassifier,
    RoleClassifierConfig,
    RoleObservation,
    counts_toward_team_totals,
    is_official,
    is_team_member,
)
from football_analytics.roles.role_voting import (
    RoleVote,
    RoleVoter,
    RoleVotingConfig,
    vote_roles,
)

__all__ = [
    "ActivePlayerRecord",
    "ActivePlayerStateConfig",
    "ActivePlayerStateTracker",
    "COUNTABLE_ROLES",
    "GoalkeeperClassifier",
    "GoalkeeperClassifierConfig",
    "GoalkeeperFeatures",
    "OFFICIAL_ROLES",
    "OfficialScores",
    "ParticipationState",
    "PersonFrameFeatures",
    "PersonRole",
    "RefereeClassifier",
    "RefereeClassifierConfig",
    "RefereeFeatures",
    "RoleClassifier",
    "RoleClassifierConfig",
    "RoleObservation",
    "RoleVote",
    "RoleVoter",
    "RoleVotingConfig",
    "TEAM_ROLES",
    "counts_toward_team_totals",
    "is_official",
    "is_team_member",
    "vote_roles",
]
