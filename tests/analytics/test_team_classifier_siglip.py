"""Tests for sports-main style team centroid resolver."""

import numpy as np

from football_analytics.analytics.team_classifier_siglip import resolve_by_team_centroids


def test_resolve_by_team_centroids_assigns_nearest():
    players = np.array([[10.0, 10.0], [12.0, 11.0], [200.0, 10.0], [202.0, 12.0]])
    teams = np.array([0, 0, 1, 1])
    queries = np.array([[15.0, 10.0], [198.0, 11.0]])
    out = resolve_by_team_centroids(players, teams, queries)
    assert list(out) == [0, 1]
