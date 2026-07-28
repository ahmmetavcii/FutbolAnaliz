# Heatmap test failure audit

## Symptom

`tests/analytics/test_opta_analytics.py::TestHeatmapActivityQuality::test_heatmap_generation`
failed with:

```
assert (tmp_path / "heatmaps" / "player_7_position.png").is_file() → False
```

Full suite previously: 665 passed, 1 failed.

## Relation to Match-node-tracker work

**Unrelated.** Match-node adapters live under
`src/football_analytics/integrations/match_node_tracker/` and
`src/football_analytics/detection/`. They do not import or modify heatmap export.

Touched for this audit only:

- `src/football_analytics/opta/aggregate.py` (`export_heatmaps` filename)

## Root cause

`export_heatmaps` wrote team-tagged filenames:

```text
player_P{display_id}_{team}_position.png
```

e.g. `player_P7_unknown_position.png`

The unit test (and the cleanup glob comment intent) expect the stable contract:

```text
player_{display_id}_position.png
```

So the heatmap **was generated**, but under a renamed path. The assertion failed on the
canonical name. Deterministic: fails every solo run with the same mismatch.

## Fix (real contract restore — not a skip / not a loosened assert)

`export_heatmaps` now saves:

1. Canonical: `player_{id}_position.png` (what the test and stable consumers expect)
2. Optional browse alias: `player_P{id}_{team}_position.png`

Heatmap generation remains enabled.

## Verification

```bash
pytest tests/analytics/test_opta_analytics.py::TestHeatmapActivityQuality::test_heatmap_generation -q
# PASSED
```

## Notes

- Failure was reproducible and not flaky.
- No production pipeline config change required for this fix.
