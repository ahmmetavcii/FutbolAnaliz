# Opta short-clip ground truth (schema 1.0.0)

Label a 15–30s clip manually. Unit-test green ≠ real accuracy.

## Files
- `gt_players.csv` — per-frame player boxes + team + stable gt_player_id
- `gt_ball.csv` — ball location when visible
- `gt_touches.csv` — contact times
- `gt_passes.csv` — pass start/end + success
- `gt_identity.csv` — real player matching notes

## Metrics produced by `evaluate_against_ground_truth`
- player ID precision/recall
- ID switches
- ball detection recall
- ball tracking coverage
- touch precision/recall
- pass precision/recall


## Unified frame events (`gt_frame_events.csv`)
Columns: frame_index, ball_x, ball_y, ball_visible, player_global_id, player_bbox_*, team_id, touch_event, pass_start, pass_end.

Do not compute accuracy metrics while this file is empty.
