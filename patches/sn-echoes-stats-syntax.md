# sn-echoes `stats.py` syntax patch

- Original file: `/home/ahmet/projects/soccernet/sn-echoes/stats.py`
- Reason: the final `print(...)` statement has an extraneous trailing `s`,
  causing an unconditional `SyntaxError`.
- Diff: `patches/sn-echoes-stats-syntax.patch`
- Apply:
  `git -C /home/ahmet/projects/soccernet/sn-echoes apply /home/ahmet/projects/football-analytics/patches/sn-echoes-stats-syntax.patch`
- Revert:
  `git -C /home/ahmet/projects/soccernet/sn-echoes apply -R /home/ahmet/projects/football-analytics/patches/sn-echoes-stats-syntax.patch`

This is a minimal one-character source correction. The pinned upstream commit
remains recorded separately in `external_repos.lock.yaml`, and the dirty state
must be reported while the patch is applied.
