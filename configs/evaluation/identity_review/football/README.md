# Identity Manual Review

```bash
cd /home/ahmet/projects/football-analytics
export PYTHONPATH=src
/home/ahmet/miniconda3/envs/ai-dev/bin/python -m streamlit run apps/identity_manual_review.py \
  --server.address 0.0.0.0 --server.port 8502
```

Open http://localhost:8502

Queue/media auto-prepare if missing. Decisions autosave to SQLite (WAL) + CSV + JSONL.
After 20/20, use **DOĞRULAMALARI UYGULA VE VİDEOYU YENİDEN ÜRET**.
