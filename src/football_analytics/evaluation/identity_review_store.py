"""SQLite-backed identity review decision store with atomic CSV snapshots."""

from __future__ import annotations

import csv
import json
import os
import sqlite3
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


DECISIONS = ("SAME", "DIFFERENT", "UNSURE")
ROLES = ("PLAYER", "GOALKEEPER", "REFEREE", "STAFF", "SPECTATOR", "UNRESOLVED")


def utc_now() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%S.%fZ")


class IdentityReviewStore:
    """WAL SQLite + CSV snapshot + append-only JSONL audit."""

    def __init__(self, root: Path) -> None:
        self.root = Path(root)
        self.root.mkdir(parents=True, exist_ok=True)
        self.db_path = self.root / "identity_review.db"
        self.csv_path = self.root / "identity_review_decisions.csv"
        self.jsonl_path = self.root / "identity_review_decisions.jsonl"
        self.backup_dir = self.root / "backups"
        self.backup_dir.mkdir(exist_ok=True)
        self._init_db()

    def _connect(self) -> sqlite3.Connection:
        conn = sqlite3.connect(str(self.db_path), timeout=30)
        conn.row_factory = sqlite3.Row
        conn.execute("PRAGMA journal_mode=WAL")
        conn.execute("PRAGMA synchronous=FULL")
        return conn

    def _init_db(self) -> None:
        with self._connect() as conn:
            conn.executescript(
                """
                CREATE TABLE IF NOT EXISTS decisions (
                    review_id TEXT PRIMARY KEY,
                    human_decision TEXT NOT NULL,
                    role_a_override TEXT,
                    role_b_override TEXT,
                    role_flag INTEGER NOT NULL DEFAULT 0,
                    updated_at TEXT NOT NULL,
                    created_at TEXT NOT NULL,
                    revision INTEGER NOT NULL DEFAULT 1
                );
                CREATE TABLE IF NOT EXISTS decision_revisions (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    review_id TEXT NOT NULL,
                    human_decision TEXT NOT NULL,
                    role_a_override TEXT,
                    role_b_override TEXT,
                    role_flag INTEGER NOT NULL DEFAULT 0,
                    revision INTEGER NOT NULL,
                    updated_at TEXT NOT NULL
                );
                CREATE INDEX IF NOT EXISTS idx_rev_review ON decision_revisions(review_id);
                """
            )
            conn.commit()

    def get_decision(self, review_id: str) -> dict[str, Any] | None:
        with self._connect() as conn:
            row = conn.execute(
                "SELECT * FROM decisions WHERE review_id=?", (review_id,)
            ).fetchone()
            return dict(row) if row else None

    def all_decisions(self) -> dict[str, dict[str, Any]]:
        with self._connect() as conn:
            rows = conn.execute("SELECT * FROM decisions").fetchall()
            return {str(r["review_id"]): dict(r) for r in rows}

    def completed_count(self) -> int:
        with self._connect() as conn:
            return int(conn.execute("SELECT COUNT(*) FROM decisions").fetchone()[0])

    def revision_history(self, review_id: str) -> list[dict[str, Any]]:
        with self._connect() as conn:
            rows = conn.execute(
                "SELECT * FROM decision_revisions WHERE review_id=? ORDER BY revision",
                (review_id,),
            ).fetchall()
            return [dict(r) for r in rows]

    def _backup_before_write(self) -> None:
        stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S")
        if self.db_path.exists():
            import shutil

            shutil.copy2(self.db_path, self.backup_dir / f"identity_review_{stamp}.db")
        if self.csv_path.exists():
            import shutil

            shutil.copy2(self.csv_path, self.backup_dir / f"identity_review_decisions_{stamp}.csv")

    def save_decision(
        self,
        review_id: str,
        human_decision: str,
        *,
        role_flag: bool = False,
        role_a_override: str | None = None,
        role_b_override: str | None = None,
    ) -> dict[str, Any]:
        if human_decision not in DECISIONS:
            raise ValueError(f"invalid decision: {human_decision}")
        if role_a_override is not None and role_a_override not in ROLES:
            raise ValueError(f"invalid role_a: {role_a_override}")
        if role_b_override is not None and role_b_override not in ROLES:
            raise ValueError(f"invalid role_b: {role_b_override}")

        self._backup_before_write()
        now = utc_now()
        with self._connect() as conn:
            existing = conn.execute(
                "SELECT * FROM decisions WHERE review_id=?", (review_id,)
            ).fetchone()
            if existing is None:
                rev = 1
                conn.execute(
                    """
                    INSERT INTO decisions(
                        review_id, human_decision, role_a_override, role_b_override,
                        role_flag, updated_at, created_at, revision
                    ) VALUES (?,?,?,?,?,?,?,?)
                    """,
                    (
                        review_id,
                        human_decision,
                        role_a_override,
                        role_b_override,
                        int(role_flag),
                        now,
                        now,
                        rev,
                    ),
                )
            else:
                rev = int(existing["revision"]) + 1
                conn.execute(
                    """
                    UPDATE decisions SET human_decision=?, role_a_override=?, role_b_override=?,
                        role_flag=?, updated_at=?, revision=?
                    WHERE review_id=?
                    """,
                    (
                        human_decision,
                        role_a_override,
                        role_b_override,
                        int(role_flag),
                        now,
                        rev,
                        review_id,
                    ),
                )
            conn.execute(
                """
                INSERT INTO decision_revisions(
                    review_id, human_decision, role_a_override, role_b_override,
                    role_flag, revision, updated_at
                ) VALUES (?,?,?,?,?,?,?)
                """,
                (
                    review_id,
                    human_decision,
                    role_a_override,
                    role_b_override,
                    int(role_flag),
                    rev,
                    now,
                ),
            )
            conn.commit()

            # verify from disk
            verified = conn.execute(
                "SELECT * FROM decisions WHERE review_id=?", (review_id,)
            ).fetchone()
            if verified is None or verified["human_decision"] != human_decision:
                raise RuntimeError("save verification failed")

        payload = dict(verified)
        self._append_jsonl(payload)
        self._write_csv_snapshot()
        # re-verify csv contains row
        if not self._csv_contains(review_id, human_decision):
            raise RuntimeError("csv snapshot verification failed")
        return payload

    def _append_jsonl(self, row: dict[str, Any]) -> None:
        line = json.dumps(dict(row), ensure_ascii=False) + "\n"
        with open(self.jsonl_path, "a", encoding="utf-8") as f:
            f.write(line)
            f.flush()
            os.fsync(f.fileno())

    def _write_csv_snapshot(self) -> None:
        rows = list(self.all_decisions().values())
        fields = [
            "review_id",
            "human_decision",
            "role_a_override",
            "role_b_override",
            "role_flag",
            "updated_at",
            "created_at",
            "revision",
        ]
        tmp = self.csv_path.with_suffix(".csv.tmp")
        with open(tmp, "w", encoding="utf-8", newline="") as f:
            w = csv.DictWriter(f, fieldnames=fields)
            w.writeheader()
            for r in sorted(rows, key=lambda x: x["review_id"]):
                w.writerow({k: r.get(k) for k in fields})
            f.flush()
            os.fsync(f.fileno())
        os.replace(tmp, self.csv_path)

    def _csv_contains(self, review_id: str, decision: str) -> bool:
        if not self.csv_path.exists():
            return False
        with open(self.csv_path, encoding="utf-8") as f:
            for row in csv.DictReader(f):
                if row.get("review_id") == review_id and row.get("human_decision") == decision:
                    return True
        return False
