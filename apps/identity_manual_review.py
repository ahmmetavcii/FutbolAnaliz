#!/usr/bin/env python3
"""Browser identity review UI — big buttons, SQLite autosave, auto-advance."""
from __future__ import annotations

import subprocess
import sys
from pathlib import Path

import pandas as pd
import streamlit as st

ROOT = Path("/home/ahmet/projects/football-analytics")
sys.path.insert(0, str(ROOT / "src"))

from football_analytics.evaluation.identity_review_store import (  # noqa: E402
    DECISIONS,
    ROLES,
    IdentityReviewStore,
)

REVIEW_ROOT = ROOT / "configs/evaluation/identity_review/football"
QUEUE_PATH = REVIEW_ROOT / "review_queue.parquet"
PREPARE = ROOT / "scripts/prepare_identity_review_queue.py"
APPLY = ROOT / "scripts/apply_identity_review_decisions.py"
PYTHON = "/home/ahmet/miniconda3/envs/ai-dev/bin/python"


def ensure_queue() -> pd.DataFrame:
    REVIEW_ROOT.mkdir(parents=True, exist_ok=True)
    need = not QUEUE_PATH.exists()
    if not need:
        q = pd.read_parquet(QUEUE_PATH)
        # check media
        if len(q) < 20:
            need = True
        else:
            sample = q.iloc[0]
            if not Path(sample.clip_a_path).exists() or not Path(sample.clip_b_path).exists():
                need = True
    if need:
        st.info("İnceleme kuyruğu ve videolar hazırlanıyor… (birkaç dakika sürebilir)")
        proc = subprocess.run(
            [PYTHON, str(PREPARE)],
            cwd=str(ROOT),
            capture_output=True,
            text=True,
            env={**dict(**{k: v for k, v in __import__("os").environ.items()}), "PYTHONPATH": str(ROOT / "src")},
            timeout=1800,
        )
        if proc.returncode != 0:
            st.error("Kuyruk hazırlama başarısız")
            st.code(proc.stdout[-2000:] + "\n" + proc.stderr[-2000:])
            st.stop()
    return pd.read_parquet(QUEUE_PATH).sort_values("review_id").reset_index(drop=True)


def first_unanswered(queue: pd.DataFrame, store: IdentityReviewStore) -> int:
    done = store.all_decisions()
    for i, r in queue.iterrows():
        if str(r.review_id) not in done:
            return int(i)
    return max(0, len(queue) - 1)


def main() -> None:
    st.set_page_config(page_title="Kimlik Doğrulama", layout="wide", initial_sidebar_state="collapsed")
    st.markdown(
        """
        <style>
        .stButton>button { height: 3.2rem; font-size: 1.15rem; font-weight: 700; }
        div[data-testid="stHorizontalBlock"] button { min-height: 3.4rem; }
        </style>
        """,
        unsafe_allow_html=True,
    )

    store = IdentityReviewStore(REVIEW_ROOT)
    queue = ensure_queue()
    n = len(queue)
    done_map = store.all_decisions()
    completed = len(done_map)

    if "idx" not in st.session_state:
        st.session_state.idx = first_unanswered(queue, store)
    if "flash" not in st.session_state:
        st.session_state.flash = None
    if "flash_err" not in st.session_state:
        st.session_state.flash_err = None
    if "show_context" not in st.session_state:
        st.session_state.show_context = False
    if "show_role" not in st.session_state:
        st.session_state.show_role = False
    if "debug" not in st.session_state:
        st.session_state.debug = False

    idx = int(st.session_state.idx)
    idx = max(0, min(idx, n - 1))
    st.session_state.idx = idx
    item = queue.iloc[idx]
    rid = str(item.review_id)

    st.title("Kimlik Doğrulama")
    st.subheader(f"Örnek {idx + 1} / {n}")
    c1, c2, c3 = st.columns(3)
    c1.metric("Tamamlanan", f"{completed}")
    c2.metric("Kalan", f"{max(0, n - completed)}")
    c3.metric("Bu örnek", "✓ kayıtlı" if rid in done_map else "bekliyor")

    if st.session_state.flash:
        st.success(st.session_state.flash)
        st.session_state.flash = None
    if st.session_state.flash_err:
        st.error(st.session_state.flash_err)
        st.session_state.flash_err = None

    # Completion screen
    if completed >= n and all(str(r.review_id) in done_map for _, r in queue.iterrows()):
        st.balloons()
        st.header("MANUEL DOĞRULAMA TAMAMLANDI")
        st.write(f"**{n}/{n}** karar kaydedildi")
        counts = {"SAME": 0, "DIFFERENT": 0, "UNSURE": 0}
        role_n = 0
        for d in done_map.values():
            counts[d["human_decision"]] = counts.get(d["human_decision"], 0) + 1
            if int(d.get("role_flag") or 0):
                role_n += 1
        st.write(
            f"- Same player: **{counts.get('SAME', 0)}**\n"
            f"- Different players: **{counts.get('DIFFERENT', 0)}**\n"
            f"- Unsure: **{counts.get('UNSURE', 0)}**\n"
            f"- Role corrections: **{role_n}**"
        )
        if st.button("DOĞRULAMALARI UYGULA VE VİDEOYU YENİDEN ÜRET", type="primary", use_container_width=True):
            prog = st.progress(0, text="Kararlar okunuyor…")
            log_box = st.empty()
            proc = subprocess.Popen(
                [PYTHON, str(APPLY)],
                cwd=str(ROOT),
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True,
                env={**dict(__import__("os").environ), "PYTHONPATH": str(ROOT / "src")},
            )
            lines = []
            steps = [
                "Kararlar okunuyor",
                "Must-link",
                "Cannot-link",
                "association",
                "Veto",
                "Video render",
                "2D",
                "Test",
            ]
            step_i = 0
            assert proc.stdout is not None
            for line in proc.stdout:
                lines.append(line.rstrip())
                low = line.lower()
                for s in steps:
                    if s.lower().split()[0] in low or s.lower() in low:
                        step_i = max(step_i, steps.index(s))
                prog.progress(min(0.95, (step_i + 1) / len(steps)), text=line.strip()[:80] or "çalışıyor…")
                log_box.code("\n".join(lines[-30:]))
            rc = proc.wait()
            prog.progress(1.0, text="Tamam")
            if rc == 0:
                st.success("Human-verified sonuçlar üretildi: /mnt/c/football_data/results/tracking_human_verified/")
            else:
                st.error("Uygulama scripti hata verdi")
                st.code("\n".join(lines[-80:]))
        st.divider()

    st.markdown("### BU İKİSİ AYNI OYUNCU MU?")
    left, right = st.columns(2)
    with left:
        st.caption("Oyuncu A")
        if Path(item.clip_a_path).exists():
            st.video(str(item.clip_a_path))
        else:
            st.warning("A videosu yok")
        if Path(getattr(item, "summary_jpg_path", "") or "").exists() is False and Path(
            str(REVIEW_ROOT / "media" / f"{rid}_summary.jpg")
        ).exists():
            pass
    with right:
        st.caption("Oyuncu B")
        if Path(item.clip_b_path).exists():
            st.video(str(item.clip_b_path))
        else:
            st.warning("B videosu yok")

    sum_path = REVIEW_ROOT / "media" / f"{rid}_summary.jpg"
    if "summary_jpg_path" in item.index and pd.notna(item.summary_jpg_path):
        sum_path = Path(str(item.summary_jpg_path))
    if sum_path.exists():
        st.image(str(sum_path), caption="Özet kare (A | B)", use_container_width=True)

    if st.button("Tam saha görüntüsünü göster / gizle"):
        st.session_state.show_context = not st.session_state.show_context
    if st.session_state.show_context and Path(item.full_context_clip_path).exists():
        st.video(str(item.full_context_clip_path))

    # Role override optional
    if st.checkbox("Bu kişilerden birinin rolü yanlış", value=st.session_state.show_role, key="role_flag_box"):
        st.session_state.show_role = True
        rc1, rc2 = st.columns(2)
        with rc1:
            role_a = st.selectbox("Oyuncu A rolü", ROLES, index=0, key=f"role_a_{rid}")
        with rc2:
            role_b = st.selectbox("Oyuncu B rolü", ROLES, index=0, key=f"role_b_{rid}")
    else:
        st.session_state.show_role = False
        role_a = role_b = None

    # Hide model decision before answer
    if st.session_state.debug and rid in done_map:
        st.caption(
            f"(debug) model={item.model_decision} conf={item.model_confidence:.2f} type={item.review_type}"
        )

    def commit(decision: str) -> None:
        try:
            store.save_decision(
                rid,
                decision,
                role_flag=bool(st.session_state.show_role),
                role_a_override=role_a if st.session_state.show_role else None,
                role_b_override=role_b if st.session_state.show_role else None,
            )
            # verify again
            v = store.get_decision(rid)
            if not v or v["human_decision"] != decision:
                st.session_state.flash_err = "KAYIT BAŞARISIZ — karar değişmedi."
                return
            st.session_state.flash = f"KAYDEDİLDİ ✓ — Örnek {idx + 1}/{n}"
            # advance to next unanswered
            done2 = store.all_decisions()
            nxt = None
            for j in range(idx + 1, n):
                if str(queue.iloc[j].review_id) not in done2:
                    nxt = j
                    break
            if nxt is None:
                for j in range(0, n):
                    if str(queue.iloc[j].review_id) not in done2:
                        nxt = j
                        break
            st.session_state.idx = nxt if nxt is not None else idx
            st.rerun()
        except Exception as exc:  # noqa: BLE001
            st.session_state.flash_err = f"KAYIT BAŞARISIZ — karar değişmedi. ({exc})"
            st.rerun()

    b1, b2, b3 = st.columns(3)
    with b1:
        if st.button("✅ AYNI OYUNCU", type="primary", use_container_width=True):
            commit("SAME")
    with b2:
        if st.button("❌ FARKLI OYUNCULAR", use_container_width=True):
            commit("DIFFERENT")
    with b3:
        if st.button("❓ EMİN DEĞİLİM", use_container_width=True):
            commit("UNSURE")

    n1, n2, n3, n4 = st.columns(4)
    with n1:
        if st.button("← Önceki", use_container_width=True) and idx > 0:
            st.session_state.idx = idx - 1
            st.rerun()
    with n2:
        if st.button("Sonraki →", use_container_width=True) and idx < n - 1:
            st.session_state.idx = idx + 1
            st.rerun()
    with n3:
        if st.button("Kararı değiştir", use_container_width=True):
            st.info("Yeni bir karar butonuna basarak bu örneği güncelleyin. Completion artmaz.")
    with n4:
        if st.button("Tekrar oynat", use_container_width=True):
            st.rerun()

    with st.expander("Gelişmiş"):
        st.session_state.debug = st.checkbox("Debug teknik bilgi (yalnız cevap sonrası)", value=st.session_state.debug)
        if rid in done_map:
            st.json(done_map[rid])
            st.write("Revision history")
            st.dataframe(pd.DataFrame(store.revision_history(rid)))


if __name__ == "__main__":
    main()
