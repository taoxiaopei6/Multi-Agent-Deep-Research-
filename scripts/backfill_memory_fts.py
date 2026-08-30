"""
Backfill search_text / search_vector for memory_entries (Phase 2 FTS).

Must use the SAME jieba/tokenize logic as the write path (tokenize_for_bm25),
so backfilled rows match newly-written rows. Processes in batches to avoid
loading the whole table at once.

Usage:
    F:/agent/deep_research_2025/.venv/Scripts/python.exe scripts/backfill_memory_fts.py \
        --dsn postgresql://root:root123@127.0.0.1:5432/deep_research --batch 500

Rerunnable: only updates rows whose search_text is NULL or stale.
"""
import argparse
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))

import psycopg
from app.mult_agents.memory.long_term import content_to_text, tokenize_for_bm25

DEFAULT_DSN = "postgresql://root:root123@127.0.0.1:5432/deep_research"


def build_search_text(summary: str, content) -> str:
    """与写入路径完全一致的 search_text：jieba 分词 summary + content，空格连接。"""
    parts = []
    if summary:
        parts.append(summary)
    try:
        parts.append(content_to_text(content))
    except Exception:
        parts.append(str(content))
    combined = "\n".join(parts)
    tokens = tokenize_for_bm25(combined)
    return " ".join(tokens)


def backfill(dsn: str, batch: int, force: bool = False) -> int:
    updated = 0
    with psycopg.connect(dsn) as conn:
        with conn.cursor() as cur:
            while True:
                if force:
                    # 强制重跑：所有行（用于 migration 兜底用未分词文本填充后，改为 jieba 分词一致）
                    cur.execute("SELECT id, summary, content FROM memory_entries LIMIT %s", (batch,))
                else:
                    # 只处理 search_text 为空的行
                    cur.execute(
                        """
                        SELECT id, summary, content
                        FROM memory_entries
                        WHERE search_text IS NULL OR search_vector IS NULL
                        LIMIT %s
                        """,
                        (batch,),
                    )
                rows = cur.fetchall()
                if not rows:
                    break
                for mid, summary, content in rows:
                    search_text = build_search_text(summary or "", content)
                    if not search_text:
                        # 无可分词内容，置空串避免反复扫描
                        search_text = ""
                    cur.execute(
                        """
                        UPDATE memory_entries
                        SET search_text = %s,
                            search_vector = to_tsvector('simple', %s)
                        WHERE id = %s
                        """,
                        (search_text, search_text, mid),
                    )
                    updated += 1
                conn.commit()
                print(f"  backfilled {updated} rows so far...", flush=True)
    return updated


def main():
    parser = argparse.ArgumentParser(description="Backfill memory_entries FTS columns")
    parser.add_argument("--dsn", default=DEFAULT_DSN, help="PostgreSQL DSN")
    parser.add_argument("--batch", type=int, default=500, help="rows per batch")
    parser.add_argument("--force", action="store_true",
                        help="re-run all rows (ignore IS NULL guard), e.g. after migration fallback")
    args = parser.parse_args()

    print(f"backfilling memory_entries FTS columns (batch={args.batch}, force={args.force})...")
    n = backfill(args.dsn, args.batch, force=args.force)
    print(f"done: {n} rows updated")


if __name__ == "__main__":
    main()
