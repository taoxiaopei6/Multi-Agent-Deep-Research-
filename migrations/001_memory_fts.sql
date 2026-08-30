-- PostgreSQL FTS migration for long-term memory lexical recall (Phase 2).
-- Adds search_text (jieba-tokenized, space-joined) and a generated search_vector
-- (to_tsvector('simple', ...)) with a GIN index, so _search_postgres can switch
-- from ILIKE %query% to indexed full-text matching.

-- search_text: 由应用层用 jieba 分词后写入（summary + content），'simple' 配置把每个
-- 空格分隔 token 当作一个 lexeme，绕开 PG 内置中文分词短板。
ALTER TABLE memory_entries ADD COLUMN IF NOT EXISTS search_text TEXT;

-- search_vector: 由 search_text 生成，应用层负责在写入时填充。
ALTER TABLE memory_entries ADD COLUMN IF NOT EXISTS search_vector TSVECTOR;

-- GIN index 加速 @@ 查询。
CREATE INDEX IF NOT EXISTS idx_memory_entries_search_vector
ON memory_entries USING GIN (search_vector);

-- 兼容已有行：若 search_vector 为空但 content/summary 存在，用现成文本兜底生成
-- （注意：这只是兜底，正式 backfill 走 Python 同 jieba 逻辑，见 scripts/backfill_memory_fts.py）
UPDATE memory_entries
SET search_text = COALESCE(NULLIF(summary, ''), '') || ' ' || COALESCE(content::text, ''),
    search_vector = to_tsvector('simple',
        COALESCE(NULLIF(summary, ''), '') || ' ' || COALESCE(content::text, ''))
WHERE search_text IS NULL AND (summary IS NOT NULL OR content IS NOT NULL);
