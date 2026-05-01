-- =============================================================================
-- migration_consistency.sql
-- Run once to prepare the database for the ConsistencyAgent.
-- =============================================================================

-- Table that stores the Speech Fidelity Index (IFD) per deputy
CREATE TABLE IF NOT EXISTS ifd_deputados (
    deputado_id   INTEGER PRIMARY KEY REFERENCES deputados(id),
    ifd           FLOAT,                   -- 0.0 (contradicts) to 1.0 (consistent)
    consistentes  INTEGER   DEFAULT 0,     -- number of consistent speech-vote pairs
    total_validos INTEGER   DEFAULT 0,     -- total valid speech-vote pairs analyzed
    alertas       JSONB     DEFAULT '[]',  -- list of high-confidence contradictions
    atualizado_em TIMESTAMP
);

-- Index for fast ranking queries (lowest IFD first)
CREATE INDEX IF NOT EXISTS idx_ifd_score ON ifd_deputados (ifd);

-- =============================================================================
-- Required columns in the discursos table.
-- Add them if they do not exist yet (populated by the BERTopic pipeline).
-- =============================================================================

ALTER TABLE discursos ADD COLUMN IF NOT EXISTS category_final VARCHAR(100);
ALTER TABLE discursos ADD COLUMN IF NOT EXISTS topic          INTEGER;
