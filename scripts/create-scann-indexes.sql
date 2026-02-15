-- ScaNN indexes for rag_chunk_embeddings
-- Run AFTER populating data (ScaNN cannot be created on empty tables in AlloyDB)
--
-- For CI/testing with pgvector (no ScaNN), use IVFFlat:
--   CREATE INDEX ix_rag_emb_vector ON rag_chunk_embeddings
--     USING ivfflat (embedding vector_cosine_ops) WITH (lists = 100);

-- ScaNN index (AlloyDB only)
CREATE INDEX IF NOT EXISTS ix_rag_emb_scann ON rag_chunk_embeddings
  USING scann (embedding cosine)
  WITH (num_leaves = 50);
