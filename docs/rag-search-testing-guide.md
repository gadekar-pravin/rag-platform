# RAG Search Testing Guide (VS Code Copilot)

Validate that the RAG search pipeline is working end-to-end by running test queries from VS Code Copilot agent mode. This guide uses the ingested **P&G Citizenship Report Analysis.pdf** (47 pages) as the test document.

## Prerequisites

1. VS Code 1.99+ with GitHub Copilot extension installed.
2. MCP server connected (see [VS Code MCP Setup](vscode-mcp-setup.md)).
3. At least one document ingested — the P&G Citizenship Report.

**Verify setup:** Open Copilot chat, switch to **Agent mode**, and click the tools icon. You should see `search` and `list_documents` from the `rag-search` server.

## Test 1: List Ingested Documents

Confirm the document was ingested and is visible.

**Prompt:**

> List all documents in the RAG knowledge base.

**Expected behavior:**
- Copilot calls the `list_documents` tool.
- The P&G Citizenship Report appears in the list with type `pdf` and a chunk count.

**Check:**
- [ ] Tool was called (you should see the tool call indicator in the chat).
- [ ] Document title matches: `P&G Citizenship Report Analysis.pdf`.
- [ ] Chunk count is non-zero (expect 30-60 chunks for a 47-page PDF).

## Test 2: Exact Topic Retrieval

These queries target specific, well-defined sections of the report. Each should return chunks that contain the expected keywords.

### 2a. Pampers Premature Babies Program

**Prompt:**

> Search the knowledge base for Pampers premature babies program

**Expected in results:** Content about Pampers Preemie Protection diapers, the `#PampersForPreemies` program, "13.4 million babies born too early", and kangaroo care.

**Check:**
- [ ] Returned chunks mention "Pampers", "premature" or "preemie".
- [ ] Content is from the Brand Programs / Community Impact section.

### 2b. Pakistan Flood Disaster Relief

**Prompt:**

> Search the knowledge base for Pakistan monsoon flood disaster relief

**Expected in results:** Content about 2022 monsoon flooding, "over 50 million residents impacted", "over 1,000 fatalities", and P&G's emergency relief response.

**Check:**
- [ ] Returned chunks mention "Pakistan" and "flood".
- [ ] Specific facts about the disaster response are present.

### 2c. Greenhouse Gas Emissions Targets

**Prompt:**

> Search the knowledge base for greenhouse gas emissions reduction science-based targets

**Expected in results:** Content about the Science Based Targets initiative (SBTi), Scope 1/2/3 emissions, and P&G being one of the first companies with a validated target in 2015.

**Check:**
- [ ] Returned chunks mention "greenhouse", "SBTi" or "science-based", and "Scope".
- [ ] Emissions data (e.g., "2,253 metric tons") may appear.

### 2d. Women in STEM India

**Prompt:**

> Search the knowledge base for women in STEM India gender equality

**Expected in results:** Content about "over 40% of graduates in STEM in India are women", P&G India's partnership with Save the Children, and specialized STEM labs for girls.

**Check:**
- [ ] Returned chunks mention "STEM", "India", and "women".
- [ ] The 40% statistic or Save the Children partnership appears.

### 2e. Water Conservation

**Prompt:**

> Search the knowledge base for water efficiency reuse billion liters

**Expected in results:** Content about P&G's 2030 water targets — "35% efficiency per unit of production", "5 billion liters recycled annually", and "3.47 billion liters reused in 2023".

**Check:**
- [ ] Returned chunks mention "water", "efficiency", and "billion".
- [ ] Specific numbers (3.47 billion liters, 1,388 Olympic swimming pools) may appear.

### 2f. Safeguard Handwashing

**Prompt:**

> Search the knowledge base for Safeguard handwashing Global Handwashing Day

**Expected in results:** Content about Safeguard being a co-founder of Global Handwashing Day, the `#SpreadHealthAcrossChina` program, and educating 100 million people.

**Check:**
- [ ] Returned chunks mention "Safeguard" and "handwashing".
- [ ] Brand program details are present, not just generic matches.

## Test 3: Semantic / Conceptual Queries

These queries don't use exact keywords from the document — they rely on vector (semantic) search to find relevant content.

### 3a. Corporate Social Responsibility Strategy

**Prompt:**

> Search the knowledge base for how the company approaches social responsibility and stakeholder value

**Expected behavior:** Should return content from the opening sections about P&G's integrated citizenship approach, "delivering for all stakeholders", and the PVP (Purpose, Values, Principles) framework.

**Check:**
- [ ] Results are topically relevant to corporate strategy / citizenship, not random sections.
- [ ] Vector-matched chunks (`[vector]` label) carry most of the relevant content.

### 3b. Diversity and Inclusion Initiatives

**Prompt:**

> Search the knowledge base for workplace diversity hiring neurodivergent employees

**Expected behavior:** Should return content about P&G's neurodiversity program, inclusive hiring practices for autism, the "50% women managers globally" milestone, and accessibility accommodations.

**Check:**
- [ ] Results touch on the Equality & Inclusion section.
- [ ] Chunks aren't just keyword matches — they reflect the conceptual query.

### 3c. Sustainable Packaging Goals

**Prompt:**

> What progress has been made toward making product packaging more environmentally friendly?

**Expected behavior:** Should return content about the 2030 goal ("100% recyclable or reusable packaging"), "78% achieved", the 50% virgin plastic reduction target, and Gillette's cardboard box transition.

**Check:**
- [ ] Results are from the Waste / Packaging section.
- [ ] Progress metrics (78%, 13% reduction) appear in the chunks.

## Test 4: Question-Answering (RAG + LLM Synthesis)

These prompts ask Copilot to synthesize an answer from the retrieved chunks, testing the full RAG pipeline.

### 4a. Factual Question

**Prompt:**

> Based on the knowledge base, how many disasters did P&G respond to in the 2022-2023 fiscal year?

**Expected answer:** "More than 30 disasters" — this exact figure is in the report.

**Check:**
- [ ] Copilot cites the "more than 30" figure.
- [ ] The answer references the fiscal year period (July 2022 - June 2023).

### 4b. Comparison Question

**Prompt:**

> According to the knowledge base, compare P&G's water and climate sustainability goals for 2030.

**Expected answer:** Should synthesize:
- **Climate:** Scope 1+2 emissions reduction (SBTi validated), Scope 3 supply chain intensity reduction by 40% per unit.
- **Water:** 35% efficiency improvement per unit (vs. 2010), 5 billion liters recycled annually.

**Check:**
- [ ] Copilot calls the search tool (possibly multiple times).
- [ ] The answer covers both water and climate, not just one.

### 4c. Summarization Question

**Prompt:**

> Based on the knowledge base, summarize P&G's key community impact programs.

**Expected answer:** Should mention several programs: Pampers for Preemies, Safeguard handwashing, Tide Loads of Hope, Always period poverty, disaster relief, and community partnerships.

**Check:**
- [ ] Multiple programs are listed (at least 3-4).
- [ ] The summary is grounded in actual document content, not hallucinated.

## Test 5: Edge Cases

### 5a. No-Match Query

**Prompt:**

> Search the knowledge base for quantum computing neural network architecture

**Expected behavior:** Results will still return (since there's only one document, it will be the top match), but the score should be low and the chunks won't be relevant to the query.

**Check:**
- [ ] Results are returned (the system doesn't error out).
- [ ] The chunks are clearly irrelevant to "quantum computing" — this is expected.

### 5b. Very Short Query

**Prompt:**

> Search the knowledge base for trees

**Expected behavior:** Should return chunks about Arbor Day Foundation, tree planting, or forestry/nature initiatives.

**Check:**
- [ ] The system handles single-word queries without errors.
- [ ] Results are at least loosely related to trees or nature.

### 5c. Non-English Query

**Prompt:**

> Search the knowledge base for betiyan

**Expected behavior:** "Betiyan" means "daughters" in Hindi. The vector search may find semantically adjacent content about women/girls empowerment programs in India, but the match will be weak.

**Check:**
- [ ] The system handles non-English queries without errors.
- [ ] Results may not be highly relevant — this is acceptable for non-English content in an English document.

## Understanding the Results

### Result Format

Each search result includes:

```
## 1. Document Title (type)
Score: 0.0328

**[vector]** chunk text from vector similarity match...

**[text]** chunk text from full-text search match...
```

### Score Interpretation

- **Score** is the Reciprocal Rank Fusion (RRF) score: `1/(K + rank_vector) + 1/(K + rank_text)` where K=60.
- With a single document, the score is always `2/(60+1) = 0.0328` (the document is rank 1 in both pools).
- With multiple documents, higher scores indicate stronger relevance. Scores typically range from 0.01 to 0.03.

### Chunk Types

- **`[vector]`** — retrieved by embedding similarity (semantic match). Good for conceptual queries.
- **`[text]`** — retrieved by PostgreSQL full-text search (keyword match). Good for exact terms and names.

Each result shows up to 2 vector chunks + 2 text chunks per document, deduplicated.

## Validation Checklist Summary

| # | Test | What it validates |
|---|---|---|
| 1 | List documents | Ingestion pipeline, document visibility, RLS |
| 2a-f | Exact topic queries | Chunk retrieval, keyword matching, vector relevance |
| 3a-c | Semantic queries | Vector embedding quality, conceptual search |
| 4a-c | Question answering | Full RAG pipeline (retrieve + synthesize) |
| 5a-c | Edge cases | Error handling, short queries, non-English input |

**Pass criteria:** Tests 1, 2a-f, and 3a-c should all return relevant content. Test 4 answers should be factually grounded. Test 5 should not produce errors.
