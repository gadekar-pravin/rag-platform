# VS Code MCP Setup Guide

Connect your VS Code agent (GitHub Copilot) to the team's RAG knowledge base. Once set up, you can search documents and browse the knowledge base directly from your editor using natural language.

## Prerequisites

- **VS Code** 1.99 or later
- **GitHub Copilot extension** (with agent mode / chat enabled)
- **No token required (current state):** `rag-mcp` is currently public (`allUsers`) and accepts unauthenticated MCP calls.
- **Optional, only if IAM auth is enabled later:** GCP access to `apexflow-ai` + `gcloud` CLI for OIDC token generation.

## Step 1: Add the MCP Configuration (Current Public Setup)

Create (or update) `.vscode/mcp.json` in your workspace root:

```json
{
  "servers": {
    "rag-search": {
      "type": "http",
      "url": "https://rag-mcp-j56xbd7o2a-uc.a.run.app/mcp"
    }
  }
}
```

> **Note:** This file can be committed so the team shares one config.

## Step 2: Connect

1. Open VS Code and open the Copilot chat panel (Ctrl+Shift+I / Cmd+Shift+I).
2. Switch to **Agent mode** (click the mode selector at the top of the chat panel).
3. VS Code should detect the MCP server configuration automatically.
4. The `rag-search` server should appear as connected. You'll see the tools listed when you click the tools icon in the chat input.

## Step 3: Test the Connection

Try these prompts in Copilot chat (agent mode) to verify everything works:

### List documents

> What documents are in the RAG knowledge base?

Copilot will call the `list_documents` tool and show you all available documents with their types and chunk counts.

### Search for content

> Search the knowledge base for API authentication best practices

Copilot will call the `search` tool and return relevant document chunks ranked by relevance.

### Ask a question

> Based on the knowledge base, how should we handle error responses in our REST API?

Copilot will search the knowledge base, read the relevant chunks, and synthesize an answer.

## Available Tools

The MCP server exposes two tools:

### `search`

Search the team's document knowledge base using natural language.

| Parameter | Type | Default | Description |
|---|---|---|---|
| `query` | string | (required) | Natural language search query |
| `limit` | integer | 10 | Max documents to return (1-50) |

**How it works:** Your query is converted to a vector embedding, then matched against documents using hybrid search (vector similarity + full-text search with Reciprocal Rank Fusion). Results include the most relevant text chunks from each matching document.

**Example results:**

```
## 1. API Design Guidelines (guidelines)
Score: 0.0328

**[vector]** Use nouns for REST URLs. Return proper HTTP status codes.
Version your API with URL paths...

**[text]** Error responses should include a machine-readable error code,
a human-readable message, and a request ID for tracing...
```

### `list_documents`

Browse available documents in the knowledge base.

| Parameter | Type | Default | Description |
|---|---|---|---|
| `limit` | integer | 20 | Max documents to return |
| `offset` | integer | 0 | Skip this many documents (for pagination) |

**Example results:**

```
Found 12 documents (showing 12):
- API Design Guidelines [guidelines] (TEAM, 8 chunks)
- Deployment Runbook [runbook] (TEAM, 15 chunks)
- Q4 Architecture Review [report] (TEAM, 23 chunks)
```

## Example Prompts

Here are some effective ways to use the knowledge base from Copilot:

| What you want | Example prompt |
|---|---|
| Find specific docs | "Search the knowledge base for database migration procedures" |
| Browse everything | "List all documents in the knowledge base" |
| Answer a question | "According to the knowledge base, what's our deployment process?" |
| Compare approaches | "Search for load balancing strategies in the knowledge base" |
| Find recent docs | "List the last 5 documents added to the knowledge base" |
| Get details | "Search the knowledge base for error handling and show me the full context" |

## Optional: If IAM Auth Is Enabled Later

If the MCP Cloud Run service is later locked down (recommended), use this auth-enabled config instead:

```json
{
  "servers": {
    "rag-search": {
      "type": "http",
      "url": "https://rag-mcp-j56xbd7o2a-uc.a.run.app/mcp",
      "headers": {
        "Authorization": "Bearer ${input:rag-token}"
      }
    }
  },
  "inputs": [
    {
      "id": "rag-token",
      "type": "promptString",
      "description": "RAG MCP OIDC token",
      "password": true
    }
  ]
}
```

For that mode, users typically need:

- `roles/run.invoker` on `rag-mcp`
- A service account they can impersonate
- `roles/iam.serviceAccountTokenCreator` on that service account

Generate a token:

```bash
gcloud auth print-identity-token \
  --impersonate-service-account=SA_NAME@apexflow-ai.iam.gserviceaccount.com \
  --audiences="https://rag-mcp-j56xbd7o2a-uc.a.run.app"
```

Then paste it when VS Code prompts.

## Refreshing Your Token (IAM Mode Only)

When IAM auth is enabled and your token expires (after ~1 hour), you'll see "Authentication failed" errors. To reconnect:

1. Generate a new token:
   ```bash
   gcloud auth print-identity-token \
     --impersonate-service-account=SA_NAME@apexflow-ai.iam.gserviceaccount.com \
     --audiences="https://rag-mcp-j56xbd7o2a-uc.a.run.app"
   ```
2. In VS Code, open the Command Palette (Ctrl+Shift+P / Cmd+Shift+P).
3. Run **MCP: List Servers**.
4. Find `rag-search` and click **Restart** (or **Stop** then **Start**).
5. Paste the new token when prompted.

## Troubleshooting

### "Authentication failed"

- This applies only when IAM auth is enabled on `rag-mcp`.
- Your token may be expired. Generate a new one (see [Refreshing Your Token](#refreshing-your-token-iam-mode-only)).
- Make sure the audience in `gcloud auth print-identity-token` is the MCP Cloud Run service URL host (for example `https://rag-mcp-j56xbd7o2a-uc.a.run.app`), not the RAG service URL.
- Ensure your principal (or impersonated service account) has `roles/run.invoker` on `rag-mcp`.

### "The search service is temporarily unavailable"

- The RAG service may be cold-starting (can take 10-15 seconds on first request). Try again.
- If the problem persists, check with the platform team — the RAG service may be down.

### "No documents found"

- The knowledge base may be empty. Ask your team lead if documents have been ingested.
- Your search query may be too specific. Try broader terms.

### MCP server not showing in VS Code

- Make sure `.vscode/mcp.json` exists in your workspace root (not a subdirectory).
- Check that VS Code is version 1.99+.
- Check that the GitHub Copilot extension is installed and you're in **Agent mode**.
- Try reloading VS Code (Ctrl+Shift+P / Cmd+Shift+P > "Developer: Reload Window").

### "gcloud auth print-identity-token" fails

- This is expected with regular user credentials when using `--audiences`.
- Use `--impersonate-service-account` and ensure you have `roles/iam.serviceAccountTokenCreator` on that service account.
- Make sure you have access to the `apexflow-ai` GCP project.

## How It Works (Architecture)

```
VS Code Copilot
     |
     | MCP protocol (streamable HTTP)
     | + optional user OIDC token (only when IAM lock-down is enabled)
     v
MCP Server (rag-mcp on Cloud Run)
     |
     | HTTP + auto-minted OIDC token
     | (service account identity)
     v
RAG Service (rag-service on Cloud Run)
     |
     | asyncpg + Row-Level Security
     v
AlloyDB (rag_* tables)
```

- **Current state:** `rag-mcp` is public (`allUsers`), so VS Code can connect without a user token.
- **The MCP server** automatically mints its own OIDC token to call the RAG service — you don't need to configure this.
- **The RAG service** uses PostgreSQL Row-Level Security to ensure you only see documents belonging to your tenant.
- **Current MCP identity model:** the MCP server calls the RAG service using service identity, not per-user passthrough.
- **Document visibility caveat:** TEAM docs are tenant-visible; PRIVATE-owner scoping is not yet enforced per individual VS Code user in this integration.
