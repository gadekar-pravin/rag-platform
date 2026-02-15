# RAG Platform Guide

## Overview

The RAG platform provides **hybrid search** capabilities for document retrieval.

## Code Example

```python
result = await store.search_hybrid(query="test", limit=10)
```

## Links

See the [deployment guide](docs/deployment.md) for production setup.

## Features

1. Vector similarity search
2. Full-text search with tsvector
3. Reciprocal Rank Fusion
