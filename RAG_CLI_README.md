# RAG Pipeline Interactive CLI

**Single script for testing all RAG pipeline features**

## 🚀 Quick Start

```bash
python rag_interactive_cli.py
```

The script automatically:
- Checks dependencies and installs `rich`
- Sets up Elasticsearch (with Docker option)
- Creates test documents
- Initializes RAG pipeline
- Provides interactive menu

## 📋 Features

1. **Health Check** - Verify system status
2. **Complete Demo** - Automated feature showcase
3. **Document Ingestion** - Add files/directories
4. **Search Testing** - Hybrid, keyword, vector search
5. **Method Comparison** - Side-by-side search comparison
6. **Statistics** - Performance metrics

## 🔍 Search Methods

- **Hybrid** - Keyword + vector fusion (recommended)
- **Keyword** - Traditional BM25 text search
- **Vector** - Semantic similarity with embeddings

## 🔧 Requirements

- Python 3.9+
- Elasticsearch on localhost:9200
- Internet connection for model downloads

## 🐛 Troubleshooting

**Elasticsearch not running:**
```bash
docker run -p 9200:9200 -e "discovery.type=single-node" elasticsearch:8.11.0
```

**Missing dependencies:**
```bash
pip install -r requirements.txt
```

**No search results:**
- Run complete demo first to ingest test documents
- Check statistics to verify document indexing

## 📊 What's Tested

- Multi-format document processing (PDF, DOCX, TXT, MD)
- Advanced chunking strategies
- Dual embedding system (general + domain-specific)
- Elasticsearch vector storage
- Hybrid search with RRF fusion
- Performance monitoring

---

**Start exploring:** `python rag_interactive_cli.py`