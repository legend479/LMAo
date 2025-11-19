# Quick Reference - Agent Server

## 🚀 Quick Start

### Start Everything
```bash
# Terminal 1: Redis
redis-server

# Terminal 2: RAG Server (optional)
python -m src.rag_pipeline.main

# Terminal 3: Agent CLI
python agent_cli.py
```

### First Steps
1. CLI starts automatically
2. Health check runs
3. Choose Option 1 (Process Message)
4. Enter your query
5. View response

## 📋 Menu Options

| # | Option | Purpose |
|---|--------|---------|
| 1 | Process Message | Send message to agent |
| 2 | List Tools | Show available tools |
| 3 | System Health | Check component status |
| 4 | Execution Trace | View execution details |
| 5 | Statistics | Session metrics |
| 6 | Help | Usage information |
| 7 | Exit | Quit application |

## 🔧 Available Tools

| Tool | Purpose | Example |
|------|---------|---------|
| knowledge_retrieval | Search knowledge base | "What are design patterns?" |
| document_generation | Create PDF/DOCX/PPTX | Create API documentation |
| compiler_runtime | Execute code | Run Python code |
| email_automation | Send emails | Email document to team |
| readability_scoring | Assess content | Check content quality |

## 💡 Common Queries

### Knowledge Retrieval
```
"What are SOLID principles?"
"Explain design patterns"
"How does REST API work?"
```

### Code Generation
```
"Write a Python function to sort a list"
"Create a REST API endpoint"
"Implement binary search"
```

### Document Creation
```
"Create a PDF about design patterns"
"Generate documentation for API"
"Export tutorial to DOCX"
```

### Multi-Tool
```
"Create a PDF about Python and email it"
"Write code and test it"
"Research topic and create document"
```

## 🐛 Troubleshooting

| Problem | Solution |
|---------|----------|
| Initialization failed | Check Redis is running |
| No response | Check System Health (Option 3) |
| RAG offline | Start: `python -m src.rag_pipeline.main` |
| Slow response | Check execution trace (Option 4) |

## 📊 Status Indicators

| Symbol | Meaning |
|--------|---------|
| ✓ | Working correctly |
| ✗ | Not working |
| ○ | Optional/Not required |

## ⚡ Quick Commands

```bash
# Start Redis
redis-server

# Start RAG
python -m src.rag_pipeline.main

# Production CLI
python agent_cli.py

# Development CLI
python agent_interactive_cli.py

# Check health
curl http://localhost:8001/health  # RAG
curl http://localhost:8000/health  # Agent
```

## 📖 Documentation

| Document | Purpose |
|----------|---------|
| PRODUCTION_CLI_GUIDE.md | Production CLI manual |
| CLI_COMPARISON.md | CLI comparison |
| CODE_REVIEW_SUMMARY.md | Code quality review |
| RAG_SERVER_INTEGRATION.md | RAG integration |
| FINAL_SUMMARY.md | Complete summary |

## ✅ Pre-flight Checklist

Before using:
- [ ] Redis running
- [ ] RAG server running (optional)
- [ ] .env configured
- [ ] Dependencies installed

## 🎯 Success Path

```
1. Start Redis
2. Start RAG (optional)
3. Launch CLI
4. Check health (Option 3)
5. Process message (Option 1)
6. View trace (Option 4)
7. Success!
```

---

**Quick Start**: `python agent_cli.py`
**Status**: ✅ Production Ready
