# Agent Server Interactive CLI

A comprehensive interactive command-line interface for testing and exploring all features of the Agent Server - a sophisticated AI orchestration system built with LangGraph.

## 🎯 What This Is

This interactive CLI provides a user-friendly way to:
- Test all Agent Server components
- Process messages through the AI agent
- Execute tools dynamically
- Visualize planning and task decomposition
- Inspect execution traces and reasoning
- Monitor memory and conversation context
- View system statistics and metrics

## 📁 Files Included

| File | Purpose |
|------|---------|
| `agent_interactive_cli.py` | Main interactive CLI application |
| `test_agent_server.py` | Quick test script to verify setup |
| `AGENT_SERVER_FEATURES.md` | Complete feature documentation |
| `AGENT_SERVER_FIX.md` | Fix documentation for initialization issue |
| `CLI_QUICK_START.md` | Quick start guide with examples |
| `README_AGENT_CLI.md` | This file |

## 🚀 Quick Start

### 1. Test the Fix
```bash
python test_agent_server.py
```

Expected output:
```
✓ Imported AgentServer
✓ Created AgentServer instance
✓ Initialized AgentServer
✓ Agent Server test completed successfully!
```

### 2. Run the Interactive CLI
```bash
python agent_interactive_cli.py
```

### 3. Explore Features
Follow the interactive menu to explore all capabilities!

## 🎨 Features

### Core Capabilities

#### 1. **Message Processing** 💬
- Natural language understanding
- Context-aware responses
- Multi-step task execution
- Streaming support (planned)

#### 2. **Planning & Orchestration** 🧠
- Intent classification (7 types)
- Entity extraction
- Hierarchical task decomposition
- Dependency analysis
- Parallel execution optimization
- Recovery strategy planning

#### 3. **Tool Execution** 🔧
- Dynamic tool discovery
- Multi-criteria tool selection
- Resource management
- Performance monitoring
- Concurrent execution

#### 4. **Memory Management** 💾
- Intelligent context pruning
- Conversation summarization
- User profiling
- Multi-level memory (short-term, working, long-term)
- Relevance scoring

#### 5. **Execution Traces** 🔍
- Workflow path visualization
- Checkpoint tracking
- Tool result inspection
- Reasoning transparency
- Error tracking

#### 6. **LangGraph Orchestration** 🌳
- Stateful workflow management
- Redis-backed checkpointing
- Error recovery with retry logic
- Conditional routing
- Parallel task execution

## 📊 Architecture Overview

```
┌─────────────────────────────────────────────────────────────┐
│                     Agent Server                            │
├─────────────────────────────────────────────────────────────┤
│                                                             │
│  ┌──────────────┐  ┌──────────────┐  ┌──────────────┐    │
│  │  LangGraph   │  │   Planning   │  │    Memory    │    │
│  │ Orchestrator │  │    Module    │  │   Manager    │    │
│  └──────────────┘  └──────────────┘  └──────────────┘    │
│                                                             │
│  ┌──────────────┐  ┌──────────────┐  ┌──────────────┐    │
│  │     Tool     │  │     Code     │  │   Content    │    │
│  │   Registry   │  │  Generation  │  │  Generation  │    │
│  └──────────────┘  └──────────────┘  └──────────────┘    │
│                                                             │
└─────────────────────────────────────────────────────────────┘
                            │
                            ▼
                    ┌───────────────┐
                    │  Redis Store  │
                    │  (Checkpoints │
                    │   & Memory)   │
                    └───────────────┘
```

## 🛠️ Components

### 1. LangGraph Orchestrator
- **Purpose:** Stateful workflow management
- **Features:**
  - Dynamic workflow graph creation
  - Redis-backed state persistence
  - Error recovery with multiple strategies
  - Parallel task execution
  - Conditional routing

### 2. Planning Module
- **Purpose:** Intelligent task decomposition
- **Features:**
  - LLM-based intent classification
  - Entity extraction (languages, frameworks, tools)
  - Complexity assessment
  - Goal identification and prioritization
  - Dependency analysis

### 3. Memory Manager
- **Purpose:** Context-aware conversation management
- **Features:**
  - Intelligent context pruning
  - Relevance-based message scoring
  - Automatic summarization
  - User profiling
  - Multi-level memory hierarchy

### 4. Tool Registry
- **Purpose:** Dynamic tool management
- **Features:**
  - Tool registration and discovery
  - Multi-criteria selection
  - Resource management
  - Performance monitoring
  - SQLite persistence

### 5. Code Generation
- **Purpose:** Generate and validate code
- **Features:**
  - Multi-language support (10+ languages)
  - Quality assessment (5 metrics)
  - Functional validation
  - Test execution
  - Security scanning

### 6. Content Generation
- **Purpose:** Educational content creation
- **Features:**
  - Audience adaptation (K12 to Expert)
  - Readability optimization
  - Technical term explanations
  - Example generation

## 📖 Documentation

### Quick References
- **Quick Start:** See `CLI_QUICK_START.md`
- **Features:** See `AGENT_SERVER_FEATURES.md`
- **Fix Details:** See `AGENT_SERVER_FIX.md`

### Key Concepts

#### Intent Types
- `KNOWLEDGE_RETRIEVAL` - Learning and understanding
- `CONTENT_GENERATION` - Creating content
- `CODE_GENERATION` - Writing code
- `DOCUMENT_GENERATION` - Creating documents
- `ANALYSIS` - Analyzing and evaluating
- `MULTI_STEP` - Complex multi-step tasks
- `GENERAL_QUERY` - Simple questions

#### Complexity Levels
- `SIMPLE` - Basic queries
- `MODERATE` - Standard requests
- `COMPLEX` - Multi-faceted tasks
- `VERY_COMPLEX` - Advanced multi-step operations

#### Memory Types
- `SHORT_TERM` - Last 10 messages
- `WORKING` - 20 relevant messages
- `LONG_TERM` - 90 days retention
- `EPISODIC` - Conversation summaries
- `SEMANTIC` - User preferences

## 🔧 Configuration

### Environment Variables (.env)
```env
# LLM Configuration
LLM_PROVIDER=google
GOOGLE_API_KEY=your-api-key-here
LLM_MODEL=gemini-2.5-flash

# Redis
REDIS_URL=redis://localhost:6379/0

# Logging
LOG_LEVEL=INFO
LOG_FORMAT=pretty

# Environment
ENVIRONMENT=development
DEBUG=true
```

## 🎯 Use Cases

### 1. Development & Testing
- Test agent responses
- Debug workflow execution
- Validate tool integration
- Monitor performance

### 2. Learning & Exploration
- Understand AI orchestration
- Learn LangGraph patterns
- Explore planning strategies
- Study memory management

### 3. Demonstration
- Showcase capabilities
- Present to stakeholders
- Training sessions
- Feature validation

### 4. Debugging
- Trace execution paths
- Inspect state transitions
- Analyze failures
- Monitor resource usage

## 🐛 Troubleshooting

### Common Issues

#### 1. Initialization Error
**Fixed!** The `ToolRegistryManager` now has the required `initialize()` method.

#### 2. Ollama Warnings
**Harmless!** These warnings can be ignored if you're using Google AI.

#### 3. Redis Connection
**Optional!** The system works without Redis, but some features are limited.

#### 4. Import Errors
**Solution:** Run `pip install rich fastapi redis langgraph langchain_core`

## 📈 Performance

### Typical Metrics
- Message processing: 1-5 seconds
- Planning: 0.5-2 seconds
- Tool execution: Varies by tool
- Memory operations: <100ms
- Context pruning: <200ms

### Resource Usage
- Memory: ~200-500 MB
- CPU: Low (mostly I/O bound)
- Redis: ~10-50 MB
- Network: Depends on LLM provider

## 🔒 Security

### Best Practices
- Keep API keys in `.env` (not in code)
- Use environment-specific configurations
- Enable rate limiting in production
- Validate all user inputs
- Monitor resource usage
- Implement timeout policies

## 🚦 Status

### What Works ✅
- ✅ Agent Server initialization
- ✅ Message processing
- ✅ Planning and task decomposition
- ✅ Memory management
- ✅ Tool registry
- ✅ Execution traces
- ✅ Health monitoring
- ✅ Interactive CLI

### Known Limitations ⚠️
- ⚠️ No tools registered by default (need to add tools)
- ⚠️ Ollama warnings (harmless, can be ignored)
- ⚠️ Limited without Redis (some features)
- ⚠️ Streaming not yet implemented in CLI

### Future Enhancements 🚀
- 🚀 Web UI dashboard
- 🚀 More built-in tools
- 🚀 Streaming responses
- 🚀 Multi-modal support
- 🚀 Advanced analytics
- 🚀 Plugin system

## 📝 Example Session

```bash
$ python agent_interactive_cli.py

╔═══════════════════════════════════════════════════════════════╗
║           AGENT SERVER INTERACTIVE CLI                        ║
║     Complete Testing & Verification Tool                      ║
╠═══════════════════════════════════════════════════════════════╣
║  🤖 Orchestration | 🧠 Planning | 🔧 Tools | 💾 Memory       ║
╚═══════════════════════════════════════════════════════════════╝

🔧 Initial Setup
✅ Dependencies OK
✅ Redis running
✅ Agent Server ready

🎯 Agent Server Interactive CLI

Choose an option:
1.  🏥 Health Check
2.  🚀 Complete Demo
3.  💬 Process Message
...

Enter your choice (1-12): 3

Enter your message: Explain design patterns

💬 Processing: 'Explain design patterns'

📤 Agent Response:
╭────────────────────────────────────────────────────────────────╮
│ Design patterns are reusable solutions to common software     │
│ design problems. They represent best practices...             │
╰────────────────────────────────────────────────────────────────╯

✅ Message processed successfully!
```

## 🤝 Contributing

To extend the CLI:
1. Add methods to `AgentServerInteractiveCLI` class
2. Update the menu in `show_main_menu()`
3. Add handlers in `run_interactive_session()`
4. Update documentation

## 📚 Additional Resources

- **LangGraph Docs:** https://langchain-ai.github.io/langgraph/
- **Redis Docs:** https://redis.io/docs/
- **Rich Library:** https://rich.readthedocs.io/

## 🎓 Learning Path

1. **Beginner:** Run the complete demo, explore basic features
2. **Intermediate:** Process custom messages, view planning
3. **Advanced:** Inspect traces, understand orchestration
4. **Expert:** Extend the system, add custom tools

## 📞 Support

For issues or questions:
1. Check `AGENT_SERVER_FIX.md` for known issues
2. Review `CLI_QUICK_START.md` for usage help
3. Read `AGENT_SERVER_FEATURES.md` for details
4. Check the inline code comments

## 🎉 Summary

You now have:
- ✅ A fully functional Agent Server
- ✅ An interactive CLI to explore all features
- ✅ Complete documentation
- ✅ Test scripts
- ✅ Quick start guides
- ✅ Troubleshooting help

**Ready to explore AI orchestration!** 🚀

Start with:
```bash
python test_agent_server.py  # Verify setup
python agent_interactive_cli.py  # Start exploring!
```

Enjoy! 🎊
