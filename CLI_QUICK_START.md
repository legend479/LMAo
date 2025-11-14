# Agent Server Interactive CLI - Quick Start Guide

## Installation & Setup

### 1. Prerequisites
- Python 3.9+
- Redis running on localhost:6379 (optional but recommended)
- Google AI API key configured in `.env`

### 2. Install Dependencies
```bash
pip install rich fastapi redis langgraph langchain_core
```

Or let the CLI auto-install them when you run it.

### 3. Verify Configuration
Check your `.env` file has:
```env
LLM_PROVIDER=google
GOOGLE_API_KEY=your-api-key-here
REDIS_URL=redis://localhost:6379/0
```

## Running the CLI

### Quick Test
```bash
# Test that everything works
python test_agent_server.py
```

### Start Interactive CLI
```bash
python agent_interactive_cli.py
```

## CLI Features Overview

### Main Menu Options

```
1.  🏥 Health Check           - Verify all components are running
2.  🚀 Complete Demo          - Automated showcase of all features
3.  💬 Process Message        - Chat with the agent
4.  🔧 List Tools             - See all registered tools
5.  ⚙️  Execute Tool           - Run a specific tool
6.  🧠 Show Planning          - View task decomposition
7.  🔍 Execution Traces       - See workflow paths & reasoning
8.  💾 Memory Context         - View conversation memory
9.  📜 Conversation History   - Review past interactions
10. 📊 Statistics             - System metrics
11. 📚 Help                   - Usage information
12. 🚪 Exit                   - Quit application
```

## Common Use Cases

### 1. First Time Setup
```
1. Run the CLI: python agent_interactive_cli.py
2. Choose option 1 (Health Check) to verify everything works
3. Choose option 2 (Complete Demo) to see all features
```

### 2. Chat with the Agent
```
1. Choose option 3 (Process Message)
2. Enter your question or request
3. View the agent's response and metadata
```

Example messages to try:
- "Explain what design patterns are"
- "Generate a Python function to calculate fibonacci numbers"
- "Create a REST API endpoint for user management"
- "What are the best practices for code documentation?"

### 3. Explore Planning
```
1. Choose option 6 (Show Planning)
2. Enter a complex query like:
   "Create a web application with user authentication and a dashboard"
3. See how the agent breaks it down into tasks
```

### 4. View Execution Traces
```
1. First process a message (option 3)
2. Then choose option 7 (Execution Traces)
3. See the workflow path, tool results, and reasoning
```

### 5. Check Memory & Context
```
1. After having a conversation, choose option 8 (Memory Context)
2. See conversation history, current topic, and user preferences
3. Understand how the agent maintains context
```

## Understanding the Output

### Health Check Output
```
Component Health Status
┏━━━━━━━━━━━━━━━━━━━━━━━━━┳━━━━━━━━━━━━━━━┳━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┓
┃ Component               ┃ Status        ┃ Details                                ┃
┡━━━━━━━━━━━━━━━━━━━━━━━━━╇━━━━━━━━━━━━━━━╇━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┩
│ Agent Server            │ healthy       │ Main orchestration engine              │
│ LangGraph Orchestrator  │ healthy       │ Workflow management                    │
│ Planning Module         │ healthy       │ Task decomposition & planning          │
│ Memory Manager          │ healthy       │ Conversation context management        │
│ Tool Registry           │ healthy       │ 0 tools registered                     │
└─────────────────────────┴───────────────┴────────────────────────────────────────┘
```

### Message Processing Output
```
💬 Processing: 'Explain design patterns'

📤 Agent Response:
╭────────────────────────────────────────────────────────────────────────────╮
│ Design patterns are reusable solutions to common software design          │
│ problems. They represent best practices and provide templates for solving  │
│ recurring design challenges...                                             │
╰────────────────────────────────────────────────────────────────────────────╯

Execution Metadata
┏━━━━━━━━━━━━━━━━━━━━━━┳━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┓
┃ Key                  ┃ Value                                              ┃
┡━━━━━━━━━━━━━━━━━━━━━━╇━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┩
│ plan_id              │ abc123...                                          │
│ tasks_completed      │ 3                                                  │
│ execution_time       │ 2.45s                                              │
└──────────────────────┴────────────────────────────────────────────────────┘
```

### Planning Output
```
📋 Execution Plan Created

Plan Overview
┏━━━━━━━━━━━━━━━━━━━━━━┳━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┓
┃ Attribute            ┃ Value                                              ┃
┡━━━━━━━━━━━━━━━━━━━━━━╇━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┩
│ Plan ID              │ plan_xyz789                                        │
│ Total Tasks          │ 5                                                  │
│ Estimated Duration   │ 15.30s                                             │
│ Priority             │ 2                                                  │
└──────────────────────┴────────────────────────────────────────────────────┘

Tasks:
┏━━━━━┳━━━━━━━━━━━━━━━━━━━━━━┳━━━━━━━━━━━━━━━━━━━━━━┳━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┓
┃ #   ┃ Task ID              ┃ Type                 ┃ Description                                      ┃
┡━━━━━╇━━━━━━━━━━━━━━━━━━━━━━╇━━━━━━━━━━━━━━━━━━━━━━╇━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┩
│ 1   │ task_1               │ knowledge_retrieval  │ Retrieve information about design patterns       │
│ 2   │ task_2               │ content_generation   │ Generate explanation content                     │
│ 3   │ task_3               │ analysis             │ Analyze complexity level                         │
└─────┴──────────────────────┴──────────────────────┴──────────────────────────────────────────────────┘
```

### Execution Traces Output
```
🔍 Execution Traces & Reasoning

Execution #1

Execution Path:
🌳 Workflow
├── start
├── task_1
├── task_2
└── task_3

Tool Results:
  • task_1
  • task_2

Metadata:
┏━━━━━━━━━━━━━━━━━━━━━━┳━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┓
┃ Key                  ┃ Value                                              ┃
┡━━━━━━━━━━━━━━━━━━━━━━╇━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┩
│ tasks_planned        │ 5                                                  │
│ tasks_completed      │ 5                                                  │
│ tasks_failed         │ 0                                                  │
└──────────────────────┴────────────────────────────────────────────────────┘
```

## Troubleshooting

### Issue: Redis Connection Failed
**Symptom:** Warning about Redis not being available

**Solution:**
1. Start Redis: `redis-server`
2. Or continue without Redis (some features will be limited)

### Issue: Ollama Warnings
**Symptom:** Warnings about Ollama connection failures

**Solution:** These are harmless warnings. The system correctly uses Google AI as configured. You can ignore them.

### Issue: Import Errors
**Symptom:** ModuleNotFoundError for various packages

**Solution:**
```bash
pip install rich fastapi redis langgraph langchain_core google-generativeai
```

### Issue: API Key Errors
**Symptom:** Authentication errors with Google AI

**Solution:**
1. Check your `.env` file has the correct `GOOGLE_API_KEY`
2. Verify the API key is valid and has quota
3. Check the key has the right permissions

## Advanced Usage

### Custom Session ID
The CLI automatically creates a session ID, but you can modify it in the code:
```python
self.session_id = "my_custom_session"
```

### Viewing Raw Data
For debugging, you can access raw data structures:
```python
# In the CLI code
print(json.dumps(result, indent=2))
```

### Extending the CLI
Add new menu options by:
1. Adding a new method to the `AgentServerInteractiveCLI` class
2. Adding a menu option in `show_main_menu()`
3. Adding a case in `run_interactive_session()`

## Tips & Best Practices

1. **Start with Health Check** - Always verify components are working
2. **Use Complete Demo** - Great way to learn all features
3. **Try Simple Messages First** - Build up to complex queries
4. **Check Execution Traces** - Understand how the agent thinks
5. **Monitor Memory** - See how context is maintained
6. **Review Statistics** - Track performance and usage

## Example Session

```
1. Start CLI: python agent_interactive_cli.py
2. Choose 1 (Health Check) - Verify everything works ✓
3. Choose 3 (Process Message) - Ask "What are design patterns?"
4. Choose 7 (Execution Traces) - See how it was processed
5. Choose 3 (Process Message) - Ask "Give me an example"
6. Choose 8 (Memory Context) - See it remembers the topic
7. Choose 10 (Statistics) - View session metrics
8. Choose 12 (Exit) - Clean shutdown
```

## Getting Help

- **In CLI:** Choose option 11 (Help)
- **Documentation:** Read `AGENT_SERVER_FEATURES.md`
- **Fix Guide:** Read `AGENT_SERVER_FIX.md`
- **Code:** Check the source code with inline comments

## Next Steps

After getting comfortable with the CLI:
1. Explore the codebase to understand the architecture
2. Try creating custom tools
3. Experiment with different planning strategies
4. Build your own agent applications using the components

Enjoy exploring the Agent Server! 🚀
