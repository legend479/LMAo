# Agent Server Enhancements - Quick Reference

## 🚀 Quick Start

### Using Prompt Templates
```python
from .prompt_templates import PromptTemplates, PromptType

# Get a prompt
prompt = PromptTemplates.get_prompt(
    PromptType.CODE_GENERATION,
    language="python",
    requirements="Create REST API",
    style_guide="PEP 8"
)

# Get system prompt
system_prompt = PromptTemplates.get_system_prompt("code_generation", language="python")
```

### Using Tool Intent Mapper
```python
from .tool_intent_mapper import ToolIntentMapper

mapper = ToolIntentMapper()
intent_match = mapper.classify_intent("Create a PDF and email it")

print(f"Intent: {intent_match.primary_intent}")
print(f"Confidence: {intent_match.confidence}")
print(f"Tools: {intent_match.suggested_tools}")
print(f"Sequence: {intent_match.tool_sequence}")
```

### Using Enhanced Planning
```python
from .enhanced_planning import EnhancedPlanningModule

planner = EnhancedPlanningModule()
await planner.initialize()

plan = await planner.create_plan(message, context)
# Plan includes optimized tool sequence with dependencies
```

## 📋 Available Prompt Types

| Prompt Type | Use Case | Key Parameters |
|-------------|----------|----------------|
| `INTENT_CLASSIFICATION` | Classify user intent | - |
| `RAG_SYNTHESIS` | Synthesize RAG results | `context`, `query` |
| `CODE_GENERATION` | Generate code | `language`, `requirements`, `style_guide` |
| `CODE_ANALYSIS` | Analyze code | `language`, `code` |
| `CONTENT_GENERATION` | Create content | `topic`, `audience`, `content_type` |
| `DOCUMENT_ANALYSIS` | Analyze documents | `document_content` |
| `MULTI_TOOL_SYNTHESIS` | Combine tool results | `original_query`, `tool_results` |
| `QUESTION_ANSWERING` | Answer questions | `context_info`, `query` |
| `EMAIL_COMPOSITION` | Compose emails | `tone`, `purpose`, `key_points` |

## 🎯 Enhanced Intent Types

| Intent Type | Description | Example Query |
|-------------|-------------|---------------|
| `KNOWLEDGE_RETRIEVAL` | Find information | "What are SOLID principles?" |
| `CODE_GENERATION` | Write code | "Create a Python function" |
| `CODE_EXECUTION` | Run/test code | "Execute this code" |
| `CONTENT_GENERATION` | Create content | "Write a tutorial" |
| `DOCUMENT_EXPORT` | Create documents | "Generate a PDF" |
| `EMAIL_AUTOMATION` | Send emails | "Email this to john@example.com" |
| `ANALYSIS` | Analyze content | "Review this code" |
| `MULTI_STEP` | Multiple tasks | "Create PDF and email it" |
| `GENERAL_QUERY` | General questions | "How are you?" |

## 🔧 Tool Capabilities

| Tool | Primary Intent | Can Chain With | Keywords |
|------|----------------|----------------|----------|
| `knowledge_retrieval` | KNOWLEDGE_RETRIEVAL | content_generation, document_generation | explain, what is, find |
| `code_generation` | CODE_GENERATION | compiler_runtime, document_generation | write code, implement |
| `compiler_runtime` | CODE_EXECUTION | document_generation, email_automation | run, execute, test |
| `content_generation` | CONTENT_GENERATION | document_generation, email_automation | generate content, tutorial |
| `document_generation` | DOCUMENT_EXPORT | email_automation | create pdf, export to |
| `email_automation` | EMAIL_AUTOMATION | - | send email, mail to |

## 🔗 Common Tool Chains

### Content → Document
```python
# Query: "Create a PDF about Python"
# Chain: content_generation → document_generation
```

### Content → Document → Email
```python
# Query: "Create a PDF about Python and email it"
# Chain: content_generation → document_generation → email_automation
```

### Code → Execute
```python
# Query: "Write a function and test it"
# Chain: code_generation → compiler_runtime
```

### Knowledge → Content → Document
```python
# Query: "Research design patterns and create a tutorial PDF"
# Chain: knowledge_retrieval → content_generation → document_generation
```

## 🔄 Parameter Injection

### Syntax
```python
# In task parameters
{
    "content": "{{ task_1.result }}",  # Inject result from task_1
    "code": "{{ task_2.data }}",       # Inject data from task_2
}
```

### Example
```python
# Task 1: Generate content
{
    "id": "task_1",
    "type": "content_generation",
    "parameters": {"topic": "Python"}
}

# Task 2: Create document (uses Task 1 output)
{
    "id": "task_2",
    "type": "tool_execution",
    "tool": "document_generation",
    "parameters": {
        "content": "{{ task_1.result }}",  # Auto-injected
        "format": "pdf"
    },
    "dependencies": ["task_1"]
}
```

## 📊 Confidence Scoring

### Interpretation
- `0.9 - 1.0`: Very confident, proceed with classification
- `0.7 - 0.9`: Confident, good classification
- `0.5 - 0.7`: Moderate confidence, may need clarification
- `< 0.5`: Low confidence, ask user for clarification

### Usage
```python
intent_match = mapper.classify_intent(message)

if intent_match.confidence < 0.5:
    # Ask for clarification
    response = "I'm not sure I understand. Could you clarify?"
elif intent_match.confidence < 0.7:
    # Confirm with user
    response = f"Do you want to {intent_match.primary_intent.value}?"
else:
    # Proceed with high confidence
    plan = create_plan(intent_match)
```

## 🐛 Debugging Tips

### Check Intent Classification
```python
intent_match = mapper.classify_intent(message)
print(f"Intent: {intent_match.primary_intent.value}")
print(f"Confidence: {intent_match.confidence:.2f}")
print(f"Reasoning: {intent_match.reasoning}")
print(f"Suggested tools: {intent_match.suggested_tools}")
```

### Check Tool Sequence
```python
for i, (tool, params) in enumerate(intent_match.tool_sequence):
    print(f"Step {i+1}: {tool}")
    print(f"  Parameters: {params}")
```

### Check Parameter Extraction
```python
params = mapper.get_tool_parameters("email_automation", message)
print(f"Extracted parameters: {params}")
```

### Enable Detailed Logging
```python
import logging
logging.getLogger("agent_server").setLevel(logging.DEBUG)
```

## ⚠️ Common Issues

### Issue: Intent misclassified
**Solution**: Check confidence score, add more keywords to patterns

### Issue: Tool chain not working
**Solution**: Verify dependencies are set correctly, check parameter injection syntax

### Issue: Parameters not extracted
**Solution**: Check regex patterns in `get_tool_parameters`, add more patterns

### Issue: Prompt not rendering
**Solution**: Verify all required parameters are provided to `get_prompt()`

## 📈 Performance Tips

### 1. Cache Intent Classifications
```python
# For repeated queries
intent_cache = {}
cache_key = hash(message)
if cache_key in intent_cache:
    return intent_cache[cache_key]
```

### 2. Parallel Tool Execution
```python
# Identify independent tasks
parallel_groups = plan.parallel_groups
# Execute groups in parallel
```

### 3. Optimize Prompts
```python
# Use shorter prompts for simple tasks
# Use detailed prompts for complex tasks
if complexity == "simple":
    max_tokens = 500
else:
    max_tokens = 2000
```

## 🔐 Security Considerations

### Email Automation
- Validate email addresses
- Check recipient whitelist
- Limit attachment sizes
- Scan attachments for malware

### Code Execution
- Use sandboxed environment
- Set execution timeouts
- Limit resource usage
- Validate code before execution

### Document Generation
- Sanitize user input
- Limit file sizes
- Validate file formats
- Check for malicious content

## 📚 Additional Resources

- **IMPLEMENTATION_GUIDE.md**: Detailed integration steps
- **ORCHESTRATOR_IMPROVEMENTS.md**: Specific code changes
- **ENHANCEMENT_SUMMARY.md**: Complete overview
- **IMPROVEMENTS_PLAN.md**: High-level plan

## 🆘 Getting Help

1. Check logs: `logger.debug()` statements throughout
2. Review intent reasoning: `intent_match.reasoning`
3. Examine tool sequence: `intent_match.tool_sequence`
4. Monitor confidence: `intent_match.confidence`
5. Test with simple queries first, then complex ones

## ✅ Testing Checklist

- [ ] Prompt templates render correctly
- [ ] Intent classification works for all types
- [ ] Tool sequences are logical
- [ ] Parameters are extracted correctly
- [ ] Dependencies are set properly
- [ ] Parameter injection works
- [ ] Error handling is robust
- [ ] Logging is comprehensive
- [ ] Performance is acceptable
- [ ] Security measures are in place
