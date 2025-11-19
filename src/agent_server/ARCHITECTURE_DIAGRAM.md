# Agent Server Enhanced Architecture

## System Flow Diagram

```
┌─────────────────────────────────────────────────────────────────┐
│                         USER REQUEST                             │
│              "Create a PDF about Python and email it"            │
└────────────────────────────┬────────────────────────────────────┘
                             │
                             ▼
┌─────────────────────────────────────────────────────────────────┐
│                      AGENT SERVER (main.py)                      │
│  ┌──────────────────────────────────────────────────────────┐  │
│  │  1. Receive message                                       │  │
│  │  2. Get conversation context from MemoryManager          │  │
│  │  3. Create execution plan via PlanningModule             │  │
│  │  4. Execute plan via Orchestrator                        │  │
│  │  5. Store interaction in MemoryManager                   │  │
│  └──────────────────────────────────────────────────────────┘  │
└────────────────────────────┬────────────────────────────────────┘
                             │
                             ▼
┌─────────────────────────────────────────────────────────────────┐
│              ENHANCED PLANNING MODULE (enhanced_planning.py)     │
│  ┌──────────────────────────────────────────────────────────┐  │
│  │  Step 1: Intent Classification                           │  │
│  │  ┌────────────────────────────────────────────────────┐  │  │
│  │  │  ToolIntentMapper.classify_intent()                │  │  │
│  │  │  - Pattern matching (regex)                        │  │  │
│  │  │  - Keyword detection                               │  │  │
│  │  │  - Context analysis                                │  │  │
│  │  │  Result: DOCUMENT_EXPORT + EMAIL_AUTOMATION        │  │  │
│  │  │  Confidence: 0.92                                  │  │  │
│  │  └────────────────────────────────────────────────────┘  │  │
│  │                                                           │  │
│  │  Step 2: Tool Sequence Generation                        │  │
│  │  ┌────────────────────────────────────────────────────┐  │  │
│  │  │  ToolIntentMapper.create_tool_sequence()           │  │  │
│  │  │  Result:                                           │  │  │
│  │  │  1. content_generation (topic: "Python")           │  │  │
│  │  │  2. document_generation (format: "pdf")            │  │  │
│  │  │  3. email_automation (attach: previous)            │  │  │
│  │  └────────────────────────────────────────────────────┘  │  │
│  │                                                           │  │
│  │  Step 3: Create Execution Plan                           │  │
│  │  ┌────────────────────────────────────────────────────┐  │  │
│  │  │  - Convert tool sequence to tasks                  │  │  │
│  │  │  - Set up dependencies                             │  │  │
│  │  │  - Add parameter injection placeholders            │  │  │
│  │  │  - Define recovery strategies                      │  │  │
│  │  └────────────────────────────────────────────────────┘  │  │
│  └──────────────────────────────────────────────────────────┘  │
└────────────────────────────┬────────────────────────────────────┘
                             │
                             ▼
┌─────────────────────────────────────────────────────────────────┐
│                 ORCHESTRATOR (orchestrator.py)                   │
│  ┌──────────────────────────────────────────────────────────┐  │
│  │  Step 1: Create Workflow Graph                           │  │
│  │  ┌────────────────────────────────────────────────────┐  │  │
│  │  │  Task 1: content_generation                        │  │  │
│  │  │    ↓                                               │  │  │
│  │  │  Task 2: document_generation                       │  │  │
│  │  │    ↓                                               │  │  │
│  │  │  Task 3: email_automation                          │  │  │
│  │  └────────────────────────────────────────────────────┘  │  │
│  │                                                           │  │
│  │  Step 2: Execute Tasks Sequentially                      │  │
│  │  ┌────────────────────────────────────────────────────┐  │  │
│  │  │  Task 1 Execution:                                 │  │  │
│  │  │  ┌──────────────────────────────────────────────┐  │  │  │
│  │  │  │ 1. Get prompt from PromptTemplates           │  │  │  │
│  │  │  │ 2. Call LLM with optimized prompt            │  │  │  │
│  │  │  │ 3. Store result: "Python is a..."           │  │  │  │
│  │  │  └──────────────────────────────────────────────┘  │  │  │
│  │  │                                                    │  │  │
│  │  │  Task 2 Execution:                                 │  │  │
│  │  │  ┌──────────────────────────────────────────────┐  │  │  │
│  │  │  │ 1. Resolve parameters:                       │  │  │  │
│  │  │  │    "{{ task_1.result }}" → "Python is a..."  │  │  │  │
│  │  │  │ 2. Call document_generation tool             │  │  │  │
│  │  │  │ 3. Store result: "/path/to/python.pdf"       │  │  │  │
│  │  │  └──────────────────────────────────────────────┘  │  │  │
│  │  │                                                    │  │  │
│  │  │  Task 3 Execution:                                 │  │  │
│  │  │  ┌──────────────────────────────────────────────┐  │  │  │
│  │  │  │ 1. Resolve parameters:                       │  │  │  │
│  │  │  │    attachment = "{{ task_2.result }}"        │  │  │  │
│  │  │  │ 2. Call email_automation tool                │  │  │  │
│  │  │  │ 3. Store result: "Email sent successfully"   │  │  │  │
│  │  │  └──────────────────────────────────────────────┘  │  │  │
│  │  └────────────────────────────────────────────────────┘  │  │
│  │                                                           │  │
│  │  Step 3: Synthesize Final Response                       │  │
│  │  ┌────────────────────────────────────────────────────┐  │  │
│  │  │  - Gather all task results                         │  │  │
│  │  │  - Use MULTI_TOOL_SYNTHESIS prompt                 │  │  │
│  │  │  - Generate coherent response                      │  │  │
│  │  │  Result: "I've created a PDF about Python and      │  │  │
│  │  │          emailed it successfully."                 │  │  │
│  │  └────────────────────────────────────────────────────┘  │  │
│  └──────────────────────────────────────────────────────────┘  │
└────────────────────────────┬────────────────────────────────────┘
                             │
                             ▼
┌─────────────────────────────────────────────────────────────────┐
│                         RESPONSE TO USER                         │
│  "I've created a PDF document about Python and emailed it       │
│   successfully. The document covers key concepts and best       │
│   practices."                                                    │
└─────────────────────────────────────────────────────────────────┘
```

## Component Interaction Diagram

```
┌──────────────────┐
│   User Request   │
└────────┬─────────┘
         │
         ▼
┌────────────────────────────────────────────────────────┐
│                    Agent Server                        │
│                                                        │
│  ┌──────────────┐  ┌──────────────┐  ┌─────────────┐ │
│  │   Memory     │  │   Planning   │  │ Orchestrator│ │
│  │   Manager    │  │   Module     │  │             │ │
│  └──────┬───────┘  └──────┬───────┘  └──────┬──────┘ │
│         │                 │                  │        │
└─────────┼─────────────────┼──────────────────┼────────┘
          │                 │                  │
          │                 ▼                  │
          │    ┌────────────────────────┐     │
          │    │  Tool Intent Mapper    │     │
          │    │  - classify_intent()   │     │
          │    │  - create_sequence()   │     │
          │    │  - extract_params()    │     │
          │    └────────────┬───────────┘     │
          │                 │                  │
          │                 ▼                  │
          │    ┌────────────────────────┐     │
          │    │   Prompt Templates     │     │
          │    │  - get_prompt()        │     │
          │    │  - get_system_prompt() │     │
          │    └────────────┬───────────┘     │
          │                 │                  │
          │                 │                  ▼
          │                 │     ┌────────────────────┐
          │                 │     │   Tool Registry    │
          │                 │     │  - get_tool()      │
          │                 │     │  - execute()       │
          │                 │     └────────┬───────────┘
          │                 │              │
          │                 │              ▼
          │                 │     ┌────────────────────┐
          │                 │     │  Individual Tools  │
          │                 │     │  - code_generation │
          │                 │     │  - document_gen    │
          │                 │     │  - email_automation│
          │                 │     └────────────────────┘
          │                 │
          ▼                 ▼
┌──────────────────────────────────────┐
│         LLM Integration              │
│  - generate_response()               │
│  - Uses optimized prompts            │
└──────────────────────────────────────┘
```

## Data Flow: Parameter Injection

```
Task 1: content_generation
┌─────────────────────────────────────┐
│ Input:                              │
│   topic: "Python"                   │
│                                     │
│ Processing:                         │
│   LLM generates content             │
│                                     │
│ Output:                             │
│   result: "Python is a high-level  │
│            programming language..." │
└──────────────┬──────────────────────┘
               │
               │ Stored in state.task_results["task_1"]
               │
               ▼
Task 2: document_generation
┌─────────────────────────────────────┐
│ Input (before resolution):          │
│   content: "{{ task_1.result }}"    │
│   format: "pdf"                     │
│                                     │
│ Parameter Resolution:               │
│   orchestrator._resolve_parameters()│
│   → Looks up task_1.result          │
│   → Injects actual content          │
│                                     │
│ Input (after resolution):           │
│   content: "Python is a high-level  │
│            programming language..." │
│   format: "pdf"                     │
│                                     │
│ Processing:                         │
│   document_generation tool creates  │
│   PDF with injected content         │
│                                     │
│ Output:                             │
│   result: "/path/to/python.pdf"     │
└──────────────┬──────────────────────┘
               │
               │ Stored in state.task_results["task_2"]
               │
               ▼
Task 3: email_automation
┌─────────────────────────────────────┐
│ Input (before resolution):          │
│   recipients: ["user@example.com"]  │
│   subject: "Python Document"        │
│   attachments: "{{ task_2.result }}"│
│                                     │
│ Parameter Resolution:               │
│   → Injects PDF path                │
│                                     │
│ Input (after resolution):           │
│   recipients: ["user@example.com"]  │
│   subject: "Python Document"        │
│   attachments: ["/path/to/python.pdf"]│
│                                     │
│ Processing:                         │
│   email_automation tool sends email │
│   with PDF attachment               │
│                                     │
│ Output:                             │
│   result: "Email sent successfully" │
└─────────────────────────────────────┘
```

## Intent Classification Flow

```
User Query: "Create a PDF about Python and email it to john@example.com"
                              │
                              ▼
┌─────────────────────────────────────────────────────────────┐
│              ToolIntentMapper.classify_intent()             │
│                                                             │
│  Step 1: Pattern Matching                                  │
│  ┌───────────────────────────────────────────────────────┐ │
│  │ Regex patterns:                                       │ │
│  │ ✓ "create.*pdf" → DOCUMENT_EXPORT (weight: 1.0)      │ │
│  │ ✓ "email.*to" → EMAIL_AUTOMATION (weight: 1.0)       │ │
│  └───────────────────────────────────────────────────────┘ │
│                                                             │
│  Step 2: Keyword Detection                                 │
│  ┌───────────────────────────────────────────────────────┐ │
│  │ Keywords found:                                       │ │
│  │ ✓ "create" → content_generation (weight: 0.5)        │ │
│  │ ✓ "pdf" → document_generation (weight: 0.5)          │ │
│  │ ✓ "email" → email_automation (weight: 0.5)           │ │
│  └───────────────────────────────────────────────────────┘ │
│                                                             │
│  Step 3: Score Calculation                                 │
│  ┌───────────────────────────────────────────────────────┐ │
│  │ Intent Scores:                                        │ │
│  │ - DOCUMENT_EXPORT: 1.5                                │ │
│  │ - EMAIL_AUTOMATION: 1.5                               │ │
│  │ - CONTENT_GENERATION: 0.5                             │ │
│  │                                                       │ │
│  │ Primary: DOCUMENT_EXPORT (highest score)              │ │
│  │ Secondary: EMAIL_AUTOMATION (score >= 50% of primary)│ │
│  │ Confidence: 0.92 (normalized score)                  │ │
│  └───────────────────────────────────────────────────────┘ │
│                                                             │
│  Step 4: Tool Sequence Generation                          │
│  ┌───────────────────────────────────────────────────────┐ │
│  │ Logic:                                                │ │
│  │ - DOCUMENT_EXPORT detected                            │ │
│  │ - No content provided → need content_generation       │ │
│  │ - EMAIL_AUTOMATION detected                           │ │
│  │ - "attach" implied → chain document to email          │ │
│  │                                                       │ │
│  │ Result:                                               │ │
│  │ 1. content_generation                                 │ │
│  │ 2. document_generation (content from #1)              │ │
│  │ 3. email_automation (attach from #2)                  │ │
│  └───────────────────────────────────────────────────────┘ │
│                                                             │
│  Step 5: Parameter Extraction                              │
│  ┌───────────────────────────────────────────────────────┐ │
│  │ Extracted:                                            │ │
│  │ - topic: "Python" (from "about Python")              │ │
│  │ - format: "pdf" (from "PDF")                          │ │
│  │ - recipients: ["john@example.com"] (regex match)     │ │
│  └───────────────────────────────────────────────────────┘ │
└─────────────────────────────────────────────────────────────┘
                              │
                              ▼
                    IntentMatch Object
┌─────────────────────────────────────────────────────────────┐
│ primary_intent: DOCUMENT_EXPORT                             │
│ secondary_intents: [EMAIL_AUTOMATION]                       │
│ confidence: 0.92                                            │
│ suggested_tools: ["content_generation", "document_generation",│
│                   "email_automation"]                       │
│ tool_sequence: [                                            │
│   ("content_generation", {"topic": "Python"}),              │
│   ("document_generation", {"format": "pdf"}),               │
│   ("email_automation", {"recipients": ["john@example.com"]})│
│ ]                                                           │
│ reasoning: "Document export requires content generation     │
│             first; Email with attachment requires document  │
│             generation"                                     │
└─────────────────────────────────────────────────────────────┘
```

## Prompt Template Usage Flow

```
Task: Code Generation
                │
                ▼
┌───────────────────────────────────────────────────────┐
│  Orchestrator._execute_code_generation_task()         │
│                                                       │
│  OLD WAY (inline prompt):                            │
│  ┌─────────────────────────────────────────────────┐ │
│  │ system_prompt = """You are an expert developer  │ │
│  │ Generate code..."""                             │ │
│  └─────────────────────────────────────────────────┘ │
│                                                       │
│  NEW WAY (template-based):                           │
│  ┌─────────────────────────────────────────────────┐ │
│  │ system_prompt = PromptTemplates.get_prompt(     │ │
│  │     PromptType.CODE_GENERATION,                 │ │
│  │     language="python",                          │ │
│  │     requirements="Create REST API",             │ │
│  │     style_guide="PEP 8"                         │ │
│  │ )                                               │ │
│  └─────────────────────────────────────────────────┘ │
└───────────────────────────┬───────────────────────────┘
                            │
                            ▼
┌───────────────────────────────────────────────────────┐
│  PromptTemplates.get_prompt()                         │
│                                                       │
│  1. Lookup template by type                          │
│  2. Substitute variables                             │
│  3. Return formatted prompt                          │
└───────────────────────────┬───────────────────────────┘
                            │
                            ▼
┌───────────────────────────────────────────────────────┐
│  Formatted Prompt (sent to LLM):                      │
│                                                       │
│  "You are an expert python developer. Generate       │
│   production-quality code.                           │
│                                                       │
│   MANDATORY REQUIREMENTS:                            │
│   1. Include comprehensive docstrings...             │
│   2. Follow python best practices and PEP 8...       │
│   3. Handle edge cases with try-except blocks...     │
│   ...                                                │
│                                                       │
│   REQUIREMENTS:                                      │
│   - Create REST API                                  │
│                                                       │
│   Generate ONLY the code with comments."             │
└───────────────────────────┬───────────────────────────┘
                            │
                            ▼
                    LLM generates code
                            │
                            ▼
                  High-quality, structured code
```

## Benefits Visualization

```
BEFORE Enhancements:
┌─────────────┐
│ User Query  │
└──────┬──────┘
       │
       ▼
┌──────────────────┐
│ Basic Pattern    │ → 75% accuracy
│ Matching         │
└──────┬───────────┘
       │
       ▼
┌──────────────────┐
│ Single Tool      │ → Limited capability
│ Selection        │
└──────┬───────────┘
       │
       ▼
┌──────────────────┐
│ Generic Prompt   │ → Inconsistent quality
└──────┬───────────┘
       │
       ▼
┌──────────────────┐
│ Basic Response   │ → 7/10 quality
└──────────────────┘

AFTER Enhancements:
┌─────────────┐
│ User Query  │
└──────┬──────┘
       │
       ▼
┌──────────────────┐
│ Enhanced Intent  │ → 90%+ accuracy
│ Classification   │ → Confidence scoring
│ (Tool Mapper)    │ → Multi-intent detection
└──────┬───────────┘
       │
       ▼
┌──────────────────┐
│ Tool Sequence    │ → Automatic chaining
│ Generation       │ → Parameter injection
│                  │ → Dependency management
└──────┬───────────┘
       │
       ▼
┌──────────────────┐
│ Optimized        │ → Directive prompts
│ Prompts          │ → Structured output
│ (Templates)      │ → Consistent quality
└──────┬───────────┘
       │
       ▼
┌──────────────────┐
│ High-Quality     │ → 8.5/10 quality
│ Response         │ → Better coherence
└──────────────────┘
```
