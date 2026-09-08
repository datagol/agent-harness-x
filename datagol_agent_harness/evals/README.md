# DataGOL Agent Harness — Evaluation Framework Guide

The evaluation framework provides first-class benchmarking and automated testing for agents built with the **DataGOL Agent Harness**. Built on top of the **LangSmith evaluation framework** (`langsmith.evaluate` / `aevaluate`), it bridges the gap between generic function evaluation and complex multi-turn, stateful agent behaviors.

For in-depth diagrams and component breakdowns, see [eval_architecture.md](eval_architecture.md).

---

## Table of Contents

- [1. Architecture \& Design](#1-architecture--design)
- [2. Core Components](#2-core-components)
  - [AgentTarget (Target Adapter)](#agenttarget-target-adapter)
  - [Evaluation Runner](#evaluation-runner)
  - [Evaluation CLI](#evaluation-cli)
- [3. CLI Usage Guide](#3-cli-usage-guide)
  - [Basic Benchmark Execution](#basic-benchmark-execution)
  - [Filtering Evaluators (`--evaluator`)](#filtering-evaluators---evaluator)
  - [Live Upload to LangSmith](#live-upload-to-langsmith)
- [4. Evaluator Suite Reference](#4-evaluator-suite-reference)
- [5. Python API \& Custom Evaluations](#5-python-api--custom-evaluations)
  - [Evaluating an Agent](#evaluating-an-agent)
  - [Running Single or Custom Evaluators](#running-single-or-custom-evaluators)
  - [Writing a Custom Evaluator](#writing-a-custom-evaluator)
- [6. Built-in Benchmark Datasets](#6-built-in-benchmark-datasets)
  - [Syncing Datasets to LangSmith Cloud](#syncing-datasets-to-langsmith-cloud)
- [7. Offline CI/CD Testing](#7-offline-cicd-testing)

---

## 1. Architecture & Design

LangSmith evaluates generic functions `(inputs: dict) -> dict`. However, agents are stateful, execute recursive multi-step loops, call external tools, and consume token budgets. 

The harness evaluation framework wraps LangSmith with agent-native telemetry and scoring:

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                            EVALUATION WORKFLOW                              │
│                                                                             │
│   1. CLI / Script          python -m datagol_agent_harness.evals.cli        │
│                                            │                                │
│   2. Runner (runner.py)    Resolves dataset & orchestrates execution        │
│                                            │                                │
│   3. Target (target.py)    Runs Agent per example with lifecycle hooks      │
│                            • Records tool calls, args, errors, duration     │
│                            • Records skills invoked & guardrail stats       │
│                            • Traps errors (MaxIterationsError, API fails)   │
│                                            │                                │
│   4. Evaluators            Scores run outputs against example constraints   │
│      (evaluators/)         • Tool selection, args, trajectory, correctness  │
│                                            │                                │
│   5. Reporting             • Terminal Rich table with pass rates & latency  │
│                            • LangSmith Web UI experiment traces (if online) │
└─────────────────────────────────────────────────────────────────────────────┘
```

### Key Capabilities
- **Zero-Dependency Offline Mode:** Run deterministic evaluations locally without API keys or network calls (`--offline`).
- **Telemetry Hooks:** Captures full tool invocation trajectories (names, parameters, outputs, error flags, and millisecond latencies).
- **Error Containment:** If an agent encounters a runaway loop (`MaxIterationsError`) or tool failure, the error is recorded for scoring rather than crashing the benchmark run.
- **Span Nesting:** When connected to LangSmith, child spans (LLM generation, tool execution) nest under the parent evaluation example run in the LangSmith Web UI.

---

## 2. Core Components

### `AgentTarget` (`target.py`)
Adapts an `Agent` instance or `AgentFactory` (`(inputs) -> Agent`) to LangSmith's target interface.
- **Fresh State:** Recommended usage is an `AgentFactory` callable so that each test case executes against a pristine conversation memory.
- **Normalized Schema:** Outputs a standardized dictionary:
  ```python
  {
      "output": str,                    # Final text response
      "tool_calls": list[dict],         # Detailed tool executions (id, name, input, output, is_error, duration_ms)
      "tools_used": list[str],          # Unique list of tool names called
      "iterations": int,                # Total loop iterations
      "token_usage": dict,              # {"input_tokens", "output_tokens", "total_tokens"}
      "cost": str,                      # Estimated dollar cost
      "skills_invoked": list[str],      # Lazily loaded skills
      "error": str | None,              # Exception message if trapped
  }
  ```

### Evaluation Runner (`runner.py`)
Programmatic entrypoint (`evaluate_agent(...)`):
- Resolves benchmark datasets from local registry or LangSmith cloud.
- Executes evaluation runs via `langsmith.evaluate()`.
- Calculates pass rates and aggregate metrics across runs.
- Prints rich terminal summary tables and returns an `EvaluationSummary` object.

### Evaluation CLI (`cli.py`)
The command-line interface for running benchmarks against different providers (`anthropic`, `openai`) and models with configurable concurrency and evaluator filters.

---

## 3. CLI Usage Guide

### Basic Benchmark Execution

Run the tool-calling benchmark suite offline (zero cost, local execution):
```bash
python -m datagol_agent_harness.evals.cli --suite tool_calling --offline
```

Available suites:
- `tool_calling`: Calculator math, file system inspection, negative controls (tool avoidance).
- `skills`: Lazy skill loading and instruction compliance (`commit-message`, `code-review`, etc.).
- `multi_agent`: Specialist agent delegation and result synthesis.
- `guardrails`: Loop capping (`max_iterations`), cost boundaries, and permissions (`DENY`).
- `memory`: Multi-turn conversational recall and context retention.
- `all`: Executes all benchmark suites combined.

### Filtering Evaluators (`--evaluator` / `-e`)

You can restrict evaluation to run **only one or a subset of evaluators** using `--evaluator`:

```bash
# Run ONLY tool selection evaluation (did the agent select the expected tools?)
python -m datagol_agent_harness.evals.cli --suite tool_calling --evaluator tool_selection --offline

# Run ONLY text correctness evaluation
python -m datagol_agent_harness.evals.cli --suite tool_calling --evaluator contains --offline

# Specify multiple evaluators using comma-separated values or repeated flags
python -m datagol_agent_harness.evals.cli --suite tool_calling -e tool_selection -e tool_args --offline
python -m datagol_agent_harness.evals.cli --suite tool_calling -e "tool_selection,contains,iteration_budget" --offline
```

### Live Upload to LangSmith

When you have a LangSmith API key, upload the experiment to view the full interactive trace tree:

```bash
export LANGSMITH_API_KEY="lsv2_pt_..."
export ANTHROPIC_API_KEY="sk-ant-..."

# Run skills benchmark against Claude 3.5 Sonnet and upload results
python -m datagol_agent_harness.evals.cli --suite skills --model claude-sonnet-4-6 --upload

# Run with concurrency
python -m datagol_agent_harness.evals.cli --suite all --concurrency 2 --upload
```

When completed, the CLI displays a clickable URL directly to the LangSmith experiment dashboard:
```
┌────────────────────────────────────────────────────────┐
│ LangSmith Experiment: eval-skills-claude-sonnet-4-6    │
│ View Results: https://smith.langchain.com/o/.../eval   │
└────────────────────────────────────────────────────────┘
```

---

## 4. Evaluator Suite Reference

The framework includes 14 evaluators categorized into 5 behavioral pillars:

| Category | Evaluator Name | CLI Flag / Alias | Description |
|---|---|---|---|
| **Tools** | `tool_selection_evaluator` | `tool_selection`, `tools` | Verifies expected tools were invoked and forbidden tools avoided. |
| **Tools** | `tool_args_evaluator` | `tool_args`, `args` | Asserts tool input arguments match expected values or schemas. |
| **Tools** | `no_tool_errors_evaluator` | `no_tool_errors` | Verifies no tool executions resulted in `is_error=True`. |
| **Tools** | `tool_call_count_evaluator` | `tool_call_count`, `count` | Asserts number of tool calls falls within `[min_calls, max_calls]`. |
| **Correctness** | `contains_evaluator` | `contains` | Verifies required substrings exist in the final output text. |
| **Correctness** | `exact_match_evaluator` | `exact_match`, `exact` | Asserts exact text equality (with optional whitespace trimming). |
| **Correctness** | `regex_evaluator` | `regex` | Matches output against a regular expression pattern. |
| **Correctness** | `json_valid_evaluator` | `json_valid`, `json` | Asserts output contains valid JSON with required keys. |
| **Trajectory** | `iteration_budget_evaluator` | `iteration_budget`, `iterations` | Verifies iterations remained within `max_allowed_iterations`. |
| **Trajectory** | `step_sequence_evaluator` | `step_sequence`, `sequence` | Asserts tools were called in an exact ordered sequence. |
| **Trajectory** | `no_agent_errors_evaluator` | `no_agent_errors` | Asserts no uncaught exceptions or error terminations occurred. |
| **Skills** | `skill_invoked_evaluator` | `skill_invoked`, `skills` | Verifies required skills were lazily loaded into context. |
| **Delegation** | `subagent_delegated_evaluator`| `subagent_delegated`, `delegation`| Verifies orchestrator delegated to the correct specialist sub-agent. |
| **Model-Graded** | `LLMJudgeEvaluator` | `llm_judge`, `judge` | LLM-as-a-judge rubric evaluating accuracy, completeness, and conciseness. |

---

## 5. Python API & Custom Evaluations

### Evaluating an Agent

```python
from datagol_agent_harness import Agent, AgentConfig, PermissionLevel
from datagol_agent_harness.evals import evaluate_agent, build_example, default_evaluators

# 1. Define agent factory (clean instance per run)
def create_agent(inputs):
    agent = Agent(config=AgentConfig(model="claude-sonnet-4-6"))
    
    @agent.tools.register(permission=PermissionLevel.ALLOW)
    def calculate(expression: str) -> str:
        """Evaluate arithmetic expressions."""
        return str(eval(expression, {"__builtins__": None}, {}))
        
    return agent

# 2. Define test dataset
dataset = [
    build_example(
        inputs={"prompt": "What is 144 / 12?"},
        outputs={
            "expected_tools": ["calculate"],
            "contains_all": ["12"],
            "max_allowed_iterations": 3,
        },
    ),
]

# 3. Run evaluation
summary = evaluate_agent(
    agent=create_agent,
    dataset=dataset,
    evaluators=default_evaluators(),
    experiment_prefix="custom-calc-benchmark",
    offline=True,
)

print(f"Pass Rate: {summary.pass_rate * 100:.1f}%")
print(f"Scores: {summary.scores}")
```

### Running Single or Custom Evaluators

To run only specific evaluators, pass them directly in the `evaluators` list:

```python
from datagol_agent_harness.evals import (
    evaluate_agent,
    tool_selection_evaluator,
    contains_evaluator,
    get_evaluator,
)

# Option 1: Pass evaluator functions directly
summary = evaluate_agent(
    agent=create_agent,
    dataset="tool_calling",
    evaluators=[tool_selection_evaluator, contains_evaluator],
    offline=True,
)

# Option 2: Resolve by name helper
my_evals = [get_evaluator("tools"), get_evaluator("iterations")]
summary = evaluate_agent(
    agent=create_agent,
    dataset="tool_calling",
    evaluators=my_evals,
    offline=True,
)
```

### Writing a Custom Evaluator

An evaluator is any callable conforming to LangSmith's signature `(run: Run, example: Example) -> dict | EvaluationResult`:

```python
from langsmith.schemas import Run, Example

def latency_under_two_seconds_evaluator(run: Run, example: Example) -> dict:
    """Passes only if the agent finished in under 2000 milliseconds."""
    tool_calls = (run.outputs or {}).get("tool_calls", [])
    total_tool_time = sum(tc.get("duration_ms", 0) or 0 for tc in tool_calls)
    
    passed = total_tool_time < 2000
    return {
        "key": "fast_tool_execution",
        "score": 1.0 if passed else 0.0,
        "comment": f"Total tool execution duration: {total_tool_time:.1f}ms",
    }
```

Then supply it to `evaluate_agent(..., evaluators=[latency_under_two_seconds_evaluator])`.

---

## 6. Built-in Benchmark Datasets

The harness ships with pre-built golden test suites in `datasets/`:

1. **`tool_calling`**: Evaluates single-step and multi-step tool calls, argument precision, error recovery on missing files, and negative controls (tool avoidance on general factual queries).
2. **`skills`**: Evaluates matching prompts for `commit-message`, `code-review`, and `sql-explain`, verifying lazy invocation and format adherence.
3. **`multi_agent`**: Evaluates orchestrator delegation to specialist sub-agents and synthesis of specialist responses.
4. **`guardrails`**: Evaluates boundary enforcement: iteration caps on runaway loops, cost ceilings, and permission enforcement (`DENY`).
5. **`memory`**: Evaluates multi-turn context retention across turns.

### Syncing Datasets to LangSmith Cloud

To upload or synchronize local benchmark datasets to your LangSmith project:

```python
from datagol_agent_harness.evals import get_dataset, sync_dataset_to_langsmith

dataset_examples = get_dataset("tool_calling")
dataset_id = sync_dataset_to_langsmith("datagol-tool-calling-benchmark", dataset_examples)
print("Uploaded dataset ID:", dataset_id)
```

---

## 7. Offline CI/CD Testing

All evaluation tests run offline without API keys or cloud connections. The harness includes offline test suites with mock providers in `tests/test_evals.py`:

```bash
# Run the complete evaluation unit test suite
pytest tests/test_evals.py
```

This verifies:
- `AgentTarget` telemetry capture and error containment.
- Every evaluator with passing and failing conditions.
- Dataset schemas and loader utilities.
- Complete end-to-end local evaluation runs without external API dependencies.
