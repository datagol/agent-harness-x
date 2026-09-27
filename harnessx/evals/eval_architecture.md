# HarnessX — Evaluation Architecture & Visual Flows

This document details the architectural design, component interactions, and execution lifecycles of the **HarnessX Evaluation Framework**.

---

## 1. System Architecture & Visual Relationships

```
┌──────────────────────────────────────────────────────────────────────────────┐
│                               USER INTERFACES                                │
│                                                                              │
│    CLI (cli.py)                              Python Script (e.g. run_evals.py)│
│    python -m harnessx.evals.cli    evaluate_agent(...)          │
└───────────────────────┬──────────────────────────────────────┬───────────────┘
                        │                                      │
                        ▼                                      ▼
┌──────────────────────────────────────────────────────────────────────────────┐
│                                RUNNER (runner.py)                            │
│  Orchestrates the evaluation experiment:                                     │
│  1. Resolves dataset ('tool_calling', 'skills', etc.)                        │
│  2. Creates AgentTarget                                                      │
│  3. Invokes LangSmith evaluate() / local offline runner                     │
│  4. Aggregates row-level metrics & prints Rich summary table                 │
└──────────────┬───────────────────────────────┬───────────────────────────────┘
               │                               │
               ▼ (runs for each Example)       ▼ (scores each Run)
┌──────────────────────────────┐   ┌───────────────────────────────────────────┐
│     TARGET (target.py)       │   │           EVALUATORS SUITE                │
│                              │   │                                           │
│  Adapts Agent -> LangSmith:  │   │  • Tools: tool_selection, tool_args,      │
│  • Instantiates fresh Agent  │   │           no_tool_errors, call_count      │
│  • Injects telemetry hooks   │   │  • Correctness: contains, exact_match,    │
│  • Runs agent.run(prompt)    │   │                 regex, json_valid         │
│  • Captures output, tools,   │   │  • Trajectory: iteration_budget,          │
│    latency, tokens & errors  │   │                no_agent_errors, step_seq  │
│  • Returns normalized dict   │   │  • Routing: skill_invoked, subagent_del   │
└──────────────┬───────────────┘   │  • Qualitative: LLMJudgeEvaluator         │
               │                   └─────────────────────┬─────────────────────┘
               │                                         │
               └───────────────────┬─────────────────────┘
                                   ▼
┌──────────────────────────────────────────────────────────────────────────────┐
│                               OUTPUT / SCORES                                │
│  • Terminal: Rich Table with Pass Rate, Iterations, Cost, per-metric %       │
│  • LangSmith Web UI (if online): Interactive traces & prompt comparisons     │
└──────────────────────────────────────────────────────────────────────────────┘
```

---

## 2. Core Components: `cli.py`, `target.py`, and `runner.py`

### 2.1 `target.py` (`AgentTarget`)
**Role:** The bridge between LangSmith and your `Agent`.

* **The Problem:** LangSmith expects an evaluation target function with a simple signature: `async (inputs: dict) -> dict`. An agent, however, maintains state, executes multi-turn loops, dispatches tools, and accumulates token usage.
* **What `AgentTarget` does:**
  1. **Clean State per Run:** Accepts an `AgentFactory` (`(inputs) -> Agent`) so each test case starts with a pristine agent instance, preventing memory and context leaks between examples.
  2. **Telemetry Capture via Lifecycle Hooks:**
     - `TOOL_CALL_START` & `TOOL_CALL_END`: Measures execution duration per tool, input parameters, return values, and whether `is_error=True`.
     - `SKILL_INVOKED`: Detects which lazy skills were pulled into prompt context.
  3. **Parent Run Tree Propagation:** Inspects `get_current_run_tree()` so agent lifecycle spans created by `LangSmithExtension` automatically nest under the parent evaluation example run in the LangSmith Web UI.
  4. **Error Containment:** Catches runtime exceptions (such as `MaxIterationsError`, `CostLimitError`, or LLM API timeouts) so the benchmark records a clean failure instead of crashing the process.
  5. **Normalized Output Schema:** Converts the rich execution trajectory into a standardized dictionary:
     ```python
     {
         "output": str,                   # Final response text
         "tool_calls": list[dict],        # List of {id, name, input, output, is_error, duration_ms}
         "tools_used": list[str],         # Distinct list of tool names invoked
         "iterations": int,               # Total loop steps taken
         "token_usage": dict,             # {"input_tokens", "output_tokens", "total_tokens"}
         "cost": str,                     # Dollar estimate (e.g., "$0.0024")
         "skills_invoked": list[str],     # Names of skills triggered
         "error": str | None,             # Trapped exception message (if any)
     }
     ```

---

### 2.2 `runner.py` (`evaluate_agent`)
**Role:** The programmatic orchestration engine.

* **Dataset Resolution:** Accepts dataset names (`"tool_calling"`, `"skills"`, `"all"`), lists of local `Example` objects, or remote LangSmith cloud dataset names.
* **Target Wrapping:** Automatically wraps the agent or agent factory in an `AgentTarget` instance, attaching the `LangSmithExtension` when uploading traces.
* **Execution Management:** Dispatches to `langsmith.evaluate()` with configurable concurrency (`max_concurrency`) and offline execution (`upload_results=False` / `offline=True`).
* **Metric Aggregation:** Aggregates row-level evaluator scores into overall averages:
  - Overall experiment pass rate (0.0 to 1.0).
  - Per-evaluator scores (e.g. `tool_selection`, `text_contains`, `iteration_budget`).
  - Average iterations and cost metrics.
* **Terminal Reporting:** Formats a Rich terminal summary table displaying test case counts, pass rates, and clickable URLs to the LangSmith dashboard.

---

### 2.3 `cli.py` (`main()`)
**Role:** The command-line interface.

* **Argument Parsing:**
  - `--suite` / `-s`: Benchmark suite (`tool_calling`, `skills`, `multi_agent`, `guardrails`, `memory`, `all`).
  - `--model` / `-m`: Model id (e.g. `claude-sonnet-4-6`, `gpt-5`).
  - `--provider` / `-p`: LLM provider (`anthropic`, `openai`).
  - `--offline`: Runs locally without uploading traces to LangSmith cloud.
  - `--upload`: Explicitly pushes traces and scores to LangSmith.
  - `--evaluator` / `-e`: Runs only specific evaluator(s) by name or alias (e.g. `-e tool_selection -e contains`).
  - `--concurrency` / `-c`: Worker concurrency limit.
  - `--judge`: Enables model-graded LLM-as-a-judge scoring.
* **Agent Factory Generation:** Builds a standard evaluation agent configured with arithmetic tools (`calculate`), safe file reading (`read_file`), and specialist delegation (`delegate_research`, `delegate_code_review`).
* **Execution & Exit Codes:** Executes the evaluation via `evaluate_agent()` and exits with code `0` on success or code `1` if pass rate falls below threshold.

---

## 3. Dataflow per Evaluation Turn

```
                       1. Example from Dataset
               inputs: {"prompt": "What is 144 / 12?"}
               outputs: {
                   "expected_tools": ["calculate"],
                   "contains_all": ["12"],
                   "max_allowed_iterations": 3
               }
                                  │
                                  ▼
                      ┌───────────────────────┐
                      │      AgentTarget      │
                      │                       │
                      │ 1. Start turn timer   │
                      │ 2. Hook tool telemetry│
                      │ 3. agent.run(prompt)  │
                      └───────────┬───────────┘
                                  │
                                  ▼
                            2. Run Output
               outputs: {
                   "output": "12",
                   "tools_used": ["calculate"],
                   "tool_calls": [{"name": "calculate", ...}],
                   "iterations": 1,
                   ...
               }
                                  │
         ┌────────────────────────┴────────────────────────┐
         ▼                                                 ▼
┌──────────────────────────────┐          ┌──────────────────────────────┐
│   tool_selection_evaluator   │          │      contains_evaluator      │
│                              │          │                              │
│ Checks if "calculate" is in  │          │ Checks if "12" is in         │
│ outputs["tools_used"].       │          │ outputs["output"].           │
│                              │          │                              │
│ Score: 1.0 (PASS)            │          │ Score: 1.0 (PASS)            │
└──────────────┬───────────────┘          └──────────────┬───────────────┘
               │                                         │
               └────────────────────┬────────────────────┘
                                    ▼
                         3. Evaluation Result
             Scores attached to Run & aggregated into Summary
```

### Detailed Lifecycle of One Turn:
1. **Fetch Example:** The runner pulls the next `Example` from the dataset containing the `inputs` (prompt) and expected `outputs` constraints.
2. **Target Dispatch:** `AgentTarget` invokes the agent factory to obtain a fresh `Agent` instance.
3. **Telemetry Registration:** Hooks are registered on `TOOL_CALL_START`, `TOOL_CALL_END`, and `SKILL_INVOKED` to capture precise runtime metrics.
4. **Agent Execution:** The agent executes its multi-step reasoning loop (calling tools, executing models).
5. **Output Normalization:** The target constructs the normalized output dictionary with final answer, tool call traces, and token usage.
6. **Evaluator Scoring:** Each registered evaluator receives the `(Run, Example)` pair and outputs an `EvaluationResult` or dictionary with a numeric score (`1.0` or `0.0`) and explanatory feedback.
7. **Trace Upload:** If connected to LangSmith (`upload_results=True`), the run, child spans, and feedback scores are committed to the experiment trace tree in the cloud.
