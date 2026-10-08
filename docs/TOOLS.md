# Tool system

Lyra reaches the outside world only through tools. A tool is a named function with a declared schema, declared permissions and declared effects. The model never calls code directly: it asks for a tool, and the host system validates, authorizes, runs and records the call.

```
model output (untrusted text)
        |
   parse_model_output        extracts one <tool_call> JSON object
        |
   ToolExecutor
        |-- registry lookup           unknown tool           -> not_found
        |-- permission check          missing permission     -> denied
        |-- argument validation       bad or extra arguments -> invalid
        |-- human confirmation        sensitive tools only   -> needs_confirmation / rejected
        |-- run with timeout          handler failure        -> error / timeout
        |-- output size limit
        |-- audit log entry
        |
   render_tool_result        escaped text returned to the model as data
```

## Call format

The model requests a tool by writing:

```
<tool_call>{"name": "calculate", "arguments": {"expression": "(12 + 8) * 3"}}</tool_call>
```

Only `name` and `arguments` are accepted; any other field is rejected. Only the first call in a response is considered. Results return inside `<tool_result name="..." status="...">` with `<` and `>` escaped, so text produced by a tool cannot forge a new call.

## Defining a tool

Every tool must declare what it reads (`data_access`) and what it does (`actions`). Tools are registered by the host system, so the core never contains product-specific code.

```python
from tools.registry import ToolRegistry
from tools.schema import ParamSpec, Tool

registry = ToolRegistry()
registry.register(Tool(
    name="buscar_cliente",
    description="Busca um cliente no sistema.",
    handler=buscar_cliente,                      # a function provided by the host
    data_access="customer table (read only)",
    actions="none",
    parameters=(ParamSpec("nome", "string", max_length=100),),
    required_permissions=frozenset({"customers:read"}),
    timeout_seconds=3.0,
))
```

| Field | Meaning |
|-------|---------|
| `name` | Lowercase letters, digits and underscores |
| `parameters` | Typed `ParamSpec` entries: `string`, `integer`, `number`, `boolean`; optional enum, min, max, max length, default, `secret` |
| `required_permissions` | Permissions in `area:action` form. No wildcards |
| `sensitive` | If true, a human must confirm before every execution |
| `timeout_seconds` | Execution time limit |
| `data_access`, `actions` | Mandatory plain-language declaration of what the tool touches and does |

## Validation rules

Unknown parameters are rejected, required parameters must be present, types are strict (a boolean is not an integer, `NaN` and infinity are not numbers), and enum, range and length limits apply. Only validated arguments reach the handler.

## Result statuses

| Status | Meaning |
|--------|---------|
| `ok` | Executed |
| `error` | The handler failed; only the error type is reported, never its message |
| `denied` | The caller lacks a required permission |
| `invalid` | Arguments failed validation |
| `not_found` | No such tool |
| `needs_confirmation` | Sensitive tool and no confirmation function available; not executed |
| `rejected` | The human declined; not executed |
| `timeout` | Exceeded the time limit |

## Agent loop

`ToolAgent` runs: model responds, the call is parsed, the executor runs it, the result is added to the history as a `tool` message, and the model responds again. It stops when the model answers without a call, when `max_steps` is reached (the extra call is not executed), or when a sensitive tool needs confirmation. Malformed calls are returned to the model as a protocol error and count toward the step limit.

## Built-in tools

| Tool | Permission | Description |
|------|------------|-------------|
| `calculate` | `math:use` | Arithmetic with `+ - * / // % **` on numbers only. Parsed with the `ast` module, never `eval`. Limits on expression length, exponent size and result size |
| `get_current_time` | `time:read` | Current UTC time |
| `read_text_file` | `files:read` | Reads `.txt`, `.md`, `.json` and `.csv` files inside one authorized folder, up to a size limit. Rejects `..`, absolute paths, drive letters and symbolic links that leave the folder. All refusals return the same message |

## Audit log

One JSON object per call: timestamp, call id, tool, user, status, duration, arguments, error and output size. Parameters marked `secret`, and any argument not declared by the tool, are logged as `***`. Outputs are not stored, only their size. The log is kept in memory (last 1,000 entries) and can also be appended to a JSONL file.

## Limitations

- The current base model cannot yet request tools: its 128-token context cannot hold tool descriptions and it was never trained on tool calls. The layer is tested with a simulated model (`scripts/tools_demo.py`). Teaching the model to emit valid calls is planned work.
- A tool that exceeds its timeout is abandoned, not killed. Its thread keeps running until it finishes, so handlers must be written to finish.
- The audit duration of a sensitive tool includes the time the human took to confirm.
- A denied call and an unknown tool return different statuses, which can reveal which tools exist. Acceptable for internal systems.
- There is no database tool yet.