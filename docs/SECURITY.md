# Security

Security rule number one: **authorization never depends on the model**. The model proposes; code outside the model decides. Every control below is enforced by the host-side code and covered by automated tests.

## Implemented controls

| Threat | Control | Where |
|--------|---------|-------|
| The model asks for something it should not do | Least-privilege permissions in `area:action` form, no wildcards; the caller (`Principal`) is immutable and created by the host, never by the model | `tools/permissions.py` |
| The model grants itself permissions through arguments | Unknown arguments are rejected before reaching any handler | `tools/schema.py` |
| Malformed or hostile arguments | Strict typed validation with limits | `tools/schema.py` |
| Destructive or sensitive actions | Tools can require human confirmation; without a confirmation function they do not run | `tools/executor.py` |
| Runaway tools | Per-tool timeout, output size cap, step limit in the agent loop | `tools/executor.py`, `tools/agent.py` |
| Injection through tool output (for example text inside a file or web page) | Tool results are returned as escaped data (`<` and `>`), and only model output is ever parsed for calls | `tools/protocol.py` |
| Forged conversation turns through user text | Whitespace and line breaks in every message are collapsed before it enters the prompt | `core/context/manager.py` |
| Arbitrary code through the calculator | AST evaluation of numbers and arithmetic only; no names, calls or attributes | `tools/builtin.py` |
| Reading files outside the authorized folder | Path checks (`..`, absolute, drive letters, backslashes), resolved-path containment (also blocks symbolic links) and an extension allow-list; identical error for every refusal | `tools/builtin.py` |
| Secrets in logs and errors | `secret` parameters and undeclared arguments are logged as `***`; handler exceptions are reported by type only; outputs are never stored | `tools/executor.py`, `tools/audit.py` |

## Not implemented yet

| Area | Status |
|------|--------|
| Database access (parameterized queries, read and write permissions, confirmation of writes) | Planned |
| Web research and defense against prompt injection inside web pages | Planned |
| Authentication of API callers, sessions and rate limiting | Planned |
| Persistent, tamper-evident audit storage and retention policy | Planned |
| Access control and deletion for the memory layers | Planned |
| Killing a tool that ignores its timeout | Not supported; handlers must finish on their own |
| Anything beyond the controls above | Not covered |

## Responsibilities of the host system

- Create `Principal` objects only from authenticated identities, with the minimum permissions required.
- Provide the confirmation function for sensitive tools and show the validated arguments to the human.
- Point `read_text_file` at a dedicated folder that holds no secrets.
- Never register a tool that executes operating-system commands or arbitrary SQL.

## Reporting

This project is under active development and has not been audited. Please open an issue in the repository for any suspected vulnerability.