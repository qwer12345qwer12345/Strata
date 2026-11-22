# ChatML tool boundary guard

This package ports the deployed Chat Completions guard to Strata 0.2.19 / Engine 0.1.65, pinned at `c6f6427d9de9016fd92d13e29740f65625261df7`.

Enable it in the server profile using `"chatml_tool_guard": true`. The default is false. The value must be a JSON boolean.

For a request with tools, a generated ChatML role control token ends the assistant turn before that token enters the text parser. The parser first finalizes any complete tool call held for lookahead. A call interrupted while incomplete ends as `length`. Chat Completions streams publish each tool call after its complete arguments have been parsed. This delays the tool's arguments until completion. Completed undeclared calls still follow the existing `undeclared_tool_calls` policy.

A literal marker inside file contents is preserved when it is ordinary text tokens. Requests with no tools keep their existing text behavior. The role-token boundary applies to ChatML and excludes the DeepSeek family. The default tool streaming path retains incremental argument chunks.

The production patch changes `terminal/serve/server.py`. Tests live in the package overlay. This package adds no engine code or model assets.

```sh
python downstream/scripts/prepare.py --feature chatml-tool-guard --target ../Strata-tool-guard
python downstream/scripts/test.py --feature chatml-tool-guard --source ../Strata-tool-guard
```

Use a Python environment with the Terminal's runtime dependencies. The test runner runs the Python suite directly for this package. See [validation](validation/README.md) for results.
