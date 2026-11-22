# Chat Completions regression tests

The suite replays synthetic output through the real loopback `/v1/chat/completions` handler. It covers generated role tokens, incomplete JSON at a role boundary or token limit, complete and undeclared calls, literal markers in file contents, requests without tools, nonstream responses, and the disabled guard's role text and incremental streaming. It also runs the eight existing `UnfinishedToolCall` fixture regressions.

Run with the feature package test command in the package README. The suite needs the Terminal's Python dependencies and loads no model.
