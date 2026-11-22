# Feature package contract

Each directory under this folder is an independently selectable feature package.

Required files:

- `README.md`: behavior, activation, ownership and scope limits.
- `feature.json`: base commit, standalone patched-tree SHA, dependencies and patch SHA256 values.
- `patches/`: ordered unified diffs against the pinned upstream source.
- `tests/test.json`: test targets, test overlays and test runner settings.
- `tests/README.md`: how to run the package tests.
- `validation/README.md`: evidence and outstanding qualification.

Test overlays are copied into a disposable prepared worktree only while tests run, then restored. Do not add test-only source files to production patches unless they are genuinely required by the feature at runtime.
