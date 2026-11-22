# Downstream scripts

- `verify.py`: validate package metadata, SHA256 values, each standalone feature tree and the complete stack tree using temporary Git indexes.
- `prepare.py`: create a detached worktree at the pinned upstream commit, apply selected features and regenerate generated Settings files.
- `apply.py`: apply selected patches to an existing clean worktree at the pinned commit.
- `test.py`: temporarily overlay a package's test sources, configure/build the declared targets, run Python/CTest checks and restore the original source files.

The scripts never reset or delete an existing worktree. Remove a disposable worktree explicitly with `git worktree remove <path>` after checking it contains no user work.
