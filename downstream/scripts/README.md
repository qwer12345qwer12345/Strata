# Downstream scripts

All commands are run from the repository root.

- `verify.py`: validate package metadata, patch SHA256 values, each standalone feature tree and the complete stack tree using temporary Git indexes.
- `prepare.py`: verify the stack, create a detached worktree at the pinned upstream commit, apply selected features and regenerate generated Settings files.
- `apply.py`: apply selected patches to an existing clean worktree at the pinned commit; it checks the base, patch hashes, applicability and resulting tree.
- `test.py`: verify that the selected feature patch is present, temporarily overlay the package test sources, configure/build declared targets, run Python/CTest checks and restore the original source files.

Typical sequence:

```bash
python downstream/scripts/verify.py
python downstream/scripts/prepare.py --feature l2-session
python downstream/scripts/test.py --feature l2-session \
  --source ../Strata-prepared --build ../Strata-build-l2
```

The scripts never reset or delete an existing worktree. `prepare.py` stops if the target path already exists. Remove a disposable worktree explicitly with `git worktree remove <path>` only after checking that it contains no user work.

The current CI checks script syntax and package/patch/tree integrity. It does not compile the C++/CUDA feature targets or replace hardware qualification.
