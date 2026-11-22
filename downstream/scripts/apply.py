#!/usr/bin/env python3
"""Apply selected feature packages to a clean checkout of the pinned upstream commit."""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

from common import (
    load_stack, load_feature, resolve_features, patch_items,
    sha256_file, tree_from_worktree, git,
)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", required=True, help="clean worktree at stack.json base_commit")
    parser.add_argument("--feature", action="append", default=[])
    parser.add_argument("--features", help="comma-separated feature IDs")
    parser.add_argument("--all", action="store_true")
    args = parser.parse_args()

    stack = load_stack()
    requested = list(args.feature)
    if args.features:
        requested.extend(x.strip() for x in args.features.split(",") if x.strip())
    if args.all or not requested:
        requested = None
    selected = resolve_features(requested, stack)
    source = Path(args.source).expanduser().resolve()
    if not source.is_dir():
        raise RuntimeError(f"source directory does not exist: {source}")

    base = stack["base_commit"]
    head = git(["rev-parse", "HEAD"], cwd=source).stdout.strip()
    if head != base:
        raise RuntimeError(f"source HEAD must be {base}, got {head}")
    dirty = git(["status", "--porcelain", "--untracked-files=all"], cwd=source).stdout.strip()
    if dirty:
        raise RuntimeError("source worktree is not clean; refusing to overwrite existing changes")

    for feature_id in selected:
        _, meta = load_feature(feature_id, stack)
        if meta.get("base_commit") != base:
            raise RuntimeError(f"{feature_id}: base_commit differs from stack.json")
        for patch_id, patch, expected_hash in patch_items(feature_id, stack):
            actual_hash = sha256_file(patch)
            if actual_hash != expected_hash:
                raise RuntimeError(
                    f"{feature_id}/{patch_id}: SHA256 mismatch; expected {expected_hash}, got {actual_hash}"
                )
            git(["apply", "--check", "--whitespace=nowarn", patch], cwd=source)
            git(["apply", "--whitespace=nowarn", patch], cwd=source)
            print(f"applied {feature_id}/{patch_id}")

    actual_tree = tree_from_worktree(source, base)
    expected = None
    if selected == stack["feature_order"]:
        expected = stack.get("patched_tree")
    elif len(selected) == 1:
        _, meta = load_feature(selected[0], stack)
        expected = meta.get("patched_tree")
    if expected and actual_tree != expected:
        raise RuntimeError(f"prepared tree mismatch; expected {expected}, got {actual_tree}")
    print(f"prepared tree: {actual_tree}")


if __name__ == "__main__":
    try:
        main()
    except Exception as error:
        print(f"patch application failed: {error}", file=sys.stderr)
        raise SystemExit(1)
