#!/usr/bin/env python3
"""Verify feature metadata, patch hashes, independent trees and the complete patch stack."""
from __future__ import annotations

import sys

from common import (
    ROOT, load_stack, load_feature, patch_items, resolve_features,
    sha256_file, tree_from_index, git,
)


def main():
    stack = load_stack()
    base = stack["base_commit"]
    if git(["cat-file", "-e", f"{base}^{{commit}}"], check=False).returncode:
        raise RuntimeError(
            f"base commit {base} is not in this clone; fetch upstream main before verification"
        )

    order = resolve_features(None, stack)
    all_patches = []
    for feature_id in order:
        feature_dir, meta = load_feature(feature_id, stack)
        if meta.get("base_commit") != base:
            raise RuntimeError(f"{feature_id}: base_commit differs from stack.json")
        patches = patch_items(feature_id, stack)
        paths = []
        for patch_id, path, expected_hash in patches:
            actual_hash = sha256_file(path)
            if actual_hash != expected_hash:
                raise RuntimeError(
                    f"{feature_id}/{patch_id}: SHA256 mismatch; expected {expected_hash}, got {actual_hash}"
                )
            paths.append(path)
            all_patches.append(path)
        actual_tree = tree_from_index(base, paths)
        expected_tree = meta.get("patched_tree")
        if not expected_tree or actual_tree != expected_tree:
            raise RuntimeError(
                f"{feature_id}: standalone tree mismatch; expected {expected_tree}, got {actual_tree}"
            )
        print(f"OK {feature_id}: {len(paths)} patch(es), tree {actual_tree}")

    actual_stack_tree = tree_from_index(base, all_patches)
    expected_stack_tree = stack.get("patched_tree")
    if not expected_stack_tree or actual_stack_tree != expected_stack_tree:
        raise RuntimeError(
            f"full stack tree mismatch; expected {expected_stack_tree}, got {actual_stack_tree}"
        )
    print(f"OK full stack: {len(all_patches)} patch(es), tree {actual_stack_tree}")


if __name__ == "__main__":
    try:
        main()
    except Exception as error:
        print(f"downstream verification failed: {error}", file=sys.stderr)
        raise SystemExit(1)
