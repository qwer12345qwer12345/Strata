#!/usr/bin/env python3
"""Create a disposable upstream worktree and prepare it with selected feature packages."""
from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

from common import ROOT, DOWNSTREAM, load_stack, resolve_features, generate_settings, git


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--feature", action="append", default=[])
    parser.add_argument("--features", help="comma-separated feature IDs")
    parser.add_argument("--all", action="store_true")
    parser.add_argument("--target", help="worktree path; defaults to a sibling Strata-prepared directory")
    parser.add_argument("--no-settings", action="store_true", help="do not regenerate Settings outputs")
    args = parser.parse_args()

    stack = load_stack()
    requested = list(args.feature)
    if args.features:
        requested.extend(x.strip() for x in args.features.split(",") if x.strip())
    if args.all or not requested:
        requested = None
    selected = resolve_features(requested, stack)
    base = stack["base_commit"]

    if git(["cat-file", "-e", f"{base}^{{commit}}"], check=False).returncode:
        git(["fetch", "--no-tags", stack["upstream"]["url"], base])

    # Verify all package hashes and tree fingerprints before creating or changing a worktree.
    verify_script = DOWNSTREAM / "scripts" / "verify.py"
    subprocess.run([sys.executable, str(verify_script)], cwd=ROOT, check=True)

    target = Path(args.target).expanduser().resolve() if args.target else ROOT.parent / f"{ROOT.name}-prepared"
    if target.exists():
        raise RuntimeError(
            f"target already exists: {target}. Inspect it and remove the worktree explicitly before retrying; "
            "prepare.py never deletes or resets an existing directory."
        )

    git(["worktree", "add", "--detach", target, base])
    apply_script = DOWNSTREAM / "scripts" / "apply.py"
    command = [sys.executable, str(apply_script), "--source", str(target), "--features", ",".join(selected)]
    subprocess.run(command, cwd=ROOT, check=True)

    if not args.no_settings:
        generator = target / "tools" / "settings_register.py"
        if generator.is_file():
            generate_settings(target, stack)

    print(f"ready: {target}")
    print("This is a disposable prepared worktree; downstream patches remain in the management branch.")


if __name__ == "__main__":
    try:
        main()
    except Exception as error:
        print(f"preparation failed: {error}", file=sys.stderr)
        raise SystemExit(1)
