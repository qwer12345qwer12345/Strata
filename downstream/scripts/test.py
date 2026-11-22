#!/usr/bin/env python3
"""Run a feature package's tests against a prepared source worktree."""
from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

from common import load_stack, load_feature, patch_items, sha256_file, generate_settings, git


def run(command, cwd):
    print("+", " ".join(str(x) for x in command), flush=True)
    subprocess.run([str(x) for x in command], cwd=cwd, check=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--feature", required=True)
    parser.add_argument("--source", required=True)
    parser.add_argument("--build", help="CMake build directory")
    parser.add_argument("--with-cuda", action="store_true", help="also build and run optional CUDA/VMM tests")
    parser.add_argument("--cmake-arg", action="append", default=[], help="extra CMake argument; may be repeated")
    parser.add_argument("--skip-configure", action="store_true")
    args = parser.parse_args()

    stack = load_stack()
    feature_dir, meta = load_feature(args.feature, stack)
    source = Path(args.source).expanduser().resolve()
    if not source.is_dir():
        raise RuntimeError(f"prepared source directory does not exist: {source}")
    base = stack["base_commit"]
    if git(["rev-parse", "HEAD"], cwd=source).stdout.strip() != base:
        raise RuntimeError(f"source HEAD is not the pinned base commit {base}")

    # Inspect the feature in a temporary index. Generated settings can differ after prepare.py.
    # A failed presence check must leave the prepared runtime source intact.
    patches = []
    for patch_id, patch, expected_hash in patch_items(args.feature, stack):
        if sha256_file(patch) != expected_hash:
            raise RuntimeError(f"{args.feature}/{patch_id}: patch SHA256 mismatch")
        patches.append(patch)
    with tempfile.TemporaryDirectory(prefix="strata-feature-presence-") as temp:
        env = dict(os.environ, GIT_INDEX_FILE=str(Path(temp) / "index"))
        git(["read-tree", base], cwd=source, env=env)
        git(["add", "-A"], cwd=source, env=env)
        before = git(["write-tree"], cwd=source, env=env).stdout.strip()
        options = ["--cached", "--whitespace=nowarn",
                   "--exclude=terminal/data/settings-register.json", "--exclude=docs/SETTINGS.md"]
        for patch in reversed(patches):
            git(["apply", "--reverse", *options, patch], cwd=source, env=env)
        for patch in patches:
            git(["apply", *options, patch], cwd=source, env=env)
        if git(["write-tree"], cwd=source, env=env).stdout.strip() != before:
            raise RuntimeError("feature presence check did not restore its temporary tree")

    test_config_path = feature_dir / meta["tests"]
    config = json.loads(test_config_path.read_text(encoding="utf-8"))
    test_dir = test_config_path.parent
    build = Path(args.build).expanduser().resolve() if args.build else source.parent / f"{source.name}-build-{args.feature}"

    overlays = config.get("overlays", [])
    generated = ["docs/SETTINGS.md", "terminal/data/settings-register.json"] if config.get("settings_generation") else []
    restore = {}
    with tempfile.TemporaryDirectory(prefix="strata-feature-test-overlay-") as temp:
        backup_root = Path(temp)
        try:
            for index, relative in enumerate(overlays + generated):
                if relative.startswith("overlay/"):
                    package_file = test_dir / relative
                    destination_relative = relative[len("overlay/"):]
                else:
                    package_file = None
                    destination_relative = relative
                destination = source / destination_relative
                destination.parent.mkdir(parents=True, exist_ok=True)
                if destination.exists():
                    backup = backup_root / str(index)
                    backup.parent.mkdir(parents=True, exist_ok=True)
                    shutil.copy2(destination, backup)
                    restore[destination] = backup
                else:
                    restore[destination] = None
                if package_file is not None:
                    if not package_file.is_file():
                        raise RuntimeError(f"missing test overlay: {package_file}")
                    shutil.copy2(package_file, destination)

            if config.get("settings_generation"):
                generate_settings(source, stack)

            for relative in config.get("python_tests", []):
                run([sys.executable, source / relative], source)
            if config.get("python_modules"):
                run([sys.executable, "-m", "unittest", *config["python_modules"]], source / "terminal")

            targets = list(config.get("targets", []))
            if args.with_cuda:
                targets.extend(config.get("optional_targets", {}).get("cuda", []))
            regex = config.get("ctest_regex", "")
            if args.with_cuda and config.get("optional_ctest_regex", {}).get("cuda"):
                extra = config["optional_ctest_regex"]["cuda"]
                regex = f"({regex})|({extra})" if regex else extra

            if not args.skip_configure and (targets or regex):
                cmake_args = list(config.get("cmake_args", []))
                if args.with_cuda and "-DSTRATA_ENABLE_CUDA=ON" not in cmake_args:
                    cmake_args.append("-DSTRATA_ENABLE_CUDA=ON")
                cmake_args.extend(args.cmake_arg)
                run(["cmake", "-S", source / "engine", "-B", build, *cmake_args], source)

            if targets:
                run(["cmake", "--build", build, "--parallel", "--target", *targets], source)

            if regex:
                run(["ctest", "--test-dir", build, "--output-on-failure", "--no-tests=error", "-R", regex], source)
        finally:
            for destination, backup in restore.items():
                if backup is None:
                    destination.unlink(missing_ok=True)
                elif backup.exists():
                    shutil.copy2(backup, destination)

    print(f"tests completed: {args.feature}")


if __name__ == "__main__":
    try:
        main()
    except Exception as error:
        print(f"feature tests failed: {error}", file=sys.stderr)
        raise SystemExit(1)
