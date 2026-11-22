#!/usr/bin/env python3
"""Shared helpers for Strata downstream feature packages."""
from __future__ import annotations

import hashlib
import json
import os
import subprocess
import sys
import tempfile
from pathlib import Path
from typing import Iterable

ROOT = Path(__file__).resolve().parents[2]
DOWNSTREAM = ROOT / "downstream"


def git(args: Iterable[str], cwd: Path = ROOT, env=None, check: bool = True):
    command = ["git", "-C", str(cwd), *[str(x) for x in args]]
    result = subprocess.run(command, capture_output=True, text=True, env=env)
    if check and result.returncode:
        detail = result.stderr.strip() or result.stdout.strip()
        raise RuntimeError(f"command failed ({result.returncode}): {' '.join(command)}\n{detail}")
    return result


def load_json(path: Path):
    return json.loads(path.read_text(encoding="utf-8"))


def load_stack():
    return load_json(DOWNSTREAM / "stack.json")


def feature_paths(stack=None):
    stack = stack or load_stack()
    return {item["id"]: DOWNSTREAM / item["path"] for item in stack["features"]}


def load_feature(feature_id: str, stack=None):
    paths = feature_paths(stack)
    if feature_id not in paths:
        raise RuntimeError(f"unknown feature: {feature_id}")
    path = paths[feature_id]
    meta = load_json(path)
    if meta.get("id") != feature_id:
        raise RuntimeError(f"{path}: feature id does not match registry")
    return path.parent, meta


def resolve_features(requested=None, stack=None):
    stack = stack or load_stack()
    order = stack["feature_order"]
    wanted = set(order if not requested else requested)
    unknown = wanted.difference(order)
    if unknown:
        raise RuntimeError("unknown feature(s): " + ", ".join(sorted(unknown)))

    def add_dependencies(feature_id):
        _, meta = load_feature(feature_id, stack)
        for dependency in meta.get("requires", []):
            if dependency not in order:
                raise RuntimeError(f"{feature_id}: unknown dependency {dependency}")
            if dependency not in wanted:
                wanted.add(dependency)
                add_dependencies(dependency)

    for feature_id in list(wanted):
        add_dependencies(feature_id)
    return [feature_id for feature_id in order if feature_id in wanted]


def patch_items(feature_id: str, stack=None):
    feature_dir, meta = load_feature(feature_id, stack)
    result = []
    seen = set()
    for item in meta.get("patches", []):
        patch_id = item.get("id")
        relative = Path(item["path"])
        if not patch_id or patch_id in seen:
            raise RuntimeError(f"{feature_id}: duplicate or invalid patch id {patch_id!r}")
        if relative.is_absolute() or ".." in relative.parts:
            raise RuntimeError(f"{feature_id}/{patch_id}: invalid patch path")
        path = (feature_dir / relative).resolve()
        if not path.is_relative_to(feature_dir.resolve()) or not path.is_file():
            raise RuntimeError(f"{feature_id}/{patch_id}: patch file is missing or escapes its package")
        seen.add(patch_id)
        result.append((patch_id, path, item["sha256"]))
    if not result:
        raise RuntimeError(f"{feature_id}: package has no patches")
    return result


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def tree_from_index(base_commit: str, patches):
    """Apply patches to a temporary Git index and return its tree SHA."""
    with tempfile.TemporaryDirectory(prefix="strata-downstream-index-") as temp:
        index = Path(temp) / "index"
        env = dict(os.environ, GIT_INDEX_FILE=str(index))
        git(["read-tree", base_commit], env=env)
        for patch in patches:
            git(["apply", "--cached", "--whitespace=nowarn", patch], env=env)
        return git(["write-tree"], env=env).stdout.strip()


def tree_from_worktree(source: Path, base_commit: str):
    """Compute a worktree tree without changing the developer's real index."""
    with tempfile.TemporaryDirectory(prefix="strata-downstream-worktree-") as temp:
        index = Path(temp) / "index"
        env = dict(os.environ, GIT_INDEX_FILE=str(index))
        git(["read-tree", base_commit], cwd=source, env=env)
        git(["add", "-A"], cwd=source, env=env)
        return git(["write-tree"], cwd=source, env=env).stdout.strip()


def all_patch_paths(feature_ids, stack=None):
    result = []
    for feature_id in feature_ids:
        result.extend(path for _, path, _ in patch_items(feature_id, stack))
    return result


def generate_settings(source: Path, stack=None):
    stack = stack or load_stack()
    generator = source / "tools" / "settings_register.py"
    subprocess.run([sys.executable, str(generator)], cwd=source, check=True)
    declared = set()
    for feature_id in stack["feature_order"]:
        _, meta = load_feature(feature_id, stack)
        declared.update(meta.get("environment", {}))
    register = source / "terminal" / "data" / "settings-register.json"
    data = load_json(register)
    for entry in data["settings"]:
        if entry["name"] in declared and not entry.get("status"):
            entry["status"] = "opt-in"
    register.write_text(json.dumps(data, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    for options in (["--docs"], ["--check"]):
        subprocess.run([sys.executable, str(generator), *options], cwd=source, check=True)
