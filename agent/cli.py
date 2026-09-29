#!/usr/bin/env python3
"""Owner dispatcher. Runs the existing command and returns one envelope."""

from __future__ import annotations

import json
import os
import shutil
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parent
REPO = ROOT.parent
REQUIRED = {
    "name",
    "version",
    "description",
    "command",
    "input_schema",
    "output_schema",
    "side_effect_class",
    "approval_policy",
    "timeout_class",
    "required_dependencies",
}
SIDE = {"READ_ONLY", "LOCAL_WRITE", "AUTHORITATIVE_WRITE", "DESTRUCTIVE"}
SAFE_SUBCOMMANDS = {
    "metrics", "corpus", "capabilities", "list", "status", "layers", "sources", "models",
    "catalog", "services", "stats", "ping", "validate", "validate-contract", "validate-config",
    "plan", "inspect", "sync-dry-run", "describe", "list-futures", "coverage", "gaps",
    "rules", "profiles", "predict", "check", "preview", "dry-run", "known", "objects",
    "dags", "runs", "formats", "types", "schema", "principle", "diagrams", "classify",
    "kinds", "recipe", "jurisdictions", "summary", "doctor", "list-packs", "review-list",
    "native-id",
}


def manifest() -> dict:
    return json.loads((ROOT / "agent.json").read_text())


def tool(document: dict, name: str) -> dict:
    for item in document["tools"]:
        if item["name"] == name:
            return item
    raise SystemExit(2)


def now() -> str:
    return datetime.now(timezone.utc).isoformat()


def resolve_bin(command: str) -> Path | None:
    candidates = []
    config = REPO / ".cargo" / "config.toml"
    if config.is_file():
        for line in config.read_text(errors="replace").splitlines():
            stripped = line.split("#", 1)[0].strip()
            if stripped.startswith("target-dir"):
                target = Path(stripped.split("=", 1)[1].strip().strip('"').strip("'"))
                candidates.extend([target / "release" / command, target / "debug" / command])
    candidates.extend([REPO / "target" / "release" / command, REPO / "target" / "debug" / command])
    shared = Path("/Volumes/reform/caches/cargo-targets")
    if shared.is_dir():
        for child in shared.iterdir():
            candidates.extend([child / "release" / command, child / "debug" / command])
    for candidate in candidates:
        if candidate.is_file() and os.access(candidate, os.X_OK):
            return candidate
    found = shutil.which(command)
    return Path(found) if found else None


def deps(item: dict) -> list[dict]:
    rows = []
    for name in item.get("required_dependencies") or []:
        if name == "local executable":
            path = resolve_bin(item["command"])
            rows.append({"name": name, "status": "available" if path else "unavailable", "detail": str(path or item["command"])})
        elif name == "Cargo":
            rows.append({"name": name, "status": "available" if shutil.which("cargo") else "unavailable", "detail": "cargo"})
        elif name == "Python":
            rows.append({"name": name, "status": "available" if shutil.which("python3") else "unavailable", "detail": "python3"})
        elif name == "Fabric":
            set_ = bool(os.environ.get("LANDMARK_FABRIC_URL", "").strip())
            rows.append({"name": name, "status": "available" if set_ else "unavailable", "detail": "LANDMARK_FABRIC_URL"})
        else:
            rows.append({"name": name, "status": "unavailable", "detail": "not configured in this environment"})
    return rows


def envelope(document: dict, item: dict, *, status: str, started: str, result, error, audit: dict, approved: bool) -> dict:
    return {
        "agent": document["agent"],
        "agent_version": document["version"],
        "owner_repository": document["repository"],
        "git_revision": document["git_revision"],
        "tool": item["name"],
        "tool_version": item["version"],
        "status": status,
        "started_at": started,
        "completed_at": now(),
        "result": result,
        "error": error,
        "dependency_status": deps(item),
        "side_effect_class": item["side_effect_class"],
        "approval": {"required": bool(item["approval_policy"].get("required")), "granted": approved},
        "audit": audit,
    }


def emit(payload: dict, code: int = 0) -> int:
    json.dump(payload, sys.stdout)
    sys.stdout.write("\n")
    return code


def validate(document: dict) -> list[str]:
    errors = []
    if not re_sha(document.get("git_revision", "")):
        errors.append("git_revision is not a 40-character sha")
    names = []
    for item in document.get("tools") or []:
        missing = REQUIRED - set(item)
        if missing:
            errors.append(f"{item.get('name')}: missing {sorted(missing)}")
        if item.get("side_effect_class") not in SIDE:
            errors.append(f"{item.get('name')}: bad side effect")
        if not isinstance(item.get("input_schema"), dict) or "type" not in item.get("input_schema", {}):
            errors.append(f"{item.get('name')}: input_schema")
        if not isinstance(item.get("output_schema"), dict) or "type" not in item.get("output_schema", {}):
            errors.append(f"{item.get('name')}: output_schema")
        if item.get("name") in names:
            errors.append(f"duplicate tool {item.get('name')}")
        names.append(item.get("name"))
        policy = item.get("approval_policy") or {}
        needs = item.get("side_effect_class") in {"AUTHORITATIVE_WRITE", "DESTRUCTIVE"}
        if needs and policy.get("required") is not True:
            errors.append(f"{item.get('name')}: approval policy")
        if not item.get("command"):
            errors.append(f"{item.get('name')}: missing command")
    return errors


def re_sha(value: str) -> bool:
    return len(value) == 40 and all(ch in "0123456789abcdef" for ch in value)


def argv_from(item: dict, arguments: dict) -> list[str]:
    schema = item["input_schema"]
    props = schema.get("properties") or {}
    if "subcommand" in props:
        sub = str(arguments["subcommand"])
        args = [sub]
        nested = ((props.get(sub) or {}).get("properties") or {})
        args.extend(flags(nested, arguments.get(sub) or {}))
        return args
    return flags(props, arguments)


def flags(props: dict, values: dict) -> list[str]:
    args = []
    for key, spec in props.items():
        if key not in values:
            continue
        if spec.get("x-cli-positional"):
            args.append(str(values[key]))
            continue
        flag = spec.get("x-cli-flag") or ("--" + key.replace("_", "-"))
        if spec.get("type") == "boolean":
            if values[key]:
                args.append(flag)
            continue
        args.extend([flag, str(values[key])])
    return args


def run_tool(document: dict, item: dict, arguments: dict, approved: bool) -> int:
    started = now()
    sub = str((arguments or {}).get("subcommand") or "")
    if item["side_effect_class"] in {"AUTHORITATIVE_WRITE", "DESTRUCTIVE"} and not approved and sub not in SAFE_SUBCOMMANDS:
        return emit(envelope(document, item, status="error", started=started, result=None, error="approval required", audit={"executed": False, "exit_code": None, "stdout": "", "stderr": ""}, approved=False), 1)
    invoke = item.get("invoke") or {}
    if invoke.get("kind") == "python" and invoke.get("module"):
        venv = REPO / ".venv" / "bin" / "python"
        python = str(venv) if venv.is_file() else (shutil.which("python3") or "python3")
        command = [python, "-m", invoke["module"], *argv_from(item, arguments)]
    else:
        binary = resolve_bin(item["command"])
        if binary is None:
            return emit(envelope(document, item, status="error", started=started, result=None, error=f"dependency unavailable: {item['command']}", audit={"executed": False, "exit_code": None, "stdout": "", "stderr": ""}, approved=approved), 1)
        command = [str(binary), *argv_from(item, arguments)]
    timeout = {"FAST": 10, "NORMAL": 60, "LONG": 300, "SERVER": 10, "short": 10, "standard": 60, "long": 300}[item["timeout_class"]]
    env = os.environ.copy()
    env.setdefault("TMPDIR", "/Volumes/gitty-up/tmp")
    base_path = os.pathsep.join(part for part in [str(REPO / "src"), str(REPO), env.get("PYTHONPATH", "")] if part)
    extra = (item.get("invoke") or {}).get("pythonpath")
    env["PYTHONPATH"] = (extra + os.pathsep + base_path) if extra else base_path
    try:
        completed = subprocess.run(command, cwd=REPO, check=False, capture_output=True, text=True, timeout=timeout, env=env)
    except subprocess.TimeoutExpired as exc:
        return emit(envelope(document, item, status="error", started=started, result=None, error="timeout", audit={"executed": True, "exit_code": None, "stdout": exc.stdout or "", "stderr": exc.stderr or "", "command": command}, approved=approved), 1)
    stdout = completed.stdout or ""
    stderr = completed.stderr or ""
    value = parse_json(stdout)
    status = "ok" if completed.returncode == 0 else "error"
    error = None if completed.returncode == 0 else (stderr.strip() or stdout.strip() or f"exit {completed.returncode}")[:2000]
    return emit(
        envelope(
            document,
            item,
            status=status,
            started=started,
            result={"value": value, "text": None if isinstance(value, (dict, list)) else stdout},
            error=error,
            audit={"executed": True, "exit_code": completed.returncode, "stdout": stdout[-8000:], "stderr": stderr[-8000:], "command": command},
            approved=approved,
        ),
        0 if completed.returncode == 0 else completed.returncode or 1,
    )


def parse_json(text: str):
    start = text.find("{")
    if start < 0:
        start = text.find("[")
    if start < 0:
        return None
    try:
        return json.loads(text[start:])
    except json.JSONDecodeError:
        return None


def main(argv: list[str]) -> int:
    document = manifest()
    if len(argv) < 2:
        return 2
    command = argv[1]
    if command == "version":
        return emit({"agent": document["agent"], "version": document["version"], "owner": document["owner"], "git_revision": document["git_revision"]})
    if command == "capabilities":
        return emit({"agent": document["agent"], "tools": document["tools"]})
    if command == "health":
        rows = [{"tool": item["name"], "dependencies": deps(item)} for item in document["tools"]]
        states = []
        for row in rows:
            have = {item["status"] for item in row["dependencies"]}
            states.append("ready" if have == {"available"} else "partial" if "available" in have else "unavailable")
        status = "ready" if states and set(states) == {"ready"} else "partial" if "ready" in states or "partial" in states else "unavailable"
        return emit({"agent": document["agent"], "git_revision": document["git_revision"], "status": status, "tools": rows})
    if command == "validate":
        errors = validate(document)
        if errors:
            return emit({"status": "error", "errors": errors}, 1)
        return emit({"status": "ok", "agent": document["agent"], "tools": len(document["tools"])})
    if command == "run" and len(argv) >= 3:
        approved = "--approve" in argv
        raw = [item for item in argv[3:] if item != "--approve"]
        arguments = json.loads(raw[0]) if raw else {}
        try:
            item = tool(document, argv[2])
        except SystemExit:
            return emit({"error": "unknown tool", "tool": argv[2]}, 2)
        return run_tool(document, item, arguments, approved)
    return emit({"error": "unknown command"}, 2)


if __name__ == "__main__":
    raise SystemExit(main(sys.argv))
