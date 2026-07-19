# Copyright (c) 2023-2026, Songlin Yang, Yu Zhang, Zhiyuan Li
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
# For a list of all contributors, visit:
#   https://github.com/fla-org/flash-linear-attention/graphs/contributors

"""Generate the authoritative CuTe DSL operator migration inventory."""

from __future__ import annotations

import argparse
import ast
import json
import re
from pathlib import Path
from typing import Any

UPSTREAM_BASELINE_COMMIT = "fe8fce9f"
CUTE_MIGRATION_BASE_COMMIT = "bef176b4"
EXPECTED_PUBLIC_OPS = 65
EXPECTED_SHARED_KERNEL_FILES = 31
INFRASTRUCTURE_DIRS = {"__pycache__", "backends", "common", "cp", "utils"}
SHARED_NAMESPACES = ("common", "utils", "cp")


def _literal_keyword(call: ast.Call, name: str, default: Any = None) -> Any:
    for keyword in call.keywords:
        if keyword.arg == name:
            try:
                return ast.literal_eval(keyword.value)
            except (TypeError, ValueError):
                return default
    return default


def _exports(init_file: Path) -> tuple[list[str], dict[str, str]]:
    tree = ast.parse(init_file.read_text())
    exports: list[str] = []
    imports = {
        alias.asname or alias.name: node.module or ""
        for node in tree.body
        if isinstance(node, ast.ImportFrom)
        for alias in node.names
    }
    for node in ast.walk(tree):
        if not isinstance(node, ast.Assign):
            continue
        if any(isinstance(target, ast.Name) and target.id == "__all__" for target in node.targets):
            exports = ast.literal_eval(node.value)
            break
    return exports, imports


def _export_role(symbol: str, source_module: str) -> str:
    if symbol.startswith("naive_") or source_module.split(".")[-1] == "naive":
        return "reference"
    if symbol == "build_wall_kv_cache":
        return "helper"
    return "optimized"


def _resolve_source_file(domain_dir: Path, source_module: str) -> Path:
    if source_module.startswith("fla.ops."):
        module_path = domain_dir.parents[2].joinpath(*source_module.split("."))
    else:
        module_path = domain_dir.joinpath(*source_module.split("."))
    module_file = module_path.with_suffix(".py")
    if module_file.is_file():
        return module_file
    package_init = module_path / "__init__.py"
    if package_init.is_file():
        return package_init
    raise FileNotFoundError(f"Cannot resolve relative module {source_module!r} below {domain_dir}")


def _backend_markers(text: str) -> list[str]:
    markers = {
        "cute": r"(?:import cutlass\.cute|from cutlass import cute|cute\.compile)",
        "external_cutlass": r"\bCUTLASS\b",
        "gluon": r"@gluon\.jit",
        "tilelang": r"(?:import|from) tilelang|@T\.prim_func",
        "triton": r"@triton\.jit",
    }
    return [backend for backend, pattern in markers.items() if re.search(pattern, text)]


def _native_cute_files(
    repo_root: Path,
    domain: str | None = None,
    module_stem: str | None = None,
    symbol: str | None = None,
) -> list[str]:
    candidates: list[Path] = []
    if domain is not None:
        candidates.extend((repo_root / "fla/ops" / domain / "backends/cute").glob("*.py"))
        candidates.append(repo_root / "fla/ops/backends/cute" / f"{domain}.py")
    if module_stem is not None:
        candidates.append(repo_root / "fla/ops/backends/cute" / f"{module_stem}.py")
    return sorted(
        str(path.relative_to(repo_root))
        for path in set(candidates)
        if path.is_file()
        and "cutlass.cute" in (text := path.read_text(errors="ignore"))
        and (symbol is None or re.search(rf"\b{re.escape(symbol)}\b", text))
    )


def _benchmark_registry(repo_root: Path) -> list[dict[str, Any]]:
    registry_file = repo_root / "benchmarks/ops/registry.py"
    tree = ast.parse(registry_file.read_text())
    entries = []
    for node in ast.walk(tree):
        if not (
            isinstance(node, ast.Call)
            and isinstance(node.func, ast.Name)
            and node.func.id == "register_op"
            and node.args
            and isinstance(node.args[0], ast.Call)
        ):
            continue
        config = node.args[0]
        name = _literal_keyword(config, "name")
        import_path = _literal_keyword(config, "import_path")
        func_name = _literal_keyword(config, "func_name", name) or name
        entries.append(
            {
                "name": name,
                "import_path": import_path,
                "func_name": func_name,
                "test_file": _literal_keyword(config, "test_file"),
                "modes": ["fwd"] if _literal_keyword(config, "skip_backward", False) else ["fwd", "fwdbwd"],
            }
        )
    return sorted(entries, key=lambda entry: entry["name"])


def _direct_tests(test_files: list[Path], import_prefix: str, repo_root: Path) -> list[str]:
    return sorted(str(path.relative_to(repo_root)) for path in test_files if import_prefix in path.read_text(errors="ignore"))


def _symbol_tests(test_files: list[Path], symbol: str, repo_root: Path) -> list[str]:
    pattern = re.compile(rf"\b{re.escape(symbol)}\b")
    return sorted(str(path.relative_to(repo_root)) for path in test_files if pattern.search(path.read_text(errors="ignore")))


def _shared_dependencies(domain_dir: Path) -> list[str]:
    dependencies: set[str] = set()
    pattern = re.compile(r"(?:from|import) fla\.ops\.(common|utils|cp)(?:\.([\w.]+))?")
    for path in domain_dir.rglob("*.py"):
        for namespace, module in pattern.findall(path.read_text(errors="ignore")):
            dependencies.add(namespace if not module else f"{namespace}.{module}")
    return sorted(dependencies)


def _public_ops(repo_root: Path, registry: list[dict[str, Any]], test_files: list[Path]) -> list[dict[str, Any]]:
    ops_root = repo_root / "fla/ops"
    records = []
    domains = sorted(path for path in ops_root.iterdir() if path.is_dir() and path.name not in INFRASTRUCTURE_DIRS)
    for domain_dir in domains:
        exports, imports = _exports(domain_dir / "__init__.py")
        domain_text = "\n".join(path.read_text(errors="ignore") for path in domain_dir.rglob("*.py"))
        domain_tests = _direct_tests(test_files, f"fla.ops.{domain_dir.name}", repo_root)
        dependencies = _shared_dependencies(domain_dir)
        for symbol in exports:
            source_module = imports.get(symbol, "")
            if _export_role(symbol, source_module) != "optimized":
                continue
            source_file = _resolve_source_file(domain_dir, source_module)
            benchmark_targets = [
                entry
                for entry in registry
                if entry["import_path"] == f"fla.ops.{domain_dir.name}" and entry["func_name"] == symbol
            ]
            cute_files = _native_cute_files(
                repo_root,
                domain=domain_dir.name,
                module_stem=source_file.stem,
                symbol=symbol,
            )
            symbol_tests = _symbol_tests(test_files, symbol, repo_root)
            records.append(
                {
                    "id": f"{domain_dir.name}.{symbol}",
                    "domain": domain_dir.name,
                    "symbol": symbol,
                    "source_module": source_module,
                    "source_file": str(source_file.relative_to(repo_root)),
                    "implementation_backends": _backend_markers(domain_text),
                    "shared_dependencies": dependencies,
                    "tests": symbol_tests,
                    "symbol_test_present": bool(symbol_tests),
                    "domain_tests": domain_tests,
                    "domain_test_present": bool(domain_tests),
                    "benchmark_targets": [target["name"] for target in benchmark_targets],
                    "benchmark_modes": sorted({mode for target in benchmark_targets for mode in target["modes"]}),
                    "benchmark_registered": bool(benchmark_targets),
                    "conversion": {
                        "status": "present_unverified" if cute_files else "missing",
                        "cute_files": cute_files,
                    },
                    "evidence": {
                        "correctness": {"status": "unrun", "command": None, "artifact": None},
                        "benchmark": {"status": "unrun", "rows": []},
                        "ncu": {"status": "unrun", "artifacts": [], "metrics": {}},
                    },
                }
            )
    return records


def _kernel_symbols(path: Path) -> list[str]:
    tree = ast.parse(path.read_text())
    symbols = []
    for node in ast.walk(tree):
        if not isinstance(node, ast.FunctionDef | ast.AsyncFunctionDef):
            continue
        decorators = [
            ast.unparse(decorator.func if isinstance(decorator, ast.Call) else decorator) for decorator in node.decorator_list
        ]
        if "triton.jit" in decorators or "T.prim_func" in decorators:
            symbols.append(node.name)
    return symbols


def _shared_internals(repo_root: Path, test_files: list[Path]) -> list[dict[str, Any]]:
    ops_root = repo_root / "fla/ops"
    op_files = list(ops_root.rglob("*.py"))
    records = []
    for namespace in SHARED_NAMESPACES:
        for path in sorted((ops_root / namespace).rglob("*.py")):
            text = path.read_text(errors="ignore")
            if not ("@triton.jit" in text or re.search(r"(?:import|from) tilelang", text) or "@T.prim_func" in text):
                continue
            module = ".".join(path.relative_to(repo_root).with_suffix("").parts)
            consumers = sorted(
                str(candidate.relative_to(repo_root))
                for candidate in op_files
                if namespace not in candidate.relative_to(ops_root).parts[:1]
                and f"from {module} import" in candidate.read_text(errors="ignore")
            )
            cute_files = _native_cute_files(repo_root, module_stem=path.stem)
            records.append(
                {
                    "id": module.removeprefix("fla.ops."),
                    "file": str(path.relative_to(repo_root)),
                    "implementation_backends": _backend_markers(text),
                    "kernel_symbols": _kernel_symbols(path),
                    "direct_consumers": consumers,
                    "consumer_count": len(consumers),
                    "tests": _direct_tests(test_files, module, repo_root),
                    "conversion": {
                        "status": "present_unverified" if cute_files else "missing",
                        "cute_files": cute_files,
                    },
                }
            )
    return records


def build_manifest(repo_root: Path) -> dict[str, Any]:
    registry = _benchmark_registry(repo_root)
    test_files = list((repo_root / "tests").rglob("test_*.py"))
    public_ops = _public_ops(repo_root, registry, test_files)
    shared_internals = _shared_internals(repo_root, test_files)
    if len(public_ops) != EXPECTED_PUBLIC_OPS:
        raise AssertionError(f"Expected {EXPECTED_PUBLIC_OPS} public optimized entry points, found {len(public_ops)}")
    if len(shared_internals) != EXPECTED_SHARED_KERNEL_FILES:
        raise AssertionError(f"Expected {EXPECTED_SHARED_KERNEL_FILES} shared kernel files, found {len(shared_internals)}")

    manifest = {
        "manifest_version": 1,
        "baseline": {
            "upstream_commit": UPSTREAM_BASELINE_COMMIT,
            "cute_migration_base_commit": CUTE_MIGRATION_BASE_COMMIT,
            "inventory_base_commit": CUTE_MIGRATION_BASE_COMMIT,
        },
        "scope": {
            "public_definition": "optimized symbols in operator subpackage __all__ exports",
            "reference_exclusions": "symbols imported from naive.py or prefixed naive_",
            "helper_exclusions": ["wall_attn.build_wall_kv_cache"],
            "shared_definition": "kernel-bearing Python files under fla/ops/common, fla/ops/utils, and fla/ops/cp",
        },
        "public_ops": public_ops,
        "shared_internals": shared_internals,
        "benchmark_registry": registry,
    }
    manifest["summary"] = {
        "public_optimized_entry_points": len(public_ops),
        "cute_public_present_unverified": sum(op["conversion"]["status"] == "present_unverified" for op in public_ops),
        "cute_public_missing": sum(op["conversion"]["status"] == "missing" for op in public_ops),
        "benchmark_registered": sum(op["benchmark_registered"] for op in public_ops),
        "benchmark_gaps": sum(not op["benchmark_registered"] for op in public_ops),
        "symbol_test_gaps": [op["id"] for op in public_ops if not op["symbol_test_present"]],
        "domain_test_gaps": [op["id"] for op in public_ops if not op["domain_test_present"]],
        "shared_kernel_files": len(shared_internals),
        "cute_shared_present_unverified": sum(
            item["conversion"]["status"] == "present_unverified" for item in shared_internals
        ),
        "cute_shared_missing": sum(item["conversion"]["status"] == "missing" for item in shared_internals),
    }
    return manifest


def main() -> int:
    repo_root = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output",
        type=Path,
        default=repo_root / "benchmarks/ops/cute_dsl_inventory.json",
        help="Generated manifest path. Default: benchmarks/ops/cute_dsl_inventory.json",
    )
    parser.add_argument("--check", action="store_true", help="Fail if the output does not match the generated manifest.")
    args = parser.parse_args()

    rendered = json.dumps(build_manifest(repo_root), indent=2, sort_keys=True) + "\n"
    output = args.output if args.output.is_absolute() else repo_root / args.output
    if args.check:
        if not output.is_file() or output.read_text() != rendered:
            raise SystemExit(f"Inventory is stale; regenerate with {Path(__file__).relative_to(repo_root)}")
        return 0
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(rendered)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
