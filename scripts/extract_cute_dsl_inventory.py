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

# Curated decisions are keyed by the inventory's exact, stable IDs. Keep these
# separate from factual CuTe detection: a decision explains why an otherwise
# missing entry is intentionally not a migration target, while `conversion`
# continues to report only what exists in the source tree.
MIGRATION_DECISIONS: dict[str, dict[str, str]] = {
    "attnres.fused_attnres": {
        "status": "no_go",
        "reason": "The fused vector-reduction candidate was substantially slower across representative rows and widths.",
        "source": "profile/cute-dsl-migration/OPT_LOG.md:45",
    },
    "based.fused_chunk_based": {
        "status": "no_go",
        "reason": "The aligned four-warp Taylor-state kernel has no removable padding or launch boundary.",
        "source": "profile/cute-dsl-migration/OPT_LOG.md:73",
    },
    "based.parallel_based": {
        "status": "no_go",
        "reason": "The fused streaming kernel beat analogous native-MMA candidates on the representative envelope.",
        "source": "profile/cute-dsl-migration/OPT_LOG.md:79",
    },
    "comba.fused_recurrent_comba": {
        "status": "no_go",
        "reason": "Its dependent FP32 GEMVs and rank-one update have no useful native-MMA row dimension.",
        "source": "profile/cute-dsl-migration/OPT_LOG.md:67",
    },
    "common.chunk_delta_h": {
        "status": "no_go",
        "reason": "The CuTe state-update candidate passed parity but measured 0.13-0.68x across representative work.",
        "source": "profile/cute-dsl-migration/OPT_LOG.md:26",
    },
    "common.chunk_scaled_dot_kkt": {
        "status": "no_go",
        "reason": "The exact native-MMA KKT candidate measured 0.59-0.65x because its separate epilogue launch dominated.",
        "source": "profile/cute-dsl-migration/OPT_LOG.md:34",
    },
    "delta_rule.fused_chunk_delta_rule": {
        "status": "deprecated",
        "reason": "The compatibility symbol unconditionally raises and directs callers to chunk_delta_rule.",
        "source": "fla/ops/delta_rule/fused_chunk.py:8",
    },
    "delta_rule.fused_recurrent_delta_rule": {
        "status": "no_go",
        "reason": "The bounded CuTe recurrence passed parity but was substantially slower than the existing fused kernel.",
        "source": "profile/cute-dsl-migration/OPT_LOG.md:62",
    },
    "deltaformer.deltaformer_attn": {
        "status": "no_go",
        "reason": "Exact dense intermediates require a large triangular solve that the current CuTe path cannot consume.",
        "source": "profile/cute-dsl-migration/OPT_LOG.md:61",
    },
    "gla.chunk_gla": {
        "status": "no_go",
        "reason": "Representative shapes already use chunk-parallel tensor-core stages and require a full multiwarp pipeline.",
        "source": "profile/cute-dsl-migration/OPT_LOG.md:58",
    },
    "gla.fused_chunk_gla": {
        "status": "deprecated",
        "reason": "The compatibility symbol unconditionally raises and directs callers to chunk_gla.",
        "source": "fla/ops/gla/fused_chunk.py:8",
    },
    "generalized_delta_rule.chunk_iplr_delta_rule": {
        "status": "no_go",
        "reason": "The sequential CuTe recurrence violates the production chunk algorithm's numerical contract.",
        "source": "profile/cute-dsl-migration/OPT_LOG.md:66",
    },
    "generalized_delta_rule.fused_recurrent_iplr_delta_rule": {
        "status": "no_go",
        "reason": "The FP32 CuTe recurrence passed parity but remained slower than the one-launch Triton kernel.",
        "source": "profile/cute-dsl-migration/OPT_LOG.md:56",
    },
    "kda.chunk_kda": {
        "status": "no_go",
        "reason": "Representative execution is an end-to-end multi-stage chunk pipeline, not a replaceable recurrence slice.",
        "source": "profile/cute-dsl-migration/OPT_LOG.md:59",
    },
    "mesa_net.mesa_net_decoding_one_step": {
        "status": "no_go",
        "reason": "The required shared state and cooperative reductions made the CuTe decoder substantially slower.",
        "source": "profile/cute-dsl-migration/OPT_LOG.md:42",
    },
    "nsa.parallel_nsa": {
        "status": "no_go",
        "reason": "Dynamic sparse selection is already fused into one autotuned program and lacks a native sparse-gather primitive.",
        "source": "profile/cute-dsl-migration/OPT_LOG.md:65",
    },
    "parallax.parallel_parallax": {
        "status": "no_go",
        "reason": "The native-MMA candidate passed parity but repeated K/V staging made it slower than Triton.",
        "source": "profile/cute-dsl-migration/OPT_LOG.md:69",
    },
    "rwkv6.chunk_rwkv6": {
        "status": "no_go",
        "reason": "Production shapes favor the existing chunk-parallel tensor-core pipeline over a persistent state recurrence.",
        "source": "profile/cute-dsl-migration/OPT_LOG.md:60",
    },
    "rwkv7.chunk_rwkv7": {
        "status": "no_go",
        "reason": "Production shapes favor the existing chunk-parallel tensor-core pipeline over a persistent state recurrence.",
        "source": "profile/cute-dsl-migration/OPT_LOG.md:60",
    },
    "rwkv7.fused_mul_recurrent_rwkv7": {
        "status": "no_go",
        "reason": "The CuTe recurrent candidate passed parity but Triton's direct register path remained faster.",
        "source": "profile/cute-dsl-migration/OPT_LOG.md:40",
    },
    "rwkv7.fused_recurrent_rwkv7": {
        "status": "no_go",
        "reason": "The shared DPLR recurrence has dependent FP32 GEMVs and rank-one updates with no useful MMA row dimension.",
        "source": "profile/cute-dsl-migration/OPT_LOG.md:77",
    },
    "ttt.fused_chunk_ttt_linear": {
        "status": "no_go",
        "reason": "Splitting its persistent fused recurrence would materialize chunk states and add substantial traffic.",
        "source": "profile/cute-dsl-migration/OPT_LOG.md:72",
    },
    "wall_attn.parallel_wall_attn_decode": {
        "status": "no_go",
        "reason": "The existing fused tensor-core decoder avoids the reductions and value rereads required by a scalar route.",
        "source": "profile/cute-dsl-migration/OPT_LOG.md:70",
    },
    "wall_attn.parallel_wall_attn": {
        "status": "no_go",
        "reason": "The scalar single-query CuTe candidate lost to the existing padded tensor-core decoder.",
        "source": "profile/cute-dsl-migration/OPT_LOG.md:44",
    },
}


def _migration_fields(item_id: str, conversion: dict[str, Any]) -> dict[str, Any]:
    decision = MIGRATION_DECISIONS.get(item_id)
    if conversion["status"] == "present_unverified":
        coverage_status = "present_unverified"
    elif conversion["status"] == "not_applicable_platform":
        coverage_status = "not_applicable_platform"
    elif decision is not None:
        coverage_status = decision["status"]
    else:
        coverage_status = "missing"
    return {
        "migration_decision": decision,
        "coverage_status": coverage_status,
    }


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


def _module_file(repo_root: Path, module: str) -> Path | None:
    path = repo_root.joinpath(*module.split("."))
    if path.with_suffix(".py").is_file():
        return path.with_suffix(".py")
    if (path / "__init__.py").is_file():
        return path / "__init__.py"
    return None


def _absolute_import_module(repo_root: Path, source_file: Path, node: ast.ImportFrom) -> str:
    relative = source_file.relative_to(repo_root).with_suffix("")
    package = list(relative.parts[:-1])
    if node.level:
        prefix = package[: len(package) - node.level + 1]
        return ".".join(prefix + ((node.module or "").split(".") if node.module else []))
    return node.module or ""


def _resolve_imported_symbols(
    repo_root: Path,
    module: str,
    names: set[str],
    seen: set[tuple[str, tuple[str, ...]]] | None = None,
) -> list[tuple[str, Path]]:
    seen = set() if seen is None else seen
    key = (module, tuple(sorted(names)))
    if key in seen:
        return []
    seen.add(key)
    path = _module_file(repo_root, module)
    if path is None:
        return []
    if path.name != "__init__.py" or "*" in names:
        return [(module, path)]
    resolved = []
    for node in ast.walk(ast.parse(path.read_text())):
        if not isinstance(node, ast.ImportFrom):
            continue
        child_module = _absolute_import_module(repo_root, path, node)
        for alias in node.names:
            if (alias.asname or alias.name) in names:
                resolved.extend(_resolve_imported_symbols(repo_root, child_module, {alias.name}, seen))
    return resolved or [(module, path)]


def _imported_modules(repo_root: Path, source_file: Path) -> list[tuple[str, Path]]:
    tree = ast.parse(source_file.read_text())
    imports = []
    for node in ast.walk(tree):
        if not isinstance(node, ast.ImportFrom):
            continue
        module = _absolute_import_module(repo_root, source_file, node)
        if not module.startswith("fla.ops."):
            continue
        imports.extend(_resolve_imported_symbols(repo_root, module, {alias.asname or alias.name for alias in node.names}))
    return imports


def _transitive_cute_files(repo_root: Path, source_file: Path) -> set[Path]:
    """Find CuTe dispatches reached through wrapper-only Python modules."""
    found: set[Path] = set()
    pending = [source_file]
    visited: set[Path] = set()
    while pending:
        path = pending.pop()
        if path in visited:
            continue
        visited.add(path)
        text = path.read_text(errors="ignore")
        imported = _imported_modules(repo_root, path)
        for module, imported_path in imported:
            if module.startswith("fla.ops.backends.cute."):
                if module != "fla.ops.backends.cute.runtime" and "cute.compile" in imported_path.read_text(errors="ignore"):
                    found.add(imported_path)
            elif "@triton.jit" not in text and "@T.prim_func" not in text:
                pending.append(imported_path)
    return found


def _native_cute_files(
    repo_root: Path,
    domain: str | None = None,
    module_stem: str | None = None,
    symbol: str | None = None,
    source_file: Path | None = None,
) -> list[str]:
    candidates: list[Path] = []
    direct_candidates: set[Path] = set()
    if domain is not None:
        candidates.extend((repo_root / "fla/ops" / domain / "backends/cute").glob("*.py"))
        candidates.append(repo_root / "fla/ops/backends/cute" / f"{domain}.py")
    if module_stem is not None:
        candidates.append(repo_root / "fla/ops/backends/cute" / f"{module_stem}.py")
    if source_file is not None:
        source_text = source_file.read_text(errors="ignore")
        for module in re.findall(r"from fla\.ops\.backends\.cute\.([\w.]+) import", source_text):
            if module == "runtime":
                continue
            path = repo_root.joinpath("fla", "ops", "backends", "cute", *module.split(".")).with_suffix(".py")
            candidates.append(path)
            direct_candidates.add(path)
        if "@triton.jit" not in source_text and "@T.prim_func" not in source_text:
            transitive_candidates = _transitive_cute_files(repo_root, source_file)
            candidates.extend(transitive_candidates)
            direct_candidates.update(transitive_candidates)
    return sorted(
        str(path.relative_to(repo_root))
        for path in set(candidates)
        if path.is_file()
        and "cutlass.cute" in (text := path.read_text(errors="ignore"))
        and (path in direct_candidates or symbol is None or re.search(rf"\b{re.escape(symbol)}\b", text))
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
                source_file=source_file,
            )
            symbol_tests = _symbol_tests(test_files, symbol, repo_root)
            item_id = f"{domain_dir.name}.{symbol}"
            conversion = {
                "status": "present_unverified" if cute_files else "missing",
                "cute_files": cute_files,
            }
            records.append(
                {
                    "id": item_id,
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
                    "conversion": conversion,
                    **_migration_fields(item_id, conversion),
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
            is_platform_backend = "backends" in path.relative_to(ops_root).parts
            if is_platform_backend:
                cute_files = []
            else:
                cute_files = sorted(
                    set(_native_cute_files(repo_root, module_stem=path.stem))
                    | set(_native_cute_files(repo_root, module_stem=f"{namespace.replace('/', '_')}_{path.stem}"))
                )
            item_id = module.removeprefix("fla.ops.")
            conversion = {
                "status": (
                    "not_applicable_platform" if is_platform_backend else ("present_unverified" if cute_files else "missing")
                ),
                "cute_files": cute_files,
            }
            records.append(
                {
                    "id": item_id,
                    "file": str(path.relative_to(repo_root)),
                    "implementation_backends": _backend_markers(text),
                    "kernel_symbols": _kernel_symbols(path),
                    "direct_consumers": consumers,
                    "consumer_count": len(consumers),
                    "tests": _direct_tests(test_files, module, repo_root),
                    "conversion": conversion,
                    **_migration_fields(item_id, conversion),
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
    all_records = [*public_ops, *shared_internals]
    inventory_ids = {item["id"] for item in all_records}
    unknown_decisions = sorted(MIGRATION_DECISIONS.keys() - inventory_ids)
    if unknown_decisions:
        raise AssertionError(f"Migration decisions reference unknown exact IDs: {unknown_decisions}")

    manifest = {
        "manifest_version": 2,
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
            "platform_exclusions": "TileLang and triton_ascend backend files are not CUDA CuTe migration targets",
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
        "public_coverage_present_unverified": sum(item["coverage_status"] == "present_unverified" for item in public_ops),
        "public_coverage_no_go": sum(item["coverage_status"] == "no_go" for item in public_ops),
        "public_coverage_deprecated": sum(item["coverage_status"] == "deprecated" for item in public_ops),
        "public_coverage_missing": sum(item["coverage_status"] == "missing" for item in public_ops),
        "shared_coverage_present_unverified": sum(
            item["coverage_status"] == "present_unverified" for item in shared_internals
        ),
        "shared_coverage_not_applicable_platform": sum(
            item["coverage_status"] == "not_applicable_platform" for item in shared_internals
        ),
        "shared_coverage_no_go": sum(item["coverage_status"] == "no_go" for item in shared_internals),
        "shared_coverage_deprecated": sum(item["coverage_status"] == "deprecated" for item in shared_internals),
        "shared_coverage_missing": sum(item["coverage_status"] == "missing" for item in shared_internals),
        "migration_decision_no_go": sum(
            item["migration_decision"] is not None and item["migration_decision"]["status"] == "no_go" for item in all_records
        ),
        "migration_decision_deprecated": sum(
            item["migration_decision"] is not None and item["migration_decision"]["status"] == "deprecated"
            for item in all_records
        ),
        "true_missing_ids": sorted(item["id"] for item in all_records if item["coverage_status"] == "missing"),
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
