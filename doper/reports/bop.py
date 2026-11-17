# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Per-recipe BOP summaries: validation AR, appendix A tables, fixed policies and Table 3.

Reads the outputs of ``doper train``, ``doper evaluate`` and ``doper report --ap/--timing``:

* ``<runs>/<method>/obj_XXXXXX/{validation/ar.json, labels_ready.json, completed.json,
  adaptation_skipped.json}``: the appendix A method x AR-arm tables (A.1 HANDAL, A.2 HOPE,
  A.3 HOT3D) with label counts and skips (``appendix``);
* ``<runs>/<follow-up>/progress.json``: the HANDAL follow-up rounds (Table 89);
* ``<runs>/test/<policy>/{summary.json, SKIPS.json, <csv>}``: the fixed-policy rows (Tables
  87, 88, 89 and 90), with base fallbacks and whether the CSV is timed;
* ``<runs>/test/<policy>/<variant>/{summary.json, variants_manifest.json}`` and
  ``<out>/timing/<policy>/deployment/summary.json``: the Table 3 variant rows, with a warning
  when a policy's CSV and variants used different forward passes (journal generations);
* ``<out>/ap_<recipe>_<policy>.json`` and ``<out>/ap_<recipe>_<policy>_<variant>.json``: the
  local BOP24 AP (T-LESS) of a policy CSV and of a variant CSV.

The validation metric is the localization-style mean of the MSSD and MSPD average recalls (percent),
not AP or VSD; datasets are never pooled. In the appendix macro means, a comparison's population is
the objects whose every method has a single-pass score, so a skipped or missing student is N/A,
never zero, and the paired deltas count improvements, regressions and ties at ``DELTA_TIE``. The
plain macro mean of a method and arm is over the objects scored for it, and the object count is
reported with it. Standard library only.
"""

from __future__ import annotations

import csv
import json
import math
from collections.abc import Mapping, Sequence
from pathlib import Path
from statistics import mean, median
from typing import Any

from ..bop import layout
from ..bop.layout import VARIANTS
from ..recipes.schema import BASE_ARM, Recipe, recipe_arms

VALIDATION_ARMS = ("single", "jitter", "gtbox")
VALIDATION_FILE = Path("validation") / "ar.json"
TEST_DIR = "test"
METRIC = "mean of MSSD and MSPD average recall, percent (not AP/VSD)"
GT_ARMS = ("gt_box_mask", "gt_box_mask_keypoints")
FOUR_WAY = ("static_pseudo", *GT_ARMS)
# A paired difference within this is a tie.
DELTA_TIE = 1e-10
SUMMARY_NAME = "summary.json"
SKIPS_NAME = "SKIPS.json"
VARIANTS_MANIFEST_NAME = "variants_manifest.json"
TIMING_DIR = "timing"
UNTIMED = -1.0


def ap_path(out_dir: Path, recipe_name: str, policy: str, variant: str | None = None) -> Path:
    """The local AP record of a policy CSV, or of one of its variant CSVs (``doper report --ap``).

    ``<out>/ap_<recipe>_<policy>.json``, or ``<out>/ap_<recipe>_<policy>_<variant>.json``.
    """
    suffix = "" if variant is None else f"_{variant}"
    return Path(out_dir) / f"ap_{recipe_name}_{policy}{suffix}.json"


def _policy_dirs(runs: Path) -> list[Path]:
    """The exported test policies' directories (``<runs>/test/<policy>``), sorted."""
    test_dir = Path(runs) / TEST_DIR
    return sorted(test_dir.iterdir()) if test_dir.is_dir() else []


def _read(path: Path) -> Any:  # noqa: ANN401
    return json.loads(path.read_text())


def validation_rows(runs: Path, methods: Sequence[str]) -> list[dict[str, Any]]:
    """One row per scored object and method: frames and AR of every validation arm.

    Args:
        runs: Runs root.
        methods: Method directories to read (``base``, later the adaptation arms).

    Returns:
        Rows sorted by method and object.
    """
    rows = []
    for method in methods:
        for path in sorted((Path(runs) / method).glob(f"obj_*/{VALIDATION_FILE}")):
            result = _read(path)
            row: dict[str, Any] = {
                "method": method,
                "object": result["obj_id"],
                "status": result["status"],
                "protocol": result["protocol"],
                "frames": result["frames"],
                "gt_instances": result["gt_instances"],
            }
            for arm in VALIDATION_ARMS:
                values = result["arms"].get(arm)
                row[f"{arm}_AR"] = None if values is None else values["ar_mean"]
                row[f"{arm}_AR_MSSD"] = None if values is None else values["ar_mssd"]
                row[f"{arm}_AR_MSPD"] = None if values is None else values["ar_mspd"]
            rows.append(row)
    return rows


def macro_means(rows: Sequence[Mapping[str, Any]]) -> dict[str, dict[str, Any]]:
    """Macro mean AR per method and arm over the objects scored for it."""
    means: dict[str, dict[str, Any]] = {}
    for method in sorted({r["method"] for r in rows}):
        selected = [r for r in rows if r["method"] == method]
        means[method] = {}
        for arm in VALIDATION_ARMS:
            values = [r[f"{arm}_AR"] for r in selected if r[f"{arm}_AR"] is not None]
            means[method][arm] = {
                "objects": len(values),
                "AR": mean(values) if values else None,
            }
    return means


def _read_optional(path: Path) -> Any:  # noqa: ANN401
    return _read(path) if path.is_file() else None


def _run_status(run_dir: Path, method: str) -> tuple[str, int | None, int | None]:
    """A run's status, labels and optimizer updates.

    The status is ``base``, ``completed``, ``skipped``, ``labels_ready`` or ``missing``.
    """
    completed = _read_optional(run_dir / layout.COMPLETED_NAME)
    if method == BASE_ARM:
        return ("base" if completed is not None else "missing"), None, None
    if completed is not None:
        return "completed", completed.get("labels"), completed.get("global_step")
    skipped = _read_optional(run_dir / layout.SKIPPED_NAME)
    if skipped is not None:
        return "skipped", skipped.get("labels"), None
    ready = _read_optional(run_dir / layout.LABELS_READY_NAME)
    if ready is not None:
        return "labels_ready", ready.get("labels"), None
    return "missing", None, None


def _arm_scores(arms: Mapping[str, Any]) -> dict[str, float | None]:
    return {
        f"{arm}_AR": None if arms.get(arm) is None else arms[arm]["ar_mean"]
        for arm in VALIDATION_ARMS
    }


def _run_objects(runs: Path, methods: Sequence[str]) -> list[int]:
    found = set()
    for method in methods:
        for path in (Path(runs) / method).glob("obj_*"):
            if path.is_dir() and path.name[4:].isdigit():
                found.add(int(path.name[4:]))
    return sorted(found)


def appendix_rows(runs: Path, methods: Sequence[str]) -> list[dict[str, Any]]:
    """One row per object and method (the base first): status, labels, updates and AR.

    A skipped, unfinished or missing run, or an unscored one, has no AR (None, never zero).

    Args:
        runs: Runs root.
        methods: Adaptation methods, in table order (the base is added).

    Returns:
        Rows by object, then method; ``by_device`` holds per-device AR (native camera).
    """
    rows = []
    for obj in _run_objects(runs, (BASE_ARM, *methods)):
        for method in (BASE_ARM, *methods):
            run_dir = layout.object_run(runs, method, obj)
            status, labels, updates = _run_status(run_dir, method)
            result = _read_optional(run_dir / layout.VALIDATION_PATH) or {}
            scored = result.get("status") == "scored"
            row: dict[str, Any] = {
                "object": obj,
                "method": method,
                "status": status,
                "labels": labels,
                "updates": updates,
                **_arm_scores(result["arms"] if scored else {}),
                "by_device": {
                    device: _arm_scores(arms)
                    for device, arms in (result.get("by_device", {}) if scored else {}).items()
                },
            }
            rows.append(row)
    return rows


def appendix_comparisons(methods: Sequence[str]) -> dict[str, tuple[str, ...]]:
    """The paired comparisons of the appendix tables, each a tuple of methods.

    ``all_common`` (every method), ``GT_paired`` (base and both GT arms), ``four_way_paired``
    (base, static pseudo-labels and both GT arms; the HANDAL four-way table) and one
    ``<method>_paired`` per method, each only when its methods exist.
    """
    comparisons = {"all_common": (BASE_ARM, *methods)}
    if set(GT_ARMS) <= set(methods):
        comparisons["GT_paired"] = (BASE_ARM, *GT_ARMS)
    if set(FOUR_WAY) <= set(methods):
        comparisons["four_way_paired"] = (BASE_ARM, *FOUR_WAY)
    comparisons.update({f"{m}_paired": (BASE_ARM, m) for m in methods})
    return comparisons


def _score(
    by_key: Mapping[tuple[int, str], Mapping[str, Any]],
    obj: int,
    method: str,
    arm: str = "single",
    device: str | None = None,
) -> float | None:
    row = by_key.get((obj, method))
    if row is None:
        return None
    scores = row if device is None else row["by_device"].get(device, {})
    return scores.get(f"{arm}_AR")


def _group_means(
    by_key: Mapping[tuple[int, str], Mapping[str, Any]],
    population: Sequence[int],
    methods: Sequence[str],
    device: str | None,
) -> dict[str, dict[str, float | None]]:
    """Each method's mean AR per arm over a paired population.

    Raises:
        ValueError: If a paired object lacks a jitter or GT-box score, or a score is not finite.
    """
    means: dict[str, dict[str, float | None]] = {}
    for method in methods:
        means[method] = {}
        for arm in VALIDATION_ARMS:
            scores = [_score(by_key, o, method, arm, device) for o in population]
            values = [v for v in scores if v is not None and math.isfinite(v)]
            if len(values) != len(scores):
                raise ValueError(
                    f"paired diagnostic population differs or a score is not finite "
                    f"({method}, {arm})"
                )
            means[method][arm] = mean(values) if values else None
    return means


def paired_groups(
    rows: Sequence[Mapping[str, Any]],
    comparisons: Mapping[str, Sequence[str]],
    device: str | None = None,
) -> dict[str, dict[str, Any]]:
    """Macro mean AR of each comparison over its paired population.

    A comparison whose paired objects lack a jitter or GT-box score (for example a validation
    run without the jitter arm), or have a non-finite score, has no means: its ``methods`` is
    empty and ``unavailable`` names the problem; the other comparisons are unaffected.

    Args:
        rows: ``appendix_rows``.
        comparisons: Name to methods (``appendix_comparisons``).
        device: Score one device's AR instead of the overall AR.

    Returns:
        Per comparison: the objects, their count and each method's mean per arm (None for an
        empty population).
    """
    by_key = {(r["object"], r["method"]): r for r in rows}
    objects = sorted({r["object"] for r in rows})
    groups = {}
    for name, methods in comparisons.items():
        population = [
            o for o in objects if all(_score(by_key, o, m, device=device) is not None
                                      for m in methods)
        ]  # fmt: skip
        group: dict[str, Any] = {"objects": population, "count": len(population)}
        try:
            group["methods"] = _group_means(by_key, population, methods, device)
        except ValueError as err:
            group.update(methods={}, unavailable=str(err))
        groups[name] = group
    return groups


def delta_stats(rows: Sequence[Mapping[str, Any]], before: str, after: str) -> dict[str, Any]:
    """Paired single-pass AR differences ``after - before`` over objects scored for both."""
    by_key = {(r["object"], r["method"]): r for r in rows}
    objects = sorted({r["object"] for r in rows})
    paired = [
        o for o in objects
        if _score(by_key, o, before) is not None and _score(by_key, o, after) is not None
    ]  # fmt: skip
    deltas = [_score(by_key, o, after) - _score(by_key, o, before) for o in paired]  # type: ignore[operator]
    return {
        "objects": paired,
        "count": len(paired),
        "mean_delta": mean(deltas) if deltas else None,
        "median_delta": median(deltas) if deltas else None,
        "improve": sum(d > DELTA_TIE for d in deltas),
        "regress": sum(d < -DELTA_TIE for d in deltas),
        "ties": sum(abs(d) <= DELTA_TIE for d in deltas),
    }


def appendix(runs: Path, methods: Sequence[str]) -> dict[str, Any]:
    """The appendix A tables of one recipe: rows, paired macro means and paired deltas.

    Args:
        runs: Runs root.
        methods: Adaptation methods, in table order.

    Returns:
        ``methods``, ``rows``, ``macro``, ``by_device_macro`` (native camera only),
        ``paired_deltas`` (base to each method) and ``added_GT_keypoints`` (GT box/mask to
        GT box/mask/keypoints, when both exist).
    """
    rows = appendix_rows(runs, methods)
    comparisons = appendix_comparisons(methods)
    devices = sorted({d for r in rows for d in r["by_device"]})
    result: dict[str, Any] = {
        "methods": [BASE_ARM, *methods],
        "rows": rows,
        "macro": paired_groups(rows, comparisons),
        "by_device_macro": {d: paired_groups(rows, comparisons, d) for d in devices},
        "paired_deltas": {m: delta_stats(rows, BASE_ARM, m) for m in methods},
    }
    if set(GT_ARMS) <= set(methods):
        result["added_GT_keypoints"] = delta_stats(rows, *GT_ARMS)
    return result


def followup_rows(runs: Path, recipe: Recipe) -> list[dict[str, Any]]:
    """One row per follow-up round (``<runs>/<follow-up>/progress.json``)."""
    rows = []
    for name, spec in recipe.followups.items():
        progress = _read_optional(Path(runs) / name / layout.PROGRESS_NAME)
        if progress is None:
            continue
        for entry in progress["rounds"]:
            rows.append(
                {
                    "followup": name,
                    "object": spec.obj_id,
                    "round": entry["round"],
                    "status": entry["status"],
                    "labels": entry["labels"],
                    "stop_reason": progress.get("stop_reason"),
                    "final_student_yield": progress.get("final_student_yield"),
                }
            )
    return rows


def _is_timed(csv_path: Path) -> bool | None:
    """Whether a BOP CSV records measured time (None when absent or empty).

    Raises:
        ValueError: If the CSV has no ``time`` column, or mixes untimed (-1) and timed rows.
    """
    if not csv_path.is_file():
        return None
    with csv_path.open(newline="") as stream:
        reader = csv.DictReader(stream)
        if "time" not in (reader.fieldnames or []):
            raise ValueError(f"{csv_path}: no time column")
        timed = {float(row["time"]) != UNTIMED for row in reader}
    if len(timed) > 1:
        raise ValueError(f"{csv_path}: mixes untimed (time -1) and timed rows")
    return timed.pop() if timed else None


def _export_row(folder: Path, export: Mapping[str, Any]) -> dict[str, Any]:
    name = Path(export["csv_path"]).name
    return {
        "csv": name,
        "images": export["images"],
        "objects": len(export["objects"]),
        "missing_objects": list(export["missing_objects"]),
        "pose_rows": export["pose_rows"],
        "timed": _is_timed(folder / name),
    }


def policy_rows(runs: Path, out_dir: Path, recipe_name: str) -> list[dict[str, Any]]:
    """One row per exported test policy: CSV, coverage, base fallbacks, timing and AP."""
    rows = []
    for path in sorted((Path(runs) / TEST_DIR).glob(f"*/{SUMMARY_NAME}")):
        folder, policy = path.parent, path.parent.name
        skips = _read_optional(folder / SKIPS_NAME)
        timing = Path(out_dir) / TIMING_DIR / policy
        rows.append(
            {
                "policy": policy,
                **_export_row(folder, _read(path)),
                "base_fallbacks": [c["object_id"] for c in (skips or {}).get("cases", [])],
                "ap": _read_optional(ap_path(out_dir, recipe_name, policy)),
                "direct_timing": _read_optional(timing / "direct" / SUMMARY_NAME),
            }
        )
    return rows


def variant_rows(runs: Path, out_dir: Path, recipe_name: str) -> list[dict[str, Any]]:
    """One row per variant export (Table 3), with its deployment timing and AP when measured."""
    rows = []
    for policy_dir in _policy_dirs(runs):
        timing = _read_optional(
            Path(out_dir) / TIMING_DIR / policy_dir.name / "deployment" / SUMMARY_NAME
        )
        for variant in VARIANTS:
            folder = policy_dir / variant
            export = _read_optional(folder / SUMMARY_NAME)
            if export is None:
                continue
            manifest = _read_optional(folder / VARIANTS_MANIFEST_NAME) or {}
            rows.append(
                {
                    "policy": policy_dir.name,
                    "variant": variant,
                    "optimizer": manifest.get("optimizer"),
                    "preparation": manifest.get("preparation"),
                    **_export_row(folder, export),
                    "deployment_timing": (timing or {}).get("variants", {}).get(variant),
                    "timing_optimizer": (timing or {}).get("optimizer"),
                    "ap": _read_optional(ap_path(out_dir, recipe_name, policy_dir.name, variant)),
                }
            )
    return rows


def _object_passes(
    sources: Mapping[str, Mapping[str, Sequence[str]]],
) -> list[str]:
    """Objects whose exports of one policy used different forward passes, with the sources."""
    problems = []
    for obj in sorted({o for table in sources.values() for o in table}, key=int):
        tables = {name: list(table[obj]) for name, table in sources.items() if obj in table}
        test = {name: g[0] for name, g in tables.items() if g}
        support = {name: g for name, g in tables.items() if len(g) > 1}
        if len(set(test.values())) > 1 or len({tuple(g) for g in support.values()}) > 1:
            problems.append(f"object {obj} ({', '.join(sorted(tables))})")
    return problems


def generation_warnings(runs: Path) -> list[str]:
    """Policies whose CSV and variant CSVs came from different forward passes of a model.

    A heatmap variant whose rescan differs from the cached journal replaces it (a new journal
    generation), so exports made before and after it hold different detections. Each policy's
    ``summary.json`` and its variants' record every object's journal generations; a policy is
    reported when, for one object, they differ.

    Returns:
        One warning per affected policy.
    """
    warnings = []
    for policy_dir in _policy_dirs(runs):
        sources = {}
        for name in ("", *VARIANTS):
            summary = _read_optional(policy_dir / name / SUMMARY_NAME)
            if summary and summary.get("journal_generations"):
                sources[name or "policy CSV"] = summary["journal_generations"]
        problems = _object_passes(sources)
        if problems:
            warnings.append(
                f"{policy_dir.name}: exports used different forward passes for "
                f"{'; '.join(problems)}; re-export them (`--force`) to compare like with like"
            )
    return warnings


def timing_note(rows: Sequence[Mapping[str, Any]]) -> str | None:
    """A note when some policy CSVs are timed and others untimed (time -1)."""
    timed = sorted(r["policy"] for r in rows if r["timed"] is True)
    untimed = sorted(r["policy"] for r in rows if r["timed"] is False)
    if not (timed and untimed):
        return None
    return (
        f"mixed timing: {', '.join(timed)} record measured per-image time; "
        f"{', '.join(untimed)} are untimed (time -1); their CSV time columns are not comparable"
    )


def test_summaries(runs: Path) -> dict[str, Any]:
    """The export summary of every test policy under ``<runs>/test``."""
    return {
        path.parent.name: _read(path)
        for path in sorted((Path(runs) / TEST_DIR).glob(f"*/{SUMMARY_NAME}"))
    }


def summarize(recipe: Recipe, runs: Path, out_dir: Path | None = None) -> dict[str, Any]:
    """Collect one recipe's validation, appendix, follow-up, policy and variant results.

    Args:
        recipe: The recipe.
        runs: Runs root.
        out_dir: Where ``doper report --ap/--timing`` wrote (default: ``runs``).

    Returns:
        The summary.
    """
    out_dir = Path(out_dir or runs)
    methods = recipe_arms(recipe)[1:]
    rows = validation_rows(runs, (BASE_ARM, *methods))
    policies = policy_rows(runs, out_dir, recipe.name)
    return {
        "recipe": recipe.name,
        "metric": METRIC,
        "validation": rows,
        "macro": macro_means(rows),
        "appendix": appendix(runs, methods),
        "followups": followup_rows(runs, recipe),
        "test": test_summaries(runs),
        "policies": policies,
        "variants": variant_rows(runs, out_dir, recipe.name),
        "timing_note": timing_note(policies),
        "generation_warnings": generation_warnings(runs),
        "pooled_cross_dataset_score": None,
        "test_score_selection": False,
    }


def _cell(value: float | None) -> str:
    return "-" if value is None else f"{value:.2f}"


def _plain(value: Any) -> str:  # noqa: ANN401
    return "-" if value is None else str(value)


def _validation_lines(summary: Mapping[str, Any]) -> list[str]:
    lines = ["## Validation AR", ""]
    lines += ["| method | object | frames | single | jitter | GT box |", "|---" * 6 + "|"]
    for row in summary["validation"]:
        arms = " | ".join(_cell(row[f"{arm}_AR"]) for arm in VALIDATION_ARMS)
        lines.append(f"| {row['method']} | {row['object']} | {row['frames']} | {arms} |")
    for method, arms in summary["macro"].items():
        cells = " | ".join(
            f"{_cell(v['AR'])} ({v['objects']})" for v in (arms[a] for a in VALIDATION_ARMS)
        )
        lines.append(f"| {method} | macro (objects) | | {cells} |")
    return lines


def _appendix_lines(table: Mapping[str, Any]) -> list[str]:
    lines = ["## Appendix A: adaptation arms", ""]
    lines += ["Skipped, unfinished and missing students are N/A (`-`), never zero.", ""]
    lines += ["| object | method | status | labels | updates | single | jitter | GT box |"]
    lines.append("|---" * 8 + "|")
    for r in table["rows"]:
        arms = " | ".join(_cell(r[f"{arm}_AR"]) for arm in VALIDATION_ARMS)
        lines.append(
            f"| {r['object']} | {r['method']} | {r['status']} | {_plain(r['labels'])} | "
            f"{_plain(r['updates'])} | {arms} |"
        )
    for name, group in table["macro"].items():
        lines += ["", f"### {name}: {group['count']} objects", f"IDs: {group['objects']}", ""]
        if "unavailable" in group:
            lines.append(f"Unavailable: {group['unavailable']}.")
            continue
        lines += ["| method | single | jitter | GT box |", "|---|---:|---:|---:|"]
        for method, values in group["methods"].items():
            cells = " | ".join(_cell(values[a]) for a in VALIDATION_ARMS)
            lines.append(f"| {method} | {cells} |")
    return lines


def _ap_cell(ap: Mapping[str, Any] | None) -> str:
    return "-" if ap is None else f"{ap['ap']:.4f} ({ap['targets_origin']})"


def _policy_lines(summary: Mapping[str, Any]) -> list[str]:
    lines = ["## Fixed policies", ""]
    lines += ["| policy | CSV | images | objects | missing | poses | base fallbacks | timed | AP |"]
    lines.append("|---" * 9 + "|")
    for r in summary["policies"]:
        ap = _ap_cell(r["ap"])
        lines.append(
            f"| {r['policy']} | {r['csv']} | {r['images']} | {r['objects']} | "
            f"{r['missing_objects']} | {r['pose_rows']} | {r['base_fallbacks']} | "
            f"{_plain(r['timed'])} | {ap} |"
        )
    if summary["timing_note"]:
        lines += ["", f"Note: {summary['timing_note']}."]
    return lines


def _variant_lines(summary: Mapping[str, Any]) -> list[str]:
    lines = ["## Table 3 variants", ""]
    lines += ["| policy | variant | optimizer | poses | images | deployment median s (opt.) | AP |"]
    lines.append("|---" * 7 + "|")
    for r in summary["variants"]:
        timing = r["deployment_timing"]
        seconds = "-" if timing is None else f"{timing['median_s']:.3f} ({r['timing_optimizer']})"
        lines.append(
            f"| {r['policy']} | {r['variant']} | {_plain(r['optimizer'])} | {r['pose_rows']} | "
            f"{r['images']} | {seconds} | {_ap_cell(r['ap'])} |"
        )
    return lines


def markdown(summary: Mapping[str, Any]) -> str:
    """Render a summary as Markdown tables."""
    lines = [f"# {summary['recipe']} results", "", f"Metric: {summary['metric']}.", ""]
    lines += _validation_lines(summary)
    lines += ["", *_appendix_lines(summary["appendix"])]
    if summary["followups"]:
        lines += ["", "## Follow-up rounds", ""]
        lines += ["| follow-up | object | round | status | labels | stop reason |"]
        lines.append("|---" * 6 + "|")
        for r in summary["followups"]:
            lines.append(
                f"| {r['followup']} | {r['object']} | {r['round']} | {r['status']} | "
                f"{r['labels']} | {_plain(r['stop_reason'])} |"
            )
    lines += ["", *_policy_lines(summary), "", *_variant_lines(summary)]
    for warning in summary["generation_warnings"]:
        lines += ["", f"Warning: {warning}."]
    return "\n".join(lines) + "\n"


def write_report(recipe: Recipe, runs: Path, out_dir: Path | None = None) -> tuple[Path, Path]:
    """Write ``report_<recipe>.json`` and ``.md`` (default: under ``runs``).

    Returns:
        The JSON and Markdown paths.
    """
    out_dir = Path(out_dir or runs)
    out_dir.mkdir(parents=True, exist_ok=True)
    summary = summarize(recipe, runs, out_dir)
    json_path = out_dir / f"report_{recipe.name}.json"
    md_path = out_dir / f"report_{recipe.name}.md"
    json_path.write_text(json.dumps(summary, indent=2, allow_nan=False) + "\n")
    md_path.write_text(markdown(summary))
    return json_path, md_path
