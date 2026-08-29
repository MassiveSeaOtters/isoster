"""Load campaign YAML + arm YAMLs and expand parameterized arms.

The orchestrator calls :func:`load_campaign` once at startup; the
returned :class:`CampaignPlan` is a pure data object with no I/O
handles, safe to pass to worker processes.
"""

from __future__ import annotations

import importlib
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml

from ..adapters.base import DatasetAdapter, expand_root

REPO_ROOT = Path(__file__).resolve().parents[3]


@dataclass
class ToolPlan:
    name: str
    enabled: bool
    arms_file: Path
    arms: dict[str, dict[str, Any]]  # arm_id -> delta dict (sentinels unresolved)
    extra: dict[str, Any] = field(default_factory=dict)


@dataclass
class DatasetPlan:
    name: str
    enabled: bool
    adapter_name: str
    adapter: DatasetAdapter
    select: list[str] | None = None
    extra: dict[str, Any] = field(default_factory=dict)


@dataclass
class CampaignPlan:
    campaign_name: str
    output_root: Path
    tools: dict[str, ToolPlan]
    datasets: dict[str, DatasetPlan]
    qa: dict[str, Any]
    execution: dict[str, Any]
    summary: dict[str, Any]
    isoster_harmonic_sweeps: list[list[int]]
    raw: dict[str, Any]  # original YAML for snapshotting


def load_campaign(yaml_path: str | Path) -> CampaignPlan:
    """Load and validate a campaign YAML into a :class:`CampaignPlan`."""
    yaml_path = Path(yaml_path).expanduser().resolve()
    with yaml_path.open() as handle:
        raw = yaml.safe_load(handle)
    if raw is None:
        raise ValueError(f"Campaign YAML is empty: {yaml_path}")

    campaign_name = _require(raw, "campaign_name", yaml_path)
    output_root = expand_root(_require(raw, "output_root", yaml_path))

    # A missing key falls back to the default sweep; an explicit empty
    # list disables higher-order harmonic expansion entirely.
    harmonic_sweeps = raw.get("isoster_harmonic_sweeps")
    if harmonic_sweeps is None:
        harmonic_sweeps = [[5, 6]]
    if not isinstance(harmonic_sweeps, list) or not all(
        isinstance(entry, list) and all(isinstance(n, int) for n in entry)
        for entry in harmonic_sweeps
    ):
        raise ValueError(
            f"{yaml_path}: isoster_harmonic_sweeps must be a list of int lists, "
            f"got {harmonic_sweeps!r}"
        )

    tools = _load_tools(raw.get("tools", {}), yaml_path, harmonic_sweeps)
    datasets = _load_datasets(raw.get("datasets", {}), yaml_path)

    qa = _dict_with_bool_defaults(
        raw.get("qa", {}),
        {
            "per_galaxy_qa": True,
            "cross_arm_overlay": True,
            "cross_tool_comparison": True,
            "summary_grids": True,
            "residual_models": True,
            "sb_profile_scale": "log10",
            "sb_asinh_softening": None,
        },
    )

    execution = {
        "max_parallel_galaxies": int(
            raw.get("execution", {}).get("max_parallel_galaxies", 1)
        ),
        "skip_existing": bool(raw.get("execution", {}).get("skip_existing", True)),
        "dry_run": bool(raw.get("execution", {}).get("dry_run", False)),
        "fail_fast": bool(raw.get("execution", {}).get("fail_fast", False)),
    }
    summary = raw.get("summary", {}) or {}

    return CampaignPlan(
        campaign_name=campaign_name,
        output_root=output_root,
        tools=tools,
        datasets=datasets,
        qa=qa,
        execution=execution,
        summary=summary,
        isoster_harmonic_sweeps=harmonic_sweeps,
        raw=raw,
    )


def _load_tools(
    raw_tools: dict[str, Any], yaml_path: Path, harmonic_sweeps: list[list[int]]
) -> dict[str, ToolPlan]:
    plans: dict[str, ToolPlan] = {}
    for tool_name, tool_cfg in raw_tools.items():
        tool_cfg = tool_cfg or {}
        enabled = bool(tool_cfg.get("enabled", False))
        arms_file_raw = tool_cfg.get("arms_file")
        if arms_file_raw is None:
            raise ValueError(f"{yaml_path}: tools.{tool_name} missing 'arms_file'")
        arms_file = _resolve_repo_path(arms_file_raw)
        if enabled and not arms_file.is_file():
            raise FileNotFoundError(
                f"{yaml_path}: tools.{tool_name}.arms_file not found: {arms_file}"
            )
        arms = _load_arms(arms_file) if arms_file.is_file() else {}
        if tool_name == "isoster" and arms:
            arms = _expand_harmonic_arms(arms, harmonic_sweeps)
        selected_arms = tool_cfg.get("select_arms")
        if selected_arms is not None:
            if not isinstance(selected_arms, list) or not all(
                isinstance(arm_id, str) for arm_id in selected_arms
            ):
                raise ValueError(
                    f"{yaml_path}: tools.{tool_name}.select_arms must be a list of strings"
                )
            missing = [arm_id for arm_id in selected_arms if arm_id not in arms]
            if missing:
                raise ValueError(
                    f"{yaml_path}: tools.{tool_name}.select_arms contains unknown arm(s) {missing}"
                )
            arms = {arm_id: arms[arm_id] for arm_id in selected_arms}

        extra = {
            k: v
            for k, v in tool_cfg.items()
            if k not in {"enabled", "arms_file", "select_arms"}
        }
        plans[tool_name] = ToolPlan(
            name=tool_name,
            enabled=enabled,
            arms_file=arms_file,
            arms=arms,
            extra=extra,
        )
    return plans


def _load_arms(arms_file: Path) -> dict[str, dict[str, Any]]:
    with arms_file.open() as handle:
        data = yaml.safe_load(handle) or {}
    arms = data.get("arms", {})
    if not isinstance(arms, dict):
        raise ValueError(f"{arms_file}: top-level 'arms' must be a mapping")
    for arm_id, delta in arms.items():
        if delta is None:
            arms[arm_id] = {}
        elif not isinstance(delta, dict):
            raise ValueError(
                f"{arms_file}: arm '{arm_id}' delta must be a mapping, got {type(delta).__name__}"
            )
    return arms


def _expand_harmonic_arms(
    arms: dict[str, dict[str, Any]], sweeps: list[list[int]]
) -> dict[str, dict[str, Any]]:
    """Emit one ``harm_higher_orders::<joined>`` arm per sweep entry.

    Each emitted arm carries ``harmonic_orders=<entry>``. Entries are
    appended to the arm dict; existing arms are kept.
    """
    out = dict(arms)  # preserve declared order on Python 3.7+
    for entry in sweeps:
        arm_id = "harm_higher_orders::" + "_".join(str(n) for n in entry)
        out[arm_id] = {"harmonic_orders": list(entry)}
    return out


def _load_datasets(
    raw_datasets: dict[str, Any], yaml_path: Path
) -> dict[str, DatasetPlan]:
    plans: dict[str, DatasetPlan] = {}
    for dataset_name, ds_cfg in raw_datasets.items():
        ds_cfg = ds_cfg or {}
        enabled = bool(ds_cfg.get("enabled", False))
        adapter_name = ds_cfg.get("adapter") or dataset_name
        root_raw = ds_cfg.get("root")
        if enabled and root_raw is None:
            raise ValueError(
                f"{yaml_path}: datasets.{dataset_name}.root is required when enabled"
            )
        select = ds_cfg.get("select")
        if select is not None and not isinstance(select, list):
            raise ValueError(
                f"{yaml_path}: datasets.{dataset_name}.select must be a list or null"
            )
        adapter: DatasetAdapter | None = None
        extra_kwargs = {
            k: v for k, v in ds_cfg.items() if k not in {"enabled", "adapter", "select"}
        }
        if enabled:
            adapter_module = _import_adapter_module(adapter_name)
            adapter_cls = getattr(adapter_module, "ADAPTER_CLASS", None)
            if adapter_cls is None:
                raise AttributeError(
                    f"Adapter module '{adapter_module.__name__}' must expose ADAPTER_CLASS"
                )
            # Pass root + any adapter-specific kwargs straight through.
            adapter = adapter_cls(**extra_kwargs)
        plans[dataset_name] = DatasetPlan(
            name=dataset_name,
            enabled=enabled,
            adapter_name=adapter_name,
            adapter=adapter,  # type: ignore[arg-type]
            select=select,
            extra=extra_kwargs,
        )
    return plans


def _import_adapter_module(adapter_name: str):
    module_path = f"benchmarks.exhausted.adapters.{adapter_name}"
    try:
        return importlib.import_module(module_path)
    except ModuleNotFoundError as exc:
        raise ModuleNotFoundError(
            f"Adapter '{adapter_name}' not found (expected module {module_path})"
        ) from exc


def _resolve_repo_path(raw: str | Path) -> Path:
    """Paths in the YAML can be absolute, $HOME-relative, or
    repo-relative (resolved against the isoster repo root)."""
    path = Path(str(raw)).expanduser()
    if path.is_absolute():
        return path
    return (REPO_ROOT / path).resolve()


def _require(mapping: dict[str, Any], key: str, source: Path) -> Any:
    if key not in mapping:
        raise ValueError(f"{source}: required key '{key}' missing")
    return mapping[key]


def _dict_with_bool_defaults(
    raw: dict[str, Any], defaults: dict[str, Any]
) -> dict[str, Any]:
    raw = raw or {}
    out = dict(defaults)
    for key, value in raw.items():
        if isinstance(defaults.get(key), bool):
            out[key] = bool(value)
        else:
            out[key] = value
    return out
