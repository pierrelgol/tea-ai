from __future__ import annotations

import argparse
import json
import os
from dataclasses import asdict, dataclass
from datetime import datetime
from pathlib import Path
import subprocess
import tempfile
import time
from threading import Event, Thread
from typing import Any

import psutil
from tqdm import tqdm

from pipeline_config import PipelineConfig, build_layout, load_pipeline_config


@dataclass(slots=True)
class StageMetrics:
    stage: str
    status: str
    duration_s: float
    cpu_percent_avg: float | None
    cpu_percent_max: float | None
    rss_mb_max: float | None
    gpu_util_percent_avg: float | None
    gpu_mem_mb_max: float | None
    notes: str | None = None


def _try_sample_gpu() -> tuple[float | None, float | None]:
    try:
        proc = subprocess.run(
            [
                "nvidia-smi",
                "--query-gpu=utilization.gpu,memory.used",
                "--format=csv,noheader,nounits",
            ],
            capture_output=True,
            text=True,
            check=True,
        )
        line = proc.stdout.strip().splitlines()[0]
        u, m = [x.strip() for x in line.split(",")[:2]]
        return float(u), float(m)
    except Exception:
        return None, None


def _monitor_process(pid: int, stop: Event, sample_gpu: bool) -> dict[str, Any]:
    cpu_vals: list[float] = []
    rss_vals: list[float] = []
    gpu_u_vals: list[float] = []
    gpu_m_vals: list[float] = []

    try:
        p = psutil.Process(pid)
    except Exception:
        return {}

    while not stop.is_set():
        try:
            procs = [p] + p.children(recursive=True)
            rss = 0
            cpu = 0.0
            for cur in procs:
                try:
                    rss += cur.memory_info().rss
                    cpu += cur.cpu_percent(interval=None)
                except Exception:
                    continue
            rss_vals.append(rss / (1024.0 * 1024.0))
            cpu_vals.append(cpu)
            if sample_gpu:
                gu, gm = _try_sample_gpu()
                if gu is not None:
                    gpu_u_vals.append(gu)
                if gm is not None:
                    gpu_m_vals.append(gm)
        except Exception:
            pass
        time.sleep(0.5)

    return {
        "cpu_avg": (sum(cpu_vals) / len(cpu_vals)) if cpu_vals else None,
        "cpu_max": max(cpu_vals) if cpu_vals else None,
        "rss_max": max(rss_vals) if rss_vals else None,
        "gpu_u_avg": (sum(gpu_u_vals) / len(gpu_u_vals)) if gpu_u_vals else None,
        "gpu_m_max": max(gpu_m_vals) if gpu_m_vals else None,
    }


def _run_stage(name: str, cmd: list[str], *, sample_gpu: bool, cwd: Path) -> StageMetrics:
    start = time.perf_counter()
    proc = subprocess.Popen(cmd, cwd=str(cwd), stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)

    stop = Event()
    monitor_out: dict[str, Any] = {}

    def _runner() -> None:
        nonlocal monitor_out
        monitor_out = _monitor_process(proc.pid, stop, sample_gpu)

    t = Thread(target=_runner, daemon=True)
    t.start()

    output_lines: list[str] = []
    assert proc.stdout is not None
    for line in proc.stdout:
        output_lines.append(line)
    rc = proc.wait()
    stop.set()
    t.join(timeout=2.0)

    end = time.perf_counter()
    status = "ok" if rc == 0 else "error"
    notes = None if rc == 0 else "".join(output_lines[-40:])

    return StageMetrics(
        stage=name,
        status=status,
        duration_s=end - start,
        cpu_percent_avg=monitor_out.get("cpu_avg"),
        cpu_percent_max=monitor_out.get("cpu_max"),
        rss_mb_max=monitor_out.get("rss_max"),
        gpu_util_percent_avg=monitor_out.get("gpu_u_avg"),
        gpu_mem_mb_max=monitor_out.get("gpu_m_max"),
        notes=notes,
    )


def _build_profile_config(base_cfg: PipelineConfig, *, dataset: str, train_epochs: int) -> Path:
    base_cfg_path = base_cfg.config_path
    payload = json.loads(base_cfg_path.read_text(encoding="utf-8"))
    payload["run"]["dataset"] = dataset
    payload["dataset"]["name"] = dataset
    payload["run"]["run_id"] = f"profile-{datetime.utcnow().strftime('%Y%m%d-%H%M%S')}"
    payload["train"]["epochs"] = int(train_epochs)
    payload["train"]["wandb_enabled"] = False
    payload["checks"]["gui"] = False
    payload["review"]["split"] = "val"
    # Persist absolute paths so temp config location does not break path resolution.
    payload["paths"]["dataset_root"] = str(base_cfg.paths["dataset_root"])
    payload["paths"]["artifacts_root"] = str(base_cfg.paths["artifacts_root"])
    payload["paths"]["configs_root"] = str(base_cfg.paths["configs_root"])
    payload["paths"]["targets_source_root"] = str(base_cfg.paths["targets_source_root"])

    tmp = Path(tempfile.gettempdir()) / f"tea-ai-profile-{os.getpid()}.json"
    tmp.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    return tmp


def main() -> None:
    parser = argparse.ArgumentParser(description="Profile the tea-ai end-to-end pipeline")
    parser.add_argument("--config", type=Path, default=Path("config.json"))
    parser.add_argument("--dataset", default=None)
    parser.add_argument("--train-epochs", type=int, default=None)
    parser.add_argument("--report-dir", type=Path, default=None)
    parser.add_argument("--no-gpu-sampling", action="store_true")
    args = parser.parse_args()

    base_cfg = load_pipeline_config(args.config.resolve())
    profile_dataset = str(args.dataset or base_cfg.profile.get("dataset", "coco128"))
    profile_epochs = int(args.train_epochs or base_cfg.profile.get("train_epochs", 50))
    sample_gpu = not bool(args.no_gpu_sampling)
    if args.no_gpu_sampling:
        sample_gpu = False
    elif "enable_gpu_sampling" in base_cfg.profile:
        sample_gpu = bool(base_cfg.profile.get("enable_gpu_sampling", True))

    root = Path.cwd().resolve()
    profile_cfg = _build_profile_config(base_cfg, dataset=profile_dataset, train_epochs=profile_epochs)
    cfg = load_pipeline_config(profile_cfg)
    layout = build_layout(
        artifacts_root=cfg.paths["artifacts_root"],
        model_key=str(cfg.run["model_key"]),
        run_id=str(cfg.run["run_id"]),
    )

    stages: list[StageMetrics] = []

    execution_plan: list[tuple[str, list[str] | None]] = [
        ("fetch-dataset", ["uv", "run", "dataset-fetcher", "--config", str(profile_cfg)]),
        ("label-targets", None),
        ("generate-dataset", ["uv", "run", "dataset-generator", "--config", str(profile_cfg)]),
        ("check-dataset", ["uv", "run", "augment-checker", "--config", str(profile_cfg)]),
        ("train", ["uv", "run", "detector-train", "--config", str(profile_cfg)]),
        ("infer", ["uv", "run", "detector-infer", "--config", str(profile_cfg)]),
        ("eval", ["uv", "run", "detector-grader", "--config", str(profile_cfg)]),
        ("review", None),
    ]
    encountered_error = False
    with tqdm(total=len(execution_plan), desc="pipeline-profile", unit="stage", dynamic_ncols=True) as progress:
        for stage_name, cmd in execution_plan:
            progress.set_postfix_str(f"stage={stage_name}")
            if encountered_error:
                stage_metrics = StageMetrics(
                    stage=stage_name,
                    status="skipped",
                    duration_s=0.0,
                    cpu_percent_avg=None,
                    cpu_percent_max=None,
                    rss_mb_max=None,
                    gpu_util_percent_avg=None,
                    gpu_mem_mb_max=None,
                    notes="skipped due to previous stage error",
                )
                stages.append(stage_metrics)
                tqdm.write(f"[skipped] {stage_name}: previous stage failed")
                progress.update(1)
                continue
            if cmd is None:
                note = "manual GUI step omitted from automated profile"
                if stage_name == "review":
                    note = "GUI reviewer skipped in automated profile"
                stage_metrics = StageMetrics(
                    stage=stage_name,
                    status="manual",
                    duration_s=0.0,
                    cpu_percent_avg=None,
                    cpu_percent_max=None,
                    rss_mb_max=None,
                    gpu_util_percent_avg=None,
                    gpu_mem_mb_max=None,
                    notes=note,
                )
                stages.append(stage_metrics)
                tqdm.write(f"[manual] {stage_name}: {note}")
                progress.update(1)
                continue

            tqdm.write(f"[run] {stage_name}")
            stage_metrics = _run_stage(stage_name, cmd, sample_gpu=sample_gpu, cwd=root)
            stages.append(stage_metrics)
            progress.update(1)
            progress.set_postfix_str(f"stage={stage_name}, status={stage_metrics.status}")
            tqdm.write(f"[{stage_metrics.status}] {stage_name}: {stage_metrics.duration_s:.2f}s")
            if stage_metrics.status == "error":
                encountered_error = True

    grade_metrics: dict[str, Any] = {}
    summary_jsons = list((layout.grade_root / "reports").glob("grade_report_*.json"))
    if summary_jsons:
        latest = sorted(summary_jsons, key=lambda p: p.stat().st_mtime, reverse=True)[0]
        payload = json.loads(latest.read_text(encoding="utf-8"))
        agg = payload.get("aggregate", {})
        run_det = agg.get("run_detection", {})
        grade_metrics = {
            "run_grade_0_100": agg.get("run_grade_0_100"),
            "precision_proxy": run_det.get("precision_proxy"),
            "recall_proxy": run_det.get("recall_proxy"),
            "miss_rate_proxy": run_det.get("miss_rate_proxy"),
        }

    total_time = float(sum(s.duration_s for s in stages))
    report = {
        "timestamp_utc": datetime.utcnow().isoformat() + "Z",
        "config": str(profile_cfg),
        "dataset": profile_dataset,
        "train_epochs": profile_epochs,
        "run_root": str(layout.run_root),
        "total_duration_s": total_time,
        "stages": [asdict(s) for s in stages],
        "quality_metrics": grade_metrics,
    }

    report_dir = args.report_dir or (layout.run_root / "profile")
    report_dir.mkdir(parents=True, exist_ok=True)
    report_json = report_dir / "profile_report.json"
    report_md = report_dir / "profile_summary.md"
    report_json.write_text(json.dumps(report, indent=2), encoding="utf-8")

    lines = [
        "# Pipeline Profile Summary",
        "",
        f"- dataset: `{profile_dataset}`",
        f"- train_epochs: `{profile_epochs}`",
        f"- run_root: `{layout.run_root}`",
        f"- total_duration_s: `{total_time:.3f}`",
        "",
        "## Stages",
    ]
    for s in stages:
        lines.append(
            f"- {s.stage}: status={s.status}, duration_s={s.duration_s:.3f}, "
            f"cpu_avg={s.cpu_percent_avg}, cpu_max={s.cpu_percent_max}, "
            f"rss_mb_max={s.rss_mb_max}, gpu_avg={s.gpu_util_percent_avg}, gpu_mem_mb_max={s.gpu_mem_mb_max}"
        )
    if grade_metrics:
        lines.extend([
            "",
            "## Quality",
            f"- run_grade_0_100: {grade_metrics.get('run_grade_0_100')}",
            f"- precision_proxy: {grade_metrics.get('precision_proxy')}",
            f"- recall_proxy: {grade_metrics.get('recall_proxy')}",
            f"- miss_rate_proxy: {grade_metrics.get('miss_rate_proxy')}",
        ])
    report_md.write_text("\n".join(lines) + "\n", encoding="utf-8")

    print("status: ok")
    print(f"profile_report: {report_json}")
    print(f"profile_summary: {report_md}")


if __name__ == "__main__":
    main()
