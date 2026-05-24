#!/usr/bin/env python3
"""
Parse TTFT_TRACE log files and compute per-stage timing statistics.

Usage:
    python parse_ttft_log.py <logfile> [--detail] [--csv <output.csv>]
                               [--top N] [--request-id <id>]

Examples:
    # Print summary table with averages
    python parse_ttft_log.py vllm.log

    # Also print per-request detail for the top-10 slowest requests
    python parse_ttft_log.py vllm.log --detail --top 10

    # Show one specific request's timing breakdown
    python parse_ttft_log.py vllm.log --request-id chatcmpl-abc123

    # Export per-request CSV
    python parse_ttft_log.py vllm.log --csv ttft_report.csv
"""

from __future__ import annotations

import csv
import re
import sys
from collections import defaultdict
from typing import Any

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

# Regex for a TTFT_TRACE line:
#   [TTFT_TRACE] request_id=<id> stage=<stage> timestamp=<float>
_TRACE_RE = re.compile(
    r"\[TTFT_TRACE\]\s+request_id=(\S+)\s+stage=(\S+)\s+timestamp=([\d.]+)"
)

# Stages that use the _start / _end naming convention (= paired).
PAIRED_STAGES: set[str] = {
    "render_chat",
    "tokenization",
    "tokenization_async",
    "input_processing",
    "engine_core_dispatch",
    "http_download",
    "http_download_async",
    "http_download_phase4",
    "image_decode",
    "image_decode_async",
    "hf_multimod  al_processor",
    "worker_forward_pass",
    "worker_model_exec",
    "mooncake_kv_load",
    "queue_wait",
    "scheduler_exec",
}

# Derived stages: computed from two point-event timestamps.
# ``{derived_stage: (start_point_event, end_point_event)}``
DERIVED_PAIRS: dict[str, tuple[str, str]] = {
    "queue_wait": ("scheduler_enqueue_waiting", "scheduler_pickup_from_waiting"),
    "scheduler_exec": ("scheduler_pickup_from_waiting", "scheduler_scheduled"),
}

# Inter-stage gaps: idle time between one stage's end and the next's start.
# Computed as end_event → start_event diff.  Negative = overlap, ≈0 = seamless.
GAP_STAGES: list[tuple[str, str, str]] = [
    ("gap_render_to_dispatch", "render_chat_end", "api_server_dispatch"),
    ("gap_input_to_dispatch", "input_processing_end", "engine_core_dispatch_start"),
    ("gap_enqueue_to_pickup", "scheduler_enqueue_waiting", "scheduler_pickup_from_waiting"),
    ("gap_scheduled_to_work", "scheduler_scheduled", "worker_model_exec_start"),
]

_GAP_CN: dict[str, str] = {
    "gap_render_to_dispatch": "渲染完成→分发 间隔(D侧)",
    "gap_input_to_dispatch": "输入处理→分发 间隔(D侧)",
    "gap_enqueue_to_pickup": "入队→取出 队列等待(P侧)",
    "gap_scheduled_to_work": "调度完成→Worker执行 间隔(P侧)",
}

TTFT_STAGE_DESCRIPTION = """
╔══════════════════════════════════════════════════════════════════════════════╗
║                           阶段字段含义说明                                     ║
╠══════════════════════════════════════════════════════════════════════════════╣
║ API Server 阶段                                                              ║
║   api_request_start       — API 服务收到请求的瞬间                            ║
║   render_chat             — 渲染(图片下载+分词+HF预处理)总耗时                  ║
║   tokenization            — tokenizer.encode() 文本转token                    ║
║   http_download(_async)   — HTTP 图片下载(每张独立)                           ║
║   image_decode(_async)    — 图片解码(PIL)                                     ║
║   hf_multimodal_processor — HF 多模态处理器                                   ║
║   api_server_dispatch     — API 完成预处理, 提交给 Engine                      ║
║ Engine 入口                                                                   ║
║   engine_generate_start   — AsyncLLM.generate() 入口                          ║
║   input_processing        — 参数校验/构造 EngineCoreRequest                    ║
║   engine_core_dispatch    — 跨进程 ZMQ 发送到 EngineCore                       ║
║ EngineCore + 调度                                                             ║
║   engine_core_add_request — EngineCore 收到请求                               ║
║   scheduler_enqueue_waiting — 请求放入等待队列                                  ║
║   scheduler_pickup_from_waiting — 调度器取出请求                                ║
║   scheduler_scheduled     — 调度完成(分配 KV cache 等)                         ║
║   scheduler_encoder_scheduled — 图像 encoder 调度完成                           ║
║   queue_wait              — 队列等待(入队→取出)                                ║
║   scheduler_exec          — 调度处理(取出→完成)                                ║
║ Worker 执行                                                                   ║
║   worker_model_exec       — 一次 execute_model 总耗时(ViT+forward)            ║
║   worker_forward_pass     — LLM 前向传播(model.forward)耗时                   ║
║   encoder_runner_vit      — ViT 图像编码(batch级,无法区分请求)                 ║
║ 输出                                                                          ║
║   first_token_output      — OutputProcessor 检测到首 token                    ║
║   first_token_api_yield   — 首 token 返回客户端                                ║
║ 间隔(派生, 前后阶段间的空闲/传输时间)                                            ║
║   gap_render_to_dispatch  — 渲染完成→分发(D侧, 同进程)                         ║
║   gap_input_to_dispatch   — 输入处理→分发(D侧, 同进程)                          ║
║   gap_scheduled_to_work   — 调度→Worker 执行排队(P侧, 同进程)                   ║
║ 统计指标                                                                       ║
║   count — 有效样本数    avg — 平均值    p50 — 中位数(50%)                      ║
║   p99 — 99分位    min/max — 最小/最大值                                       ║
║   TTFT — Time To First Token = api_request_start → first_token_api_yield     ║
╚══════════════════════════════════════════════════════════════════════════════╝
"""

# Point-in-time events (single timestamp, no _end counterpart).
# Listed in pipeline order.
POINT_STAGES: list[str] = [
    "api_request_start",
    "render_chat_start",
    "api_server_dispatch",
    "engine_generate_start",
    "engine_core_add_request",
    "scheduler_add_request",
    "scheduler_enqueue_waiting",
    "scheduler_pickup_from_waiting",
    "scheduler_scheduled",
    "scheduler_encoder_scheduled",
    "first_token_output",
    "first_token_api_yield",
]

# Batch-level stages (request_id == "batch").  Reported separately.
BATCH_STAGES: set[str] = {
    "engine_core_step_start",
    "engine_core_step_end",
    "model_runner_mm_encoder_start",
    "model_runner_mm_encoder_end",
    "encoder_runner_vit_start",
    "encoder_runner_vit_end",
}

# Temporary / anonymous IDs used before the real request_id is known.
TEMP_ID_PREFIXES: tuple[str, ...] = ("media_connector", "renderer-mm-")

# Requests excluded from per-request statistics.
EXCLUDED_IDS: frozenset[str] = frozenset({"batch"})


# The patch emits ``id_mapped_from_<raw_id>`` events with the
# ``chatcmpl-xxx`` request_id to link API-server and engine-side IDs.
_ID_MAPPED_PREFIX = "id_mapped_from_"


def _build_id_map(entries: list[dict[str, Any]]) -> dict[str, str]:
    """Build a mapping from raw API-server request_id → engine-side
    ``chatcmpl-xxx`` request_id using ``id_mapped_from_`` events."""
    id_map: dict[str, str] = {}
    for e in entries:
        stage = e["stage"]
        if stage.startswith(_ID_MAPPED_PREFIX):
            raw_id = stage[len(_ID_MAPPED_PREFIX):]
            chatcmpl_id = e["request_id"]  # e.g. "chatcmpl-abc123"
            id_map[raw_id] = chatcmpl_id
    return id_map


def _normalize_id(req_id: str, *, id_map: dict[str, str] | None = None) -> str:
    """Map a raw or chatcmpl request_id to a canonical key."""
    if id_map is None:
        return req_id
    # If this is a raw ID known to the map, return the chatcmpl version
    if req_id in id_map:
        return id_map[req_id]
    # If this is a chatcmpl ID, return as-is
    return req_id


def _preferred_id(raw_ids: set[str]) -> str:
    """Pick the most readable ID."""
    # Prefer longer IDs (usually chatcmpl-xxx format)
    return sorted(raw_ids, key=len, reverse=True)[0]


# ---------------------------------------------------------------------------
# Parsing helpers
# ---------------------------------------------------------------------------

def _is_temp_id(req_id: str) -> bool:
    return req_id.startswith(TEMP_ID_PREFIXES)


def _is_real_request(req_id: str) -> bool:
    return req_id not in EXCLUDED_IDS and not _is_temp_id(req_id)


def parse_log(filepath: str) -> list[dict[str, Any]]:
    """Parse a log file and return a list of trace entry dicts.

    Each entry::
        {"request_id": str, "stage": str, "timestamp": float}
    """
    entries: list[dict[str, Any]] = []
    with open(filepath, encoding="utf-8") as f:
        for line in f:
            m = _TRACE_RE.search(line)
            if not m:
                continue
            entries.append({
                "request_id": m.group(1),
                "stage": m.group(2),
                "timestamp": float(m.group(3)),
            })
    return entries


def group_by_request(
    entries: list[dict[str, Any]],
) -> tuple[
    dict[str, list[dict[str, Any]]],
    dict[str, str],
]:
    """Group entries by normalized request_id.

    Returns
    -------
    groups : dict
        ``{normalized_id: [events_sorted_by_ts]}``
    id_map : dict
        ``{normalized_id: display_id}``  — maps back to the preferred
        (``chatcmpl-xxx``) form for display.
    """
    # Build the mapping from raw → chatcmpl IDs via id_mapped_from_ events
    id_map_raw_to_cc = _build_id_map(entries)

    # First pass: collect all raw IDs that belong to each normalized id
    raw_ids_by_norm: dict[str, set[str]] = defaultdict(set)
    for e in entries:
        norm = _normalize_id(e["request_id"], id_map=id_map_raw_to_cc)
        raw_ids_by_norm[norm].add(e["request_id"])

    # Build display_id_map: normalized → preferred display id
    display_id_map: dict[str, str] = {
        norm: _preferred_id(raw_ids) for norm, raw_ids in raw_ids_by_norm.items()
    }

    # Second pass: group by normalized id
    groups: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for e in entries:
        norm = _normalize_id(e["request_id"], id_map=id_map_raw_to_cc)
        groups[norm].append(e)

    for req_id in groups:
        groups[req_id].sort(key=lambda e: e["timestamp"])
    return dict(groups), display_id_map


# ---------------------------------------------------------------------------
# Duration computation
# ---------------------------------------------------------------------------

def compute_request_durations(
    events: list[dict[str, Any]],
) -> dict[str, float | None]:
    """Given a sorted list of events for one request_id, compute per-stage
    durations and return a dict of ``{stage: duration_seconds}``.

    Paired stages (e.g. ``render_chat``) are derived from their
    ``_start`` / ``_end`` timestamps.  Point events are returned as-is
    (their absolute timestamp) and their relative offsets are computed
    later by the caller.
    """
    # Build lookup for individual events
    event_map: dict[str, float] = {}
    for e in events:
        event_map[e["stage"]] = e["timestamp"]

    result: dict[str, float | None] = {}

    # Paired stages: diff between _start and _end
    for base in PAIRED_STAGES:
        s_key = f"{base}_start"
        e_key = f"{base}_end"
        if s_key in event_map and e_key in event_map:
            result[base] = event_map[e_key] - event_map[s_key]
        elif s_key in event_map:
            result[base] = None  # had start but not end
        # else: stage not present for this request

    # Derived pairs: computed from two point-event timestamps
    for derived, (pt_start, pt_end) in DERIVED_PAIRS.items():
        if pt_start in event_map and pt_end in event_map:
            result[derived] = event_map[pt_end] - event_map[pt_start]

    # Inter-stage gaps: idle time between consecutive stages
    for gap_name, prev_end, next_start in GAP_STAGES:
        if prev_end in event_map and next_start in event_map:
            result[gap_name] = event_map[next_start] - event_map[prev_end]

    # Point events: store absolute timestamp
    for pt in POINT_STAGES:
        if pt in event_map:
            result[pt] = event_map[pt]

    return result


# ---------------------------------------------------------------------------
# Aggregation and statistics
# ---------------------------------------------------------------------------

def percentile(sorted_vals: list[float], pct: float) -> float:
    """Return the pct-th percentile (0..100) of a sorted list."""
    if not sorted_vals:
        return 0.0
    idx = (len(sorted_vals) - 1) * pct / 100.0
    lo = int(idx)
    hi = min(lo + 1, len(sorted_vals) - 1)
    frac = idx - lo
    return sorted_vals[lo] * (1 - frac) + sorted_vals[hi] * frac


def aggregate_stage_durations(
    per_request: dict[str, dict[str, float | None]],
) -> dict[str, dict[str, float]]:
    """Aggregate per-request stage durations into summary stats.

    Returns ``{stage: {avg, min, max, p50, p99, count, missing}}``.
    """
    # Collect all values per stage
    stage_values: dict[str, list[float]] = defaultdict(list)
    for req_id, stages in per_request.items():
        for stage, val in stages.items():
            if val is not None and not stage.endswith("_start"):
                # For point events, we compute relative offsets later.
                # Here we only aggregate paired durations.
                if stage in PAIRED_STAGES:
                    stage_values[stage].append(val)

    stats: dict[str, dict[str, float]] = {}
    for stage in sorted(stage_values.keys()):
        vals = stage_values[stage]
        sorted_vals = sorted(vals)
        stats[stage] = {
            "count": len(vals),
            "avg": sum(vals) / len(vals) if vals else 0,
            "min": min(vals) if vals else 0,
            "max": max(vals) if vals else 0,
            "p50": percentile(sorted_vals, 50),
            "p99": percentile(sorted_vals, 99),
        }

    return stats


def aggregate_gaps(
    per_request: dict[str, dict[str, float | None]],
) -> dict[str, dict[str, float]]:
    gap_names = {g[0] for g in GAP_STAGES}
    gap_vals: dict[str, list[float]] = defaultdict(list)
    for req_id, stages in per_request.items():
        for name in gap_names:
            val = stages.get(name)
            if val is not None:
                gap_vals[name].append(val)
    stats: dict[str, dict[str, float]] = {}
    for name in sorted(gap_vals.keys()):
        vals = gap_vals[name]
        sv = sorted(vals)
        stats[name] = {
            "count": len(vals),
            "avg": sum(vals) / len(vals) if vals else 0,
            "min": min(vals) if vals else 0,
            "max": max(vals) if vals else 0,
            "p50": percentile(sv, 50),
            "p99": percentile(sv, 99),
        }
    return stats


def aggregate_point_offsets(
    per_request: dict[str, dict[str, float | None]],
) -> dict[str, dict[str, float]]:
    """Compute relative time offsets for point events.

    Each point event's offset is ``event_ts - api_request_start_ts``.
    """
    stage_values: dict[str, list[float]] = defaultdict(list)
    for req_id, stages in per_request.items():
        ref = stages.get("api_request_start")
        if ref is None:
            continue
        for stage, ts in stages.items():
            if ts is None or stage in PAIRED_STAGES:
                continue
            if stage.endswith("_start"):
                continue
            # ts is absolute timestamp for point event
            offset = ts - ref  # type: ignore[operator]
            stage_values[stage].append(offset)

    stats: dict[str, dict[str, float]] = {}
    for stage in sorted(stage_values.keys()):
        vals = stage_values[stage]
        sorted_vals = sorted(vals)
        stats[stage] = {
            "count": len(vals),
            "avg_ms": (sum(vals) / len(vals) * 1000) if vals else 0,
            "min_ms": (min(vals) * 1000) if vals else 0,
            "max_ms": (max(vals) * 1000) if vals else 0,
            "p50_ms": percentile(sorted_vals, 50) * 1000,
            "p99_ms": percentile(sorted_vals, 99) * 1000,
        }
    return stats


# ---------------------------------------------------------------------------
# Total TTFT
# ---------------------------------------------------------------------------

def compute_ttft_stats(
    per_request: dict[str, dict[str, float | None]],
) -> dict[str, float]:
    """Compute overall TTFT (api_request_start → first_token_api_yield)."""
    vals: list[float] = []
    for req_id, stages in per_request.items():
        start = stages.get("api_request_start")
        end = stages.get("first_token_api_yield")
        if start is not None and end is not None:
            vals.append(end - start)  # type: ignore[operator]

    if not vals:
        return {}
    sv = sorted(vals)
    return {
        "count": len(vals),
        "avg_ms": (sum(vals) / len(vals) * 1000),
        "min_ms": min(vals) * 1000,
        "max_ms": max(vals) * 1000,
        "p50_ms": percentile(sv, 50) * 1000,
        "p99_ms": percentile(sv, 99) * 1000,
    }


# ---------------------------------------------------------------------------
# Rendering
# ---------------------------------------------------------------------------

def _ms(v: float) -> str:
    if v > 1000:
        return f"{v / 1000:.2f}s"
    return f"{v:.1f}ms"


def print_summary(
    per_request: dict[str, dict[str, float | None]],
    stage_stats: dict[str, dict[str, float]],
    point_stats: dict[str, dict[str, float]],
    ttft_stats: dict[str, float],
) -> None:
    """Print a human-readable summary table."""

    req_count = sum(
        1 for _ in per_request.values() if "api_request_start" in _
    )

    print("=" * 90)
    print(f"  TTFT Trace Analysis  —  {req_count} requests")
    print("=" * 90)

    # ---- Overall TTFT ----
    if ttft_stats:
        print()
        print("  Total TTFT (api_request_start → first_token_api_yield)")
        print("  ─" * 40)
        print(f"    count : {ttft_stats['count']}")
        print(f"    avg   : {_ms(ttft_stats['avg_ms'])}")
        print(f"    min   : {_ms(ttft_stats['min_ms'])}")
        print(f"    max   : {_ms(ttft_stats['max_ms'])}")
        print(f"    p50   : {_ms(ttft_stats['p50_ms'])}")
        print(f"    p99   : {_ms(ttft_stats['p99_ms'])}")

    # ---- Paired stage durations ----
    if stage_stats:
        print()
        print(f"  {'Stage':35s} {'count':>6s} {'avg':>10s} {'p50':>10s} "
              f"{'p99':>10s} {'min':>10s} {'max':>10s}")
        print("  " + "─" * 89)
        for stage, s in stage_stats.items():
            print(
                f"  {stage:35s} {s['count']:>6d} "
                f"{_ms(s['avg'] * 1000):>10s} "
                f"{_ms(s['p50'] * 1000):>10s} "
                f"{_ms(s['p99'] * 1000):>10s} "
                f"{_ms(s['min'] * 1000):>10s} "
                f"{_ms(s['max'] * 1000):>10s}"
            )

    # ---- Inter-stage gaps ----
    gap_stats = aggregate_gaps(per_request)
    if gap_stats:
        print()
        print("  Inter-stage gaps (idle/transfer time)")
        print(f"  {'Gap':35s} {'count':>6s} {'avg':>10s} {'p50':>10s} "
              f"{'p99':>10s} {'min':>10s} {'max':>10s}")
        print("  " + "─" * 89)
        for stage, s in gap_stats.items():
            cn = _GAP_CN.get(stage, stage)
            print(
                f"  {cn:35s} {s['count']:>6d} "
                f"{_ms(s['avg'] * 1000):>10s} "
                f"{_ms(s['p50'] * 1000):>10s} "
                f"{_ms(s['p99'] * 1000):>10s} "
                f"{_ms(s['min'] * 1000):>10s} "
                f"{_ms(s['max'] * 1000):>10s}"
            )

    # ---- Point event offsets from api_request_start ----
    if point_stats:
        print()
        print("  Point event offsets from api_request_start")
        print(f"  {'Stage':35s} {'count':>6s} {'avg':>10s} {'p50':>10s} "
              f"{'p99':>10s} {'min(M)':>10s} {'max(M)':>10s}")
        print("  " + "─" * 89)
        for stage, s in point_stats.items():
            print(
                f"  {stage:35s} {s['count']:>6d} "
                f"{_ms(s['avg_ms']):>10s} "
                f"{_ms(s['p50_ms']):>10s} "
                f"{_ms(s['p99_ms']):>10s} "
                f"{_ms(s['min_ms']):>10s} "
                f"{_ms(s['max_ms']):>10s}"
            )

    print()
    print("=" * 90)


def print_batch_stats(
    entries: list[dict[str, Any]],
) -> None:
    """Print batch-level timing info (engine_core_step, model runner, etc.)."""
    batch_entries = [e for e in entries if e["request_id"] == "batch"]
    if not batch_entries:
        return
    grouped, _ = group_by_request(batch_entries)
    batch_events = grouped.get("batch", [])
    event_map: dict[str, list[float]] = defaultdict(list)
    by_step: dict[int, dict[str, float]] = {}
    step_no = 0

    for e in sorted(batch_events, key=lambda x: x["timestamp"]):
        if e["stage"] == "engine_core_step_start":
            step_no += 1
            by_step.setdefault(step_no, {})["step_start"] = e["timestamp"]
        elif e["stage"] == "engine_core_step_end":
            by_step.setdefault(step_no, {})["step_end"] = e["timestamp"]
        else:
            event_map[e["stage"]].append(e["timestamp"])

    # Compute step duration
    step_durations: list[float] = []
    for step in by_step.values():
        if "step_start" in step and "step_end" in step:
            step_durations.append(step["step_end"] - step["step_start"])

    print()
    print("  Batch-level stages")
    print("  ─" * 40)
    if step_durations:
        sv = sorted(step_durations)
        print(f"    engine_core_step  count={len(sv)}  "
              f"avg={_ms(sum(sv)/len(sv)*1000)}  "
              f"p50={_ms(percentile(sv,50)*1000)}  "
              f"p99={_ms(percentile(sv,99)*1000)}")

    paired_batch = [
        ("model_runner_mm_encoder", "mm_encoder"),
        ("encoder_runner_vit", "vit_forward"),
    ]
    for base, label in paired_batch:
        starts = event_map.get(f"{base}_start", [])
        ends = event_map.get(f"{base}_end", [])
        durations: list[float] = []
        for s, e in zip(starts, ends):
            durations.append(e - s)
        if durations:
            sv = sorted(durations)
            print(
                f"    {base:30s} count={len(sv):>4d}  "
                f"avg={_ms(sum(sv) / len(durations) * 1000):>10s}  "
                f"p50={_ms(percentile(sv, 50) * 1000):>10s}  "
                f"p99={_ms(percentile(sv, 99) * 1000):>10s}"
            )


def print_temp_stats(
    entries: list[dict[str, Any]],
) -> None:
    """Print statistics for temporary-ID stages (media download, HF processor)."""
    temp_entries = [e for e in entries if _is_temp_id(e["request_id"])]
    if not temp_entries:
        return
    grouped, _ = group_by_request(temp_entries)
    all_durations: dict[str, list[float]] = defaultdict(list)

    for req_id, events in grouped.items():
        event_map = {e["stage"]: e["timestamp"] for e in events}
        for base in PAIRED_STAGES:
            s_key = f"{base}_start"
            e_key = f"{base}_end"
            if s_key in event_map and e_key in event_map:
                all_durations[base].append(event_map[e_key] - event_map[s_key])

    if not all_durations:
        return

    print()
    print("  Anonymous stages (media_connector / renderer-mm-xxx)")
    print("  ─" * 40)
    for stage, vals in sorted(all_durations.items()):
        sv = sorted(vals)
        print(
            f"    {stage:30s} count={len(sv):>4d}  "
            f"avg={_ms(sum(sv) / len(sv) * 1000):>10s}  "
            f"p50={_ms(percentile(sv, 50) * 1000):>10s}  "
            f"p99={_ms(percentile(sv, 99) * 1000):>10s}"
        )


def print_request_detail(
    req_id: str,
    stages: dict[str, float | None],
) -> None:
    """Print a timeline breakdown for one request."""
    ref = stages.get("api_request_start")
    print(f"\n  Request: {req_id}")
    print(f"  {'Stage':35s} {'duration':>10s}  {'offset':>10s}")
    print("  " + "─" * 60)

    display_order = [
        # paired stages
        ("render_chat", True),
        ("tokenization", True),
        ("tokenization_async", True),
        ("http_download", True),
        ("http_download_async", True),
        ("http_download_phase4", True),
        ("image_decode", True),
        ("image_decode_async", True),
        ("hf_multimodal_processor", True),
        ("input_processing", True),
        ("engine_core_dispatch", True),
        # point events
        ("api_request_start", False),
        ("api_server_dispatch", False),
        ("engine_generate_start", False),
        ("engine_core_add_request", False),
        ("scheduler_add_request", False),
        ("scheduler_enqueue_waiting", False),
        ("scheduler_enqueue_waiting", False),
        ("scheduler_pickup_from_waiting", False),
        ("queue_wait", True),
        ("scheduler_scheduled", False),
        ("scheduler_exec", True),
        ("scheduler_encoder_scheduled", False),
        ("worker_model_exec", True),
        ("worker_forward_pass", True),
        ("first_token_output", False),
        ("first_token_api_yield", False),
        ("mooncake_kv_load", True),
        ("gap_render_to_dispatch", True),
        ("gap_input_to_dispatch", True),
        ("gap_scheduled_to_work", True),
    ]

    for stage, is_duration in display_order:
        if stage in stages:
            display_name = _GAP_CN.get(stage, stage) if stage.startswith("gap_") else stage
            val = stages[stage]
            if val is None:
                print(f"  {display_name:35s} {'(missing)':>10s}")
            elif is_duration:
                offset_str = ""
                if ref is not None and stage in PAIRED_STAGES:
                    pass  # paired duration, no offset
                print(
                    f"  {display_name:35s} {_ms(val * 1000):>10s}"
                )
            else:
                # point event
                if ref is not None:
                    offset = val - ref  # type: ignore[operator]
                    print(
                        f"  {display_name:35s} {'—':>10s}  "
                        f"{_ms(offset * 1000):>10s}"
                    )

    # Total TTFT
    if "api_request_start" in stages and "first_token_api_yield" in stages:
        total = (stages["first_token_api_yield"]  # type: ignore[operator]
                 - stages["api_request_start"])  # type: ignore[operator]
        print("  " + "─" * 60)
        print(f"  {'TTFT (total)':35s} {_ms(total * 1000):>10s}")


# ---------------------------------------------------------------------------
# CSV export
# ---------------------------------------------------------------------------

def export_csv(
    per_request: dict[str, dict[str, float | None]],
    filepath: str,
) -> None:
    """Export per-request durations to a CSV file."""
    all_stages: set[str] = set()
    for stages in per_request.values():
        all_stages.update(stages.keys())
    columns = ["request_id"] + sorted(all_stages)

    with open(filepath, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(columns)
        for req_id, stages in per_request.items():
            row = [req_id]
            for col in columns[1:]:
                val = stages.get(col)
                if val is None:
                    row.append("")
                else:
                    row.append(round(val, 6))
            writer.writerow(row)

    print(f"\n  CSV exported to: {filepath}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

# ---------------------------------------------------------------------------
#  Timeline visualization
# ---------------------------------------------------------------------------

# Pipeline order for the timeline: (stage_name, is_duration, label)
_TIMELINE_ORDER: list[tuple[str, bool, str]] = [
    ("api_request_start", False, "API入口"),
    ("render_chat", True, "渲染(下载+分词+HF)"),
    ("gap_render_to_dispatch", True, "> 分发"),
    ("input_processing", True, "输入处理"),
    ("engine_core_dispatch", True, "ZMQ发送"),
    ("queue_wait", True, "队列等待"),
    ("scheduler_exec", True, "调度处理"),
    ("gap_scheduled_to_work", True, "> Worker排队"),
    ("worker_model_exec", True, "Worker执行"),
    ("worker_forward_pass", True, "  LLM前向"),
    ("gap_work_to_output", True, "> 结果回传"),
    ("first_token_api_yield", False, "首Token返回"),
]


def _bar(width: int, char: str = "█") -> str:
    return char * max(width, 0)


def print_timeline(
    per_request: dict[str, dict[str, float | None]],
    ttft_stats: dict[str, float],
) -> None:
    """Print a visual time-axis chart of the average request pipeline."""
    if not ttft_stats:
        return

    total_ms = ttft_stats.get("avg_ms", 0)
    if total_ms <= 0:
        return

    # Collect average values for each stage across all requests
    avg_vals: dict[str, float] = {}

    # Average durations
    for stage_name, is_dur, _ in _TIMELINE_ORDER:
        vals: list[float] = []
        for stages in per_request.values():
            v = stages.get(stage_name)
            if v is not None:
                vals.append(v)
        if vals:
            avg_vals[stage_name] = sum(vals) / len(vals)
        else:
            avg_vals[stage_name] = 0

    # Build timeline segments
    bar_width = 80
    scale = bar_width / total_ms if total_ms > 0 else 0

    print()
    print("  " + "═" * 90)
    print("                         Average Pipeline Timeline")
    print("  " + "═" * 90)
    print()
    print(f"  Total TTFT: {_ms(total_ms)}  (scale: 1 char ≈ {total_ms / bar_width:.1f}ms)")
    print()

    # Build a linear timeline with segments
    timeline_segments: list[tuple[str, float, str]] = []  # (label, width_chars, kind)

    for stage_name, is_dur, label in _TIMELINE_ORDER:
        val_ms = avg_vals.get(stage_name, 0) * 1000
        if val_ms <= 0.01 and is_dur:
            continue  # skip trivial gaps/stages
        w = int(val_ms * scale) if val_ms > 0 else 0
        if w > 0:
            if is_dur:
                timeline_segments.append((label, w, "dur"))
            else:
                timeline_segments.append((label, 1, "pt"))

    # Draw bar
    parts: list[str] = []

    for label, w, kind in timeline_segments:
        if kind == "dur":
            bar_char = "█"
        else:
            bar_char = "▌"
        parts.append(_bar(w, bar_char))
        pos += w

    bar_line = "  |" + "".join(parts) + "|"
    print(bar_line)

    # Draw time markers
    marker_positions: list[tuple[int, str]] = []
    pos = 1
    for label, w, kind in timeline_segments:
        mid = pos + w // 2
        short = label[:10]
        marker_positions.append((mid, short))
        pos += w

    # Simple label line
    label_line = "  ｜"
    for mid, short in marker_positions:
        # Place label at segment midpoint
        pad = mid - len(label_line)
        if pad > 0:
            label_line += " " * (pad - 1) + short
    print(label_line)

    # Legend
    print()
    print(f"  ██ = 耗时阶段    ▌ = 时间点    ")


# ---------------------------------------------------------------------------
#  Main
# ---------------------------------------------------------------------------

def main() -> None:
    import argparse

    parser = argparse.ArgumentParser(
        description="Parse TTFT_TRACE logs and compute per-stage statistics"
    )
    parser.add_argument("logfile", help="Path to the log file")
    parser.add_argument(
        "--detail",
        action="store_true",
        help="Show per-request detail for the slowest requests",
    )
    parser.add_argument(
        "--top",
        type=int,
        default=5,
        help="Number of slowest requests to show in --detail mode (default: 5)",
    )
    parser.add_argument(
        "--request-id",
        type=str,
        default=None,
        help="Show detailed breakdown for a specific request_id",
    )
    parser.add_argument(
        "--csv",
        type=str,
        default=None,
        help="Export per-request data to CSV file",
    )
    args = parser.parse_args()

    # ---- Parse ----
    entries = parse_log(args.logfile)
    if not entries:
        print("No TTFT_TRACE entries found in the log file.", file=sys.stderr)
        sys.exit(1)

    grouped, id_map = group_by_request(entries)

    # ---- Compute per-request durations ----
    per_request_norm: dict[str, dict[str, float | None]] = {}
    for norm_id, events in grouped.items():
        if not _is_real_request(norm_id):
            continue
        per_request_norm[norm_id] = compute_request_durations(events)

    if not per_request_norm:
        print("No real-request entries found.", file=sys.stderr)
        sys.exit(1)

    # Build display-friendly version (normalized_id → display_id)
    per_request: dict[str, dict[str, float | None]] = {}
    for norm_id, stages in per_request_norm.items():
        display_id = id_map.get(norm_id, norm_id)
        per_request[display_id] = stages

    # ---- Aggregate ----
    stage_stats = aggregate_stage_durations(per_request)
    point_stats = aggregate_point_offsets(per_request)
    ttft_stats = compute_ttft_stats(per_request)

    # ---- Print ----
    print_summary(per_request, stage_stats, point_stats, ttft_stats)
    print_timeline(per_request, ttft_stats)
    print_batch_stats(entries)
    print_temp_stats(entries)

    # ---- Show slowest requests ----
    if args.detail and ttft_stats:
        # Sort by TTFT descending
        ranked: list[tuple[str, float]] = []
        for req_id, stages in per_request.items():
            start = stages.get("api_request_start")
            end = stages.get("first_token_api_yield")
            if start is not None and end is not None:
                ranked.append((req_id, end - start))  # type: ignore[operator]
        ranked.sort(key=lambda x: x[1], reverse=True)

        top_n = ranked[: args.top]
        if top_n:
            print(f"\n  Top-{len(top_n)} slowest requests by TTFT:")
            for req_id, total in top_n:
                print(f"    {req_id}  —  {_ms(total * 1000)}")
                if args.top == 1:
                    print_request_detail(req_id, per_request[req_id])

    # ---- Show specific request ----
    if args.request_id:
        # Try to resolve any raw ID → chatcmpl-xxx via the mapping
        id_map_raw_to_cc = _build_id_map(entries)
        resolved = id_map_raw_to_cc.get(args.request_id, args.request_id)
        data = per_request.get(resolved) or per_request.get(args.request_id)
        if data is None:
            print(f"\n  Request '{args.request_id}' not found.")
        else:
            print_request_detail(resolved, data)

    # ---- CSV export ----
    if args.csv:
        export_csv(per_request, args.csv)

    print()
    print(TTFT_STAGE_DESCRIPTION)


if __name__ == "__main__":
    main()
