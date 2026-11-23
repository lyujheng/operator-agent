#!/usr/bin/env python3
"""Trace log analyzer — generates a readable execution report.

Usage:
    python scripts/analyze_trace.py output/logs/trace_xxxx.jsonl
    python scripts/analyze_trace.py  # auto-pick latest trace
"""

from __future__ import annotations

import json
import sys
from collections import Counter, defaultdict
from datetime import datetime
from pathlib import Path


def load_events(path: str) -> list[dict]:
    with open(path) as f:
        return [json.loads(line) for line in f if line.strip()]


def parse_time(ts: str) -> datetime:
    # Handle both timezone-aware and naive timestamps.
    ts = ts.replace("+00:00", "").replace("Z", "")
    for fmt in ("%Y-%m-%dT%H:%M:%S.%f", "%Y-%m-%dT%H:%M:%S"):
        try:
            return datetime.strptime(ts, fmt)
        except ValueError:
            continue
    return datetime.min


def fmt_duration(seconds: float) -> str:
    if seconds < 1:
        return f"{seconds*1000:.0f}ms"
    if seconds < 60:
        return f"{seconds:.1f}s"
    m, s = divmod(int(seconds), 60)
    if m < 60:
        return f"{m}分{s}秒"
    h, m = divmod(m, 60)
    return f"{h}时{m}分{s}秒"


def span_duration(span: dict) -> float:
    start = parse_time(span.get("started_at", ""))
    end = parse_time(span.get("ended_at", ""))
    return max((end - start).total_seconds(), 0)


def analyze(events: list[dict]) -> None:
    # ── Overall ──
    starts = [parse_time(e["started_at"]) for e in events if "started_at" in e]
    ends = [parse_time(e["ended_at"]) for e in events if "ended_at" in e]
    total_secs = (max(ends) - min(starts)).total_seconds() if starts and ends else 0

    types = Counter(e.get("span_type") for e in events)
    statuses = Counter(e.get("status") for e in events)

    print("═" * 60)
    print("  执行报告")
    print("═" * 60)
    print(f"  耗时: {fmt_duration(total_secs)}")
    print(f"  Spans: {len(events)}  (LLM调用: {types.get('llm_call', 0)}, "
          f"Tool调用: {types.get('tool', 0)}, Agent: {types.get('agent', 0)}, "
          f"Step: {types.get('step', 0)})")
    print(f"  状态: {dict(statuses)}")

    # ── Steps ──
    top_steps = [e for e in events if e.get("parent_id") is None and e.get("span_type") == "step"]
    plan_gen = [e for e in events if e.get("name") == "plan_generation"]

    if plan_gen:
        pg = plan_gen[0]
        print(f"\n── Plan Generation [{pg['status']}] ({fmt_duration(span_duration(pg))}) ──")

    # Group step spans by name to detect retries.
    step_groups: dict[str, list[dict]] = defaultdict(list)
    for s in top_steps:
        step_groups[s["name"]].append(s)

    # Build parent → children index.
    by_parent: dict[str, list[dict]] = defaultdict(list)
    for e in events:
        pid = e.get("parent_id")
        if pid:
            by_parent[pid].append(e)

    step_idx = 0
    for step_name, spans in step_groups.items():
        step_idx += 1
        first = spans[0]
        last = spans[-1]
        attempts = len(spans)
        passed = last.get("output_data") is not None
        status_str = "通过" if passed else "未通过"
        total_dur = sum(span_duration(s) for s in spans)

        retry_str = f", 重试{attempts}次" if attempts > 1 else ""
        print(f"\n── Step {step_idx}: {step_name.replace('step_', '')} "
              f"[{status_str}{retry_str}] ({fmt_duration(total_dur)}) ──")

        # Print output summary.
        out = last.get("output_data")
        if out:
            for k, v in out.items():
                if isinstance(v, list) and len(v) > 3:
                    print(f"  {k}: [{len(v)} items]")
                else:
                    print(f"  {k}: {v}")

        err = last.get("error")
        if err:
            print(f"  错误: {err}")

        # Show agent details for this step's time range.
        step_start = parse_time(first["started_at"])
        step_end = parse_time(last["ended_at"])

        agent_spans = [
            e for e in events
            if e.get("span_type") == "agent"
            and parse_time(e["started_at"]) >= step_start
            and parse_time(e["ended_at"]) <= step_end
        ]

        if agent_spans:
            # Group by agent_name, dedup across retries.
            seen: set[str] = set()
            for a in agent_spans:
                agent = a.get("metadata", {}).get("agent_name", "?")
                op_name = a.get("name", "?")
                key = f"{agent}/{op_name}"
                if key in seen:
                    continue
                seen.add(key)

                children = by_parent.get(a["span_id"], [])
                llm_count = sum(1 for c in children if c["span_type"] == "llm_call")
                tool_count = sum(1 for c in children if c["span_type"] == "tool")
                dur = span_duration(a)

                # Collect tool names.
                tool_names = [
                    c.get("input_data", {}).get("tool", "?")
                    for c in children if c["span_type"] == "tool"
                ]
                tool_summary = ", ".join(f"{t}×{c}" for t, c in Counter(tool_names).most_common(3))

                print(f"  ├─ {op_name} ({agent}) {fmt_duration(dur)}")
                print(f"  │   LLM: {llm_count}轮, Tool: {tool_count}次 [{tool_summary}]")

    # ── Tool Statistics ──
    tool_calls = [e for e in events if e.get("span_type") == "tool"]
    if tool_calls:
        print(f"\n── Tool 调用统计 ({len(tool_calls)} 次) ──")
        tool_counter = Counter(
            e.get("input_data", {}).get("tool", "?") for e in tool_calls
        )
        max_name_len = max(len(n) for n in tool_counter)
        for name, count in tool_counter.most_common():
            print(f"  {name:<{max_name_len}}  {count}次")

    # ── Errors ──
    errors = [e for e in events if e.get("status") in ("error", "failed") or e.get("error")]
    if errors:
        print(f"\n── 错误 ({len(errors)}) ──")
        for err in errors:
            agent = err.get("metadata", {}).get("agent_name", "?")
            print(f"  [{err['name']}] agent={agent} error={err.get('error', '')[:200]}")

    print("\n" + "═" * 60)


def find_latest_trace() -> str | None:
    log_dir = Path("output/logs")
    if not log_dir.exists():
        return None
    traces = sorted(log_dir.glob("trace_*.jsonl"), key=lambda p: p.stat().st_mtime)
    return str(traces[-1]) if traces else None


def main() -> None:
    if len(sys.argv) > 1:
        path = sys.argv[1]
    else:
        path = find_latest_trace()
        if not path:
            print("未找到 trace 文件，请指定路径: python scripts/analyze_trace.py <path>")
            sys.exit(1)
        print(f"[自动选择最新 trace: {path}]\n")

    events = load_events(path)
    analyze(events)


if __name__ == "__main__":
    main()
