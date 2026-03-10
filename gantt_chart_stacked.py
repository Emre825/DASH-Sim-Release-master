"""
Gantt chart for trace files with format:
App Name, Job ID, Task ID, Fused, PE, Start Time (ns), Finish Time (ns), Exec. Time (ns)
"""
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import Patch
import numpy as np
import csv
import argparse
import sys

from collections import namedtuple

ScheduleEvent = namedtuple('ScheduleEvent', 'app_name job_id task_name task_id fused start end proc')

# ── Appearance config ──────────────────────────────────────────────────────────
# Task-type based colors
TASK_TYPE_COLORS = {
    'Conv2d':  '#4363d8',  # blue
    'Relu':    '#e6194b',  # red
    'Fused':   '#3cb44b',  # green
}
FUSED_HATCH   = ''
UNFUSED_HATCH = ''

# Fixed PE list and capacity
TRACKED_PES      = ['GPU_1', 'GPU_2', 'GPU_3']
LANES_PER_PE     = 4   # each PE has 4 capacity slots
# ──────────────────────────────────────────────────────────────────────────────


def ns_to_ms(ns: int) -> float:
    return ns / 1_000_000.0


def assign_lanes(events, n_lanes=LANES_PER_PE):
    """
    Given a list of ScheduleEvents for one PE, assign each event
    to one of the fixed *n_lanes* capacity slots so that no two
    overlapping events share the same lane.
    Returns list of (event, lane_index).
    """
    sorted_evs = sorted(events, key=lambda e: e.start)
    lane_end_times = [0] * n_lanes  # pre-allocate fixed lanes
    assignments = []

    for ev in sorted_evs:
        for i in range(n_lanes):
            if ev.start >= lane_end_times[i]:
                lane_end_times[i] = ev.end
                assignments.append((ev, i))
                break

    return assignments


def get_task_type(ev):
    """Return the task type for color coding: 'Conv2d', 'Relu', or 'Fused'."""
    if ev.fused:
        return 'Fused'
    name_lower = ev.task_name.lower()
    if 'conv2d' in name_lower:
        return 'Conv2d'
    if 'relu' in name_lower:
        return 'Relu'
    return 'Conv2d'  # fallback


def show_gantt_chart(proc_schedules, time_offset: int = 0):
    processors = TRACKED_PES  # fixed list: GPU_1, GPU_2, GPU_3

    # ── Assign lanes (fixed 4 per PE) ────────────────────────────────────────
    lane_data = {}  # proc -> list of (event, lane_idx)
    for proc in processors:
        events = proc_schedules.get(proc, [])
        lane_data[proc] = assign_lanes(events) if events else []

    # ── Compute y-positions: each proc always gets LANES_PER_PE lanes ────────
    ROW_HEIGHT   = 0.4   # height of a single lane bar
    ROW_PADDING  = 0.25  # gap between processor groups

    proc_y_base = {}  # bottom y of each processor's block
    y_cursor = ROW_PADDING
    for proc in processors:
        proc_y_base[proc] = y_cursor
        y_cursor += LANES_PER_PE * ROW_HEIGHT + ROW_PADDING

    total_height = y_cursor

    # y tick positions and labels (centered in each proc block)
    tick_positions = []
    for proc in processors:
        block_center = proc_y_base[proc] + (LANES_PER_PE * ROW_HEIGHT) / 2.0
        tick_positions.append(block_center)

    # ── Determine global x-axis extent ───────────────────────────────────────
    all_ends = [ev.end for proc in processors for ev in proc_schedules.get(proc, [])]
    global_end_ms = ns_to_ms(max(all_ends) - time_offset) * 1.05 if all_ends else 1.0

    # ── Draw ─────────────────────────────────────────────────────────────────
    fig_height = max(5, total_height * 1.8)
    fig, ax = plt.subplots(figsize=(20, fig_height))

    for proc in processors:
        y_base = proc_y_base[proc]
        band_h = LANES_PER_PE * ROW_HEIGHT

        # Light background band spanning all 4 lanes (always visible)
        ax.barh(
            y_base + band_h / 2.0,
            global_end_ms,
            left=0,
            height=band_h,
            align='center',
            color='#f0f0f0',
            edgecolor='#cccccc',
            linewidth=0.5,
            zorder=0,
        )

        for ev, lane_idx in lane_data[proc]:
            task_type = get_task_type(ev)
            color    = TASK_TYPE_COLORS[task_type]
            hatch    = FUSED_HATCH if ev.fused else UNFUSED_HATCH
            alpha    = 0.92 if ev.fused else 0.75
            start_ms = ns_to_ms(ev.start - time_offset)
            dur_ms   = ns_to_ms(ev.end   - ev.start)

            # Lane center y
            y_center = y_base + lane_idx * ROW_HEIGHT + ROW_HEIGHT / 2.0

            # Colored bar
            ax.barh(
                y_center, dur_ms,
                left=start_ms,
                height=ROW_HEIGHT * 0.85,
                align='center',
                color=color,
                edgecolor='black',
                linewidth=0.2,
                alpha=alpha,
                zorder=2,
            )
            # Hatch overlay for fused
            if hatch:
                ax.barh(
                    y_center, dur_ms,
                    left=start_ms,
                    height=ROW_HEIGHT * 0.85,
                    align='center',
                    color='none',
                    edgecolor='white',
                    linewidth=0.2,
                    hatch=hatch,
                    alpha=0.85,
                    zorder=3,
                )

    # ── Axes formatting ───────────────────────────────────────────────────────
    ax.set_yticks(tick_positions)
    ax.set_yticklabels(processors, fontsize=13)
    plt.ylabel('Processors', fontsize=16)
    plt.xlabel('Time (ms)',   fontsize=16)
    ax.set_ylim(0, total_height)
    ax.tick_params(axis='x', labelsize=13)
    ax.grid(color='grey', linestyle=':', alpha=0.4, zorder=1)

    # Horizontal separator lines between processor groups
    for proc in processors:
        ax.axhline(y=proc_y_base[proc], color='#999999', linewidth=0.8, linestyle='-', zorder=1)

    # ── Legend ────────────────────────────────────────────────────────────────
    legend_elements = [
        Patch(facecolor=TASK_TYPE_COLORS['Conv2d'], edgecolor='black',
              linewidth=0.5, label='Conv2d (unfused)'),
        Patch(facecolor=TASK_TYPE_COLORS['Relu'], edgecolor='black',
              linewidth=0.5, label='Relu (unfused)'),
        Patch(facecolor=TASK_TYPE_COLORS['Fused'], edgecolor='white',
              linewidth=0.5, hatch=FUSED_HATCH, label='Fused (Conv2d+Relu)'),
    ]

    ax.legend(
        handles=legend_elements,
        loc='upper right',
        ncol=len(legend_elements),
        fontsize=12,
        framealpha=0.9,
    )

    plt.tight_layout()
    plt.savefig("gantt_output.png", dpi=150, bbox_inches='tight')
    print("Saved → gantt_output.png")


def generate_argparser():
    parser = argparse.ArgumentParser(
        description="Gantt chart plotter for fused/unfused multi-app traces"
    )
    parser.add_argument("inputFile", help="CSV trace file to plot")
    return parser


if __name__ == "__main__":
    args = generate_argparser().parse_args()

    proc_schedules: dict[str, list[ScheduleEvent]] = {}
    time_offset = sys.maxsize

    with open(args.inputFile, newline='', encoding='utf-8') as f:
        reader = csv.reader(f)
        next(reader)
        for row in reader:
            if len(row) < 9:
                continue
            try:
                start_ns = int(row[6].strip())
                time_offset = min(time_offset, start_ns)
            except ValueError:
                pass

    if time_offset == sys.maxsize:
        time_offset = 0

    with open(args.inputFile, newline='', encoding='utf-8') as f:
        reader = csv.reader(f)
        next(reader)
        total, skipped, plotted = 0, 0, 0
        for row in reader:
            if len(row) < 9:
                continue
            total += 1
            try:
                app_name  = row[0].strip()
                job_id    = int(row[1].strip())
                task_name = row[2].strip()
                task_id   = int(row[3].strip())
                fused     = row[4].strip().lower() == 'true'
                proc      = row[5].strip()
                start_ns  = int(row[6].strip())
                end_ns    = int(row[7].strip())
            except (ValueError, IndexError):
                skipped += 1
                continue

            if end_ns <= start_ns:
                skipped += 1
                continue

            # Only track the fixed PE list
            if proc not in TRACKED_PES:
                skipped += 1
                continue

            plotted += 1
            ev = ScheduleEvent(app_name, job_id, task_name, task_id, fused, start_ns, end_ns, proc)
            proc_schedules.setdefault(proc, []).append(ev)

    print(f"Total: {total} | Skipped (dummy): {skipped} | Plotted: {plotted}")
    show_gantt_chart(proc_schedules, time_offset=time_offset)