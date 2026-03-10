"""
Gantt chart for trace files with format:
App Name, Job ID, Task ID, Fused, PE, Start Time (ns), Finish Time (ns), Exec. Time (ns)
"""
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.font_manager as font_manager
from matplotlib.patches import Patch, Rectangle
import numpy as np
import csv
import argparse
import sys

from collections import namedtuple

ScheduleEvent = namedtuple('ScheduleEvent', 'app_name job_id task_id fused start end proc')

# ── Appearance config ──────────────────────────────────────────────────────────
APP_COLORS = {}          # filled dynamically; override here if you like
DEFAULT_PALETTE = [
    '#e6194b',  # red
    '#4363d8',  # blue
    '#3cb44b',  # green
    '#f58231',  # orange
    '#911eb4',  # purple
    '#42d4f4',  # cyan
    '#f032e6',  # magenta
    '#bfef45',  # lime
    '#fabed4',  # pink
    '#469990',  # teal
]
FUSED_HATCH   = '///'     # hatched → fused
UNFUSED_HATCH = ''        # solid   → unfused
FUSED_ALPHA   = 0.90
UNFUSED_ALPHA = 0.65
# ──────────────────────────────────────────────────────────────────────────────


def ns_to_ms(ns: int) -> float:
    return ns / 1_000_000.0


def show_gantt_chart(proc_schedules, time_offset: int = 0):
    processors = sorted(proc_schedules.keys())

    # Assign a unique color per application name
    app_names = sorted({ev.app_name for evs in proc_schedules.values() for ev in evs})
    for i, name in enumerate(app_names):
        APP_COLORS[name] = DEFAULT_PALETTE[i % len(DEFAULT_PALETTE)]

    ilen = len(processors)
    pos  = np.arange(0.5, ilen * 0.5 + 0.5, 0.5)

    fig, ax = plt.subplots(figsize=(18, max(5, ilen * 1.1)))

    # ── Draw bars ─────────────────────────────────────────────────────────────
    for idx, proc in enumerate(processors):
        y_center = (idx * 0.5) + 0.5
        for ev in proc_schedules[proc]:
            color  = APP_COLORS[ev.app_name]
            hatch  = FUSED_HATCH if ev.fused else UNFUSED_HATCH
            alpha  = FUSED_ALPHA if ev.fused else UNFUSED_ALPHA
            start_ms = ns_to_ms(ev.start - time_offset)
            dur_ms   = ns_to_ms(ev.end - ev.start)

            # Main colored bar
            ax.barh(
                y_center, dur_ms,
                left=start_ms,
                height=0.35,
                align='center',
                color=color,
                edgecolor='black',
                linewidth=0.3,
                alpha=alpha,
            )
            # Hatch overlay (use white hatch on unfused for contrast)
            if hatch:
                ax.barh(
                    y_center, dur_ms,
                    left=start_ms,
                    height=0.35,
                    align='center',
                    color='none',
                    edgecolor='white',
                    linewidth=0.3,
                    hatch=hatch,
                    alpha=0.9,
                )

    # ── Axes formatting ───────────────────────────────────────────────────────
    plt.yticks(pos, processors, fontsize=13)
    plt.ylabel('Processors', fontsize=16)
    plt.xlabel('Time (ms)',   fontsize=16)

    ax.set_ylim(ymin=-0.1, ymax=ilen * 0.5 + 0.6)
    ax.tick_params(axis='x', labelsize=13)
    ax.tick_params(axis='y', labelsize=13)
    ax.grid(color='grey', linestyle=':', alpha=0.4)

    # ── Legend ────────────────────────────────────────────────────────────────
    legend_elements = []

    # Per-application color patches
    for name, color in sorted(APP_COLORS.items()):
        legend_elements.append(
            Patch(facecolor=color, edgecolor='black', linewidth=0.5, label=name)
        )

    # Fused / unfused visual guide
    legend_elements.append(
        Patch(facecolor='grey', edgecolor='black', linewidth=0.5,
              hatch=FUSED_HATCH, label='Fused')
    )
    legend_elements.append(
        Patch(facecolor='grey', edgecolor='white', linewidth=0.5,
              hatch=UNFUSED_HATCH, label='Unfused')
    )

    ax.legend(
        handles=legend_elements,
        loc='upper right',
        ncol=len(legend_elements),
        fontsize=12,
        framealpha=0.9,
    )

    plt.tight_layout()
    # plt.show()
    plt.savefig("gantt_output.png", dpi=150, bbox_inches='tight')


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

    # ── First pass: find global time offset ──────────────────────────────────
    with open(args.inputFile, newline='', encoding='utf-8') as f:
        reader = csv.reader(f)
        next(reader)  # skip header
        for row in reader:
            if len(row) < 7:
                continue
            try:
                start_ns = int(row[5].strip())
                time_offset = min(time_offset, start_ns)
            except ValueError:
                pass

    if time_offset == sys.maxsize:
        time_offset = 0

    # ── Second pass: build schedule ───────────────────────────────────────────
    with open(args.inputFile, newline='', encoding='utf-8') as f:
        reader = csv.reader(f)
        next(reader)  # skip header
        for row in reader:
            if len(row) < 7:
                continue
            try:
                app_name = row[0].strip()
                job_id   = int(row[1].strip())
                task_id  = int(row[2].strip())
                fused    = row[3].strip().lower() == 'true'
                proc     = row[4].strip()
                start_ns = int(row[5].strip())
                end_ns   = int(row[6].strip())
            except (ValueError, IndexError):
                continue

            # Skip zero-duration placeholder rows (exec time == 0)
            if end_ns <= start_ns:
                continue

            ev = ScheduleEvent(app_name, job_id, task_id, fused, start_ns, end_ns, proc)
            proc_schedules.setdefault(proc, []).append(ev)

    show_gantt_chart(proc_schedules, time_offset=time_offset)