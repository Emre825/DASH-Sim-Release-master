"""
Multi-panel Gantt chart for three trace files.

Each input CSV has format:
App Name, Job ID, Task ID, Fused, PE, Start Time (ns), Finish Time (ns), Exec. Time (ns)

This script plots only a given time window (relative to each file's earliest trace start)
for three files in a single figure:
- Top:    unfused
- Middle: TVM
- Bottom: DTF

All three panels:
- have identical horizontal/vertical size,
- share the same x-axis,
- have independent y-axes with PE1/PE2/PE3 labels.
"""

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import Patch
import csv
import argparse
import sys

from collections import namedtuple

ScheduleEvent = namedtuple('ScheduleEvent', 'app_name job_id task_name task_id fused start end proc')

# ── Appearance config ──────────────────────────────────────────────────────────
TASK_TYPE_COLORS = {
	'BN2D': '#4363d8',
	'Relu':   '#e6194b',
	'Fused':  '#7BC47F',
}
FUSED_HATCH = ''
UNFUSED_HATCH = ''

TRACKED_PES = ['GPU_1', 'GPU_2', 'GPU_3']
DISPLAY_PE_NAMES = {'GPU_1': 'PE1', 'GPU_2': 'PE2', 'GPU_3': 'PE3'}
LANES_PER_PE = 5

WINDOW_START_MS = 15.0
WINDOW_END_MS = 35.0
MIN_VISIBLE_BAR_RATIO = 0.001
# ──────────────────────────────────────────────────────────────────────────────
def ns_to_ms(ns: int) -> float:
	return ns / 1_000_000.0

def ms_to_ns(ms: float) -> int:
	return int(ms * 1_000_000.0)

def assign_lanes(events, n_lanes=LANES_PER_PE):
	sorted_evs = sorted(events, key=lambda e: e.start)
	lane_end_times = [0] * n_lanes
	assignments = []

	for ev in sorted_evs:
		for lane_idx in range(n_lanes):
			if ev.start >= lane_end_times[lane_idx]:
				lane_end_times[lane_idx] = ev.end
				assignments.append((ev, lane_idx))
				break

	return assignments

def get_task_type(ev):
	if ev.fused:
		return 'Fused'
	name_lower = ev.task_name.lower()
	if 'bn2d' in name_lower:
		return 'BN2D'
	if 'relu' in name_lower:
		return 'Relu'
	return 'BN2D'

def parse_trace_file(input_file):
	proc_schedules = {}
	time_offset = sys.maxsize

	with open(input_file, newline='', encoding='utf-8') as f:
		reader = csv.reader(f)
		next(reader, None)
		for row in reader:
			if len(row) < 9:
				continue
			try:
				start_ns = int(row[6].strip())
				time_offset = min(time_offset, start_ns)
			except ValueError:
				continue

	if time_offset == sys.maxsize:
		time_offset = 0

	total, skipped, plotted = 0, 0, 0
	with open(input_file, newline='', encoding='utf-8') as f:
		reader = csv.reader(f)
		next(reader, None)
		for row in reader:
			if len(row) < 9:
				continue
			total += 1

			try:
				app_name = row[0].strip()
				job_id = int(row[1].strip())
				task_name = row[2].strip()
				task_id = int(row[3].strip())
				fused = row[4].strip().lower() == 'true'
				proc = row[5].strip()
				start_ns = int(row[6].strip())
				end_ns = int(row[7].strip())
			except (ValueError, IndexError):
				skipped += 1
				continue

			if end_ns <= start_ns:
				skipped += 1
				continue

			if proc not in TRACKED_PES:
				skipped += 1
				continue

			plotted += 1
			ev = ScheduleEvent(app_name, job_id, task_name, task_id, fused, start_ns, end_ns, proc)
			proc_schedules.setdefault(proc, []).append(ev)

	return proc_schedules, time_offset, total, skipped, plotted

def draw_panel(ax, proc_schedules, time_offset, panel_ylabel):
	processors = TRACKED_PES
	lane_data = {}
	for proc in processors:
		lane_data[proc] = assign_lanes(proc_schedules.get(proc, []))

	row_height = 0.15
	row_padding = 0.20
	proc_y_base = {}
	y_cursor = row_padding
	for proc in processors:
		proc_y_base[proc] = y_cursor
		y_cursor += LANES_PER_PE * row_height + row_padding
	total_height = y_cursor

	tick_positions = []
	for proc in processors:
		block_center = proc_y_base[proc] + (LANES_PER_PE * row_height) / 2.0
		tick_positions.append(block_center)

	window_start_ns = time_offset + ms_to_ns(WINDOW_START_MS)
	window_end_ns = time_offset + ms_to_ns(WINDOW_END_MS)
	x_min_ms = WINDOW_START_MS
	x_max_ms = WINDOW_END_MS
	global_span_ms = x_max_ms - x_min_ms
	min_visible_bar_ms = global_span_ms * MIN_VISIBLE_BAR_RATIO

	for proc in processors:
		y_base = proc_y_base[proc]
		band_h = LANES_PER_PE * row_height

		ax.barh(
			y_base + band_h / 2.0,
			global_span_ms,
			left=x_min_ms,
			height=band_h,
			align='center',
			color='#f0f0f0',
			edgecolor='#cccccc',
			linewidth=0.5,
			zorder=0,
		)

		for ev, lane_idx in lane_data[proc]:
			draw_start_ns = max(ev.start, window_start_ns)
			draw_end_ns = min(ev.end, window_end_ns)
			if draw_end_ns <= draw_start_ns:
				continue

			task_type = get_task_type(ev)
			color = TASK_TYPE_COLORS[task_type]
			hatch = FUSED_HATCH if ev.fused else UNFUSED_HATCH
			alpha = 1.0
			start_ms = ns_to_ms(draw_start_ns - time_offset)
			dur_ms = ns_to_ms(draw_end_ns - draw_start_ns)
			draw_dur_ms = max(dur_ms, min_visible_bar_ms)

			y_center = y_base + lane_idx * row_height + row_height / 2.0

			ax.barh(
				y_center,
				draw_dur_ms,
				left=start_ms,
				height=row_height * 0.85,
				align='center',
				color=color,
				edgecolor='none',
				linewidth=0,
				alpha=alpha,
				zorder=2,
			)

			if hatch:
				ax.barh(
					y_center,
					draw_dur_ms,
					left=start_ms,
					height=row_height * 0.85,
					align='center',
					color='none',
					edgecolor='white',
					linewidth=0.2,
					hatch=hatch,
					alpha=0.85,
					zorder=3,
				)

	ax.set_yticks(tick_positions)
	ax.set_yticklabels([DISPLAY_PE_NAMES.get(proc, proc) for proc in processors], fontsize=14)
	ax.set_ylabel(panel_ylabel, fontsize=26)
	ax.set_ylim(0, total_height)
	ax.set_xlim(x_min_ms, x_max_ms)
	ax.tick_params(axis='x', labelsize=14)
	ax.grid(color='grey', linestyle=':', alpha=0.4, zorder=1)

	for proc in processors:
		ax.axhline(y=proc_y_base[proc], color='#999999', linewidth=0.8, linestyle='-', zorder=1)

def generate_argparser():
	parser = argparse.ArgumentParser(
		description='Plot stacked 3-panel Gantt (unfused/TVM/DTF) for a given time window.'
	)
	parser.add_argument('unfusedFile', help='CSV for top panel (unfused)')
	parser.add_argument('tvmFile', help='CSV for middle panel (TVM)')
	parser.add_argument('dtfFile', help='CSV for bottom panel (DTF)')
	parser.add_argument('--output', default='gantt_multiple_output.pdf', help='Output image file')
	return parser

if __name__ == '__main__':
	args = generate_argparser().parse_args()

	panel_specs = [
		(args.unfusedFile, 'BASELINE'),
		(args.tvmFile, 'PyTorch'),
		(args.dtfFile, 'DTF'),
	]

	parsed = []
	for file_path, label in panel_specs:
		proc_schedules, time_offset, total, skipped, plotted = parse_trace_file(file_path)
		print(f"[{label}] Total: {total} | Skipped (dummy): {skipped} | Plotted: {plotted}")
		parsed.append((proc_schedules, time_offset, label))

	fig, axes = plt.subplots(3, 1, figsize=(20, 11), sharex=True, gridspec_kw={'hspace': 0.12})

	for ax, (proc_schedules, time_offset, panel_label) in zip(axes, parsed):
		draw_panel(ax, proc_schedules, time_offset, panel_label)

	legend_elements = [
		Patch(facecolor=TASK_TYPE_COLORS['BN2D'], edgecolor='black', linewidth=0.5, label='BN2D (unfused)'),
		Patch(facecolor=TASK_TYPE_COLORS['Relu'], edgecolor='black', linewidth=0.5, label='Relu (unfused)'),
		Patch(facecolor=TASK_TYPE_COLORS['Fused'], edgecolor='white', linewidth=0.5, hatch=FUSED_HATCH, label='Fused (BN2D+Relu)'),
	]
	fig.legend(
		handles=legend_elements,
		loc='upper center',
		bbox_to_anchor=(0.5, 0.985),
		ncol=3,
		fontsize=18,
		framealpha=0.9,
		borderaxespad=0.2,
	)

	axes[-1].set_xlabel('Time (ms)', fontsize=20)
	plt.tight_layout(rect=(0, 0, 1, 0.975))
	plt.savefig(args.output, dpi=150, bbox_inches='tight')
	print(f"Saved → {args.output}")
