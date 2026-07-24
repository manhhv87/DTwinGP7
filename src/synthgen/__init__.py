"""
synthgen — calibration-anchored synthetic-data engine (paper contributions C2 + C3).

C2 pipeline:  SynthgenConfig (anchored_config) → SceneSampler (scene_sampler)
              → scene_*.json → render_blenderproc.py (under `blenderproc run`)
              → auto_label.convert_render_output → YOLO-seg dataset.
C3 pipeline:  trial CSVs → failure_miner.bin_failures → budget_alloc.allocate
              → per-mode SceneSampler conditions → next C2 round.

CLI entry points: scripts/20_generate_synth.py, scripts/21_mine_failures.py.
"""
from .anchored_config import LIGHTING_CONDITION_SCALE, SynthgenConfig
from .auto_label import convert_render_output, instance_map_to_yolo_lines, mask_to_polygon
from .budget_alloc import allocate, condition_for
from .failure_miner import bin_failures, kmeans_sensitivity, load_report, load_trials, save_report
from .scene_sampler import SceneSampler, load_sigma_json

__all__ = [
    "LIGHTING_CONDITION_SCALE",
    "SynthgenConfig",
    "SceneSampler",
    "load_sigma_json",
    "convert_render_output",
    "instance_map_to_yolo_lines",
    "mask_to_polygon",
    "bin_failures",
    "kmeans_sensitivity",
    "load_trials",
    "load_report",
    "save_report",
    "allocate",
    "condition_for",
]
