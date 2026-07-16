"""Loader for SE-augmented ablation data (the new run).

Single source of truth: ablation_se.json (built by the DATA agent from
ablations_results/sheets/*.csv). Import this instead of re-parsing the CSVs
or the workbook.

    from ablation_data import load, sheets, meta
    rows = load("GT_SR")          # -> list of {task, obs, method, mean, std, se}
    rows = load("A6_KAG_Off")     # ablated arm carries mean/std/se (SE=std/sqrt(n))

Method labels are DISEIL (never DISTIL). SE = std / sqrt(n), n = 9 for
GridWorld 5x5, n = 5 for the robot tasks (Push-T, Lift, Wipe, Door).
"""
import json
import os

_HERE = os.path.dirname(os.path.abspath(__file__))
_JSON = os.path.join(_HERE, "ablation_se.json")

with open(_JSON, encoding="utf-8") as _f:
    _DATA = json.load(_f)


def sheets():
    """Return the list of available sheet names."""
    return list(_DATA.keys())


def load(sheet):
    """Return the list of tidy rows for a sheet.

    Each row is a dict with at least: task, obs, method, mean, std, se
    (std/se are None where the source has no dispersion, e.g. GT_InfoGain).
    """
    if sheet not in _DATA:
        raise KeyError(f"unknown sheet {sheet!r}; available: {sheets()}")
    return _DATA[sheet]["rows"]


def meta(sheet):
    """Return the metadata dict for a sheet (source CSV, quantity, value note)."""
    return _DATA[sheet]["meta"]


def n_for(task):
    """Seeds per task used for SE: 9 for GridWorld, 5 for robot tasks."""
    return 9 if str(task).startswith("GridWorld") else 5


if __name__ == "__main__":
    for s in sheets():
        print(f"{s:24s} {len(load(s)):4d} rows  ({meta(s).get('source','')})")
