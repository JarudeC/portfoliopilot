# pgportfolio_pytorch/tools/configprocess.py
# ------------------------------------------
# Minimal rewrite of the original util — no TensorFlow imports,
# just default-filling and JSON helpers.  Path logic now targets the
# new package name “pgportfolio_pytorch”.

from __future__ import annotations
import json, os, sys, time
from datetime import datetime
from typing import Any, Dict

# ─── root path of repository ---------------------------------------------------
rootpath = (
    os.path.dirname(os.path.abspath(__file__))
    .replace("\\pgportfolio_pytorch\\tools", "")
    .replace("/pgportfolio_pytorch/tools", "")
)

# Python-2 fallback (kept for parity)
try:
    unicode       # type: ignore  # pyright: ignore[reportUndefinedVariable]
except NameError:
    unicode = str

# ──────────────────────────── public helpers ──────────────────────────────────
def preprocess_config(cfg: Dict[str, Any]) -> Dict[str, Any]:
    """Fill missing keys with defaults, byte-ify on Py-2."""
    fill_default(cfg)
    if sys.version_info[0] == 2:  # pragma: no cover
        return byteify(cfg)
    return cfg


def complete_config(user_cfg: Dict[str, Any]) -> Dict[str, Any]:
    """
    The original PGPortfolio returned a merged config.
    For our use-case we just preprocess and return.
    """
    return preprocess_config(user_cfg)

# ──────────────────────── default setters  ────────────────────────────────────
def fill_default(cfg: Dict[str, Any]) -> None:
    set_missing(cfg, "random_seed", 0)
    set_missing(cfg, "agent_type", "NNAgent")

    if "input" in cfg:
        fill_input_default(cfg["input"])
    if "training" in cfg:
        fill_train_default(cfg["training"])


def fill_train_default(train: Dict[str, Any]) -> None:
    set_missing(train, "fast_train", True)
    set_missing(train, "decay_rate", 1.0)
    set_missing(train, "decay_steps", 50_000)
    set_missing(train, "dropout", 0.2)


def fill_input_default(inp: Dict[str, Any]) -> None:
    set_missing(inp, "save_memory_mode", False)
    set_missing(inp, "portion_reversed", False)
    set_missing(inp, "market", "poloniex")
    set_missing(inp, "norm_method", "absolute")
    set_missing(inp, "is_permed", False)
    set_missing(inp, "fake_ratio", 1)

# ──────────────────────── misc utilities  ─────────────────────────────────────
def set_missing(d: Dict[str, Any], key: str, value: Any) -> None:
    if key not in d:
        d[key] = value


def byteify(inp: Any):  # pragma: no cover (Py-3 only)
    if isinstance(inp, dict):
        return {byteify(k): byteify(v) for k, v in inp.items()}
    if isinstance(inp, list):
        return [byteify(el) for el in inp]
    if isinstance(inp, unicode):  # type: ignore
        return str(inp)
    return inp


def parse_time(time_str: str) -> float:
    """Convert 'YYYY/MM/DD' to epoch seconds (local)."""
    return time.mktime(datetime.strptime(time_str, "%Y/%m/%d").timetuple())


def load_config(index: int | None = None) -> Dict[str, Any]:
    """
    If `index` is None: load + preprocess `net_config.json` at repo root.
    Otherwise load it from `train_package/<index>/net_config.json`.
    """
    if index is None:
        path = os.path.join(rootpath, "pgportfolio_pytorch", "net_config.json")
    else:
        path = os.path.join(rootpath, "train_package", str(index), "net_config.json")

    with open(path, encoding="utf-8") as fh:
        cfg = json.load(fh)
    return preprocess_config(cfg)


def check_input_same(a: Dict[str, Any], b: Dict[str, Any]) -> bool:
    """Quick equivalence check on key input fields."""
    FIELDS = ("start_date", "end_date", "test_portion")
    return all(a["input"].get(k) == b["input"].get(k) for k in FIELDS)
