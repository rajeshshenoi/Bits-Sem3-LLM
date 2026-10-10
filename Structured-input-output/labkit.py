"""
labkit.py — ONE portable bootstrap for every LLM lab notebook.

WHY THIS EXISTS
    The labs must run unchanged on Colab and Kubeflow. Each puts persistent
    storage, the GPU, and the pre-installed packages in a *different* place.
    When every notebook detects all that on its own, they drift apart (which is
    exactly what happened). Here it is decided once; notebooks just read the
    answers back.

STANDARD FIRST CELL (identical in every notebook)
    import labkit
    L = labkit.setup()                 # env + HF cache + device + dtype + paths
    labkit.ensure({"trl": "trl==0.12.1"})   # install ONLY if missing (never upgrades)

    # then use:  L.DEVICE  L.DTYPE  L.CACHE_ROOT  L.WORK_DIR  L.IS_GPU  L.GPU_MEM_GB

ORDER MATTERS
    setup() sets HF_HOME, which Hugging Face reads only at import time. So call
    setup() BEFORE importing transformers/datasets/trl. Put it in cell 1.
"""

from __future__ import annotations
import os, sys, shutil, subprocess, importlib.util
from types import SimpleNamespace

_STATE: SimpleNamespace | None = None   # filled by setup(), reused by helpers


# --------------------------------------------------------------------------
# 1. Where are we running? Signals, most specific first.
# --------------------------------------------------------------------------
# Only two backends are supported: Colab and Kubeflow. `local` is a bare
# safety fallback so setup() never crashes if run somewhere unexpected.
def detect_env() -> str:
    if importlib.util.find_spec("google.colab") is not None:
        return "colab"
    if "NB_PREFIX" in os.environ or "KUBERNETES_SERVICE_HOST" in os.environ \
       or os.path.isdir("/home/jovyan/data"):
        return "kubeflow"
    return "local"


# Per-env PERSISTENT root — the volume that survives a restart. Big caches and
# training outputs go here. Order in each list = preference; first writable one
# wins, else we fall back to cwd.
_PERSIST_ROOTS = {
    "colab":    ["/content"],                                    # ephemeral — wiped each session; models re-download (no Drive space used)
    "kubeflow": ["/home/jovyan/data"],                           # NFS volume — persists across pod restarts
    "local":    [os.path.expanduser("~/lab-data"), os.getcwd()], # bare fallback only
}


def _first_writable(paths) -> str:
    for p in paths:
        try:
            os.makedirs(p, exist_ok=True)
            t = os.path.join(p, ".labkit_write_test")
            open(t, "w").close(); os.remove(t)
            return p
        except Exception:
            continue
    return os.getcwd()   # last resort: always writable


# --------------------------------------------------------------------------
# 2. setup() — the one call every notebook makes.
# --------------------------------------------------------------------------
def setup(quiet_logs: bool = True) -> SimpleNamespace:
    """Detect env (Colab or Kubeflow), point the HF cache at the right place,
    resolve device/dtype.

    On Colab the cache is the ephemeral /content disk on purpose: models
    re-download each session and use NO Google Drive space. (Mount Drive in the
    notebook's first cell only so `import labkit` can find labkit.py — not for
    the cache.) On Kubeflow the cache lives on the persistent ~/data volume.
    """
    global _STATE
    env = detect_env()
    root = _first_writable(_PERSIST_ROOTS[env])

    # HF cache -> <root>/hf-cache. Set BEFORE any transformers import.
    cache_root = os.path.join(root, "hf-cache")
    os.environ["HF_HOME"]           = cache_root
    os.environ["HF_DATASETS_CACHE"] = os.path.join(cache_root, "datasets")
    os.makedirs(os.environ["HF_DATASETS_CACHE"], exist_ok=True)

    if quiet_logs:
        os.environ.setdefault("TRANSFORMERS_VERBOSITY", "error")
        os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
        os.environ.setdefault("HF_HUB_VERBOSITY", "error")
        os.environ.setdefault("HF_HUB_DISABLE_SYMLINKS_WARNING", "1")

    # Device + dtype. Import torch lazily so `import labkit` stays cheap.
    device, dtype, gpu_name, gpu_mem = "cpu", "float32", None, 0.0
    try:
        import torch
        if torch.cuda.is_available():
            device = "cuda"
            gpu_name = torch.cuda.get_device_name(0)
            gpu_mem = round(torch.cuda.get_device_properties(0).total_memory / 1e9, 1)
            # bf16 is numerically safer than fp16 for Qwen/Llama; use it when supported.
            dtype = "bfloat16" if torch.cuda.is_bf16_supported() else "float16"
        elif getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
            device, dtype = "mps", "float16"   # Apple Silicon
    except Exception:
        pass

    _STATE = SimpleNamespace(
        ENV=env, PERSIST_ROOT=root, CACHE_ROOT=cache_root,
        WORK_DIR=os.path.join(root, "lab-work"),
        DEVICE=device, DTYPE=dtype, IS_GPU=(device == "cuda"),
        GPU_NAME=gpu_name, GPU_MEM_GB=gpu_mem,
    )
    os.makedirs(_STATE.WORK_DIR, exist_ok=True)

    gpu_str = f"{gpu_name} ({gpu_mem} GB)" if gpu_name else "none"
    print(f"env={env}  device={device}  dtype={dtype}  gpu={gpu_str}")
    print(f"persistent root : {root}")
    print(f"HF cache        : {cache_root}")
    return _STATE


def _require_setup() -> SimpleNamespace:
    if _STATE is None:
        raise RuntimeError("Call labkit.setup() first (in cell 1).")
    return _STATE


# --------------------------------------------------------------------------
# 3. ensure() — install ONLY what's missing; never upgrade.
#    Unifies the three different install cells. On Kubeflow (setup.sh already
#    ran) every package is present, so this is a no-op — which respects the
#    "additive installs OK, upgrades not" rule of the pinned venv.
# --------------------------------------------------------------------------
def ensure(packages: dict[str, str]) -> None:
    """packages: {importable_module_name: pip_spec}, e.g. {"faiss": "faiss-cpu==1.9.0"}."""
    missing = [spec for mod, spec in packages.items()
               if importlib.util.find_spec(mod) is None]
    if not missing:
        print("deps: all present (no install needed)")
        return
    print("deps: installing", ", ".join(missing))
    subprocess.run([sys.executable, "-m", "pip", "install", "-q", *missing], check=True)


# --------------------------------------------------------------------------
# 4. Small helpers notebooks were hardcoding.
# --------------------------------------------------------------------------
def by_hardware(gpu_choice, cpu_choice):
    """Pick a model (or anything) by available hardware. Replaces the scattered
    `"...-3B" if DEVICE=="cuda" else "...-0.5B"` lines."""
    return gpu_choice if _require_setup().IS_GPU else cpu_choice


def training_dir(name: str) -> str:
    """A per-lab output_dir on persistent storage. Pair with save_total_limit=1
    in your TrainingArguments/DPOConfig so optimizer-state checkpoints can't pile
    up (that pile-up is what filled 13 GB)."""
    d = os.path.join(_require_setup().WORK_DIR, name)
    os.makedirs(d, exist_ok=True)
    return d


# --------------------------------------------------------------------------
# 5. Disk hygiene — the 15 GB student budget lives or dies here.
# --------------------------------------------------------------------------
def _dir_gb(path: str) -> float:
    if not os.path.isdir(path):
        return 0.0
    total = 0
    for dp, _, files in os.walk(path):
        for f in files:
            try: total += os.path.getsize(os.path.join(dp, f))
            except OSError: pass
    return round(total / 1e9, 2)


def disk_report(budget_gb: float = 15.0) -> None:
    """Print where space is going and warn if the persistent volume nears budget.
    Default budget = a student's 15 GB volume."""
    s = _require_setup()
    used = shutil.disk_usage(s.PERSIST_ROOT)
    print(f"volume ({s.PERSIST_ROOT}): {used.used/1e9:.1f} used / {used.total/1e9:.1f} GB total")
    print(f"  HF cache : {_dir_gb(s.CACHE_ROOT):.2f} GB")
    print(f"  lab work : {_dir_gb(s.WORK_DIR):.2f} GB")
    if used.used / 1e9 > 0.9 * budget_gb:
        print(f"  ⚠ over 90% of a {budget_gb:.0f} GB budget — run labkit.cleanup().")


def cleanup() -> None:
    """Free this lab's TRAINING output — the checkpoint/optimizer-state folders that
    a run writes fresh each time (this is what overflows a 15 GB volume).

    The model cache is deliberately NOT touched: your notebooks load cache-first
    (local_files_only=True, download only on a miss), so the cache is a re-usable
    asset, not garbage — deleting it would just force a re-download. Only training
    labs (DPO, LoRA) produce anything for cleanup(); inference labs write nothing.
    Use disk_report() everywhere as a read-only 'how full am I' check."""
    s = _require_setup()
    freed = _dir_gb(s.WORK_DIR)
    shutil.rmtree(s.WORK_DIR, ignore_errors=True)
    os.makedirs(s.WORK_DIR, exist_ok=True)
    print(f"cleaned training output at {s.WORK_DIR}  (~{freed:.2f} GB freed; model cache kept)")
