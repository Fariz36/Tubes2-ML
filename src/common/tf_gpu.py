import ctypes
import os
import site
from functools import lru_cache
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path

CUDA_LIBRARY_ORDER = (
    ("cuda_runtime", "libcudart.so.12"),
    ("cuda_nvrtc", "libnvrtc.so.12"),
    ("cublas", "libcublas.so.12"),
    ("cublas", "libcublasLt.so.12"),
    ("cufft", "libcufft.so.11"),
    ("curand", "libcurand.so.10"),
    ("cusolver", "libcusolver.so.11"),
    ("cusparse", "libcusparse.so.12"),
    ("cudnn", "libcudnn.so.9"),
    ("nccl", "libnccl.so.2"),
    ("nvjitlink", "libnvJitLink.so.12"),
    ("cuda_cupti", "libcupti.so.12"),
)

@lru_cache(maxsize=1)
def find_nvidia_site_packages_root():
    for package_root in site.getsitepackages():
        candidate = Path(package_root) / "nvidia"
        if candidate.exists():
            return Path(package_root)
    raise FileNotFoundError("Could not locate site-packages/nvidia in the active environment.")

def get_nvidia_library_dirs():
    root = find_nvidia_site_packages_root() / "nvidia"
    return sorted(path for path in root.glob("*/lib") if path.is_dir())

def build_ld_library_path(include_wsl_driver=True):
    paths = [str(path) for path in get_nvidia_library_dirs()]
    if include_wsl_driver:
        paths.append("/usr/lib/wsl/lib")
    return ":".join(paths)

def build_ld_preload():
    root = find_nvidia_site_packages_root() / "nvidia"
    preload_paths = [root / package / "lib" / library_name for package, library_name in CUDA_LIBRARY_ORDER]
    missing = [str(path) for path in preload_paths if not path.exists()]
    if missing:
        raise FileNotFoundError(f"Missing expected NVIDIA libraries: {missing}")
    return ":".join(str(path) for path in preload_paths)

def ensure_cudnn_aliases():
    try:
        cudnn_version = version("nvidia-cudnn-cu12")
    except PackageNotFoundError:
        return []

    parts = cudnn_version.split(".")
    if len(parts) < 3:
        return []

    major, minor, patch = parts[:3]
    alias_suffixes = (f".{major}.{minor}", f".{major}.{minor}.{patch}")

    cudnn_lib_dir = find_nvidia_site_packages_root() / "nvidia" / "cudnn" / "lib"
    created = []
    for library_path in sorted(cudnn_lib_dir.glob("libcudnn*.so.9")):
        for suffix in alias_suffixes:
            alias = cudnn_lib_dir / f"{library_path.name}{suffix[2:]}"
            if alias.exists() or alias.is_symlink():
                continue
            alias.symlink_to(library_path.name)
            created.append(alias)
    return created

def preload_nvidia_libraries():
    root = find_nvidia_site_packages_root() / "nvidia"
    loaded = []
    for package_name, library_name in CUDA_LIBRARY_ORDER:
        library_path = root / package_name / "lib" / library_name
        ctypes.CDLL(str(library_path), mode=ctypes.RTLD_GLOBAL)
        loaded.append(library_path)
    return loaded

def configure_tensorflow_gpu():
    ensure_cudnn_aliases()

    ld_library_path = build_ld_library_path(include_wsl_driver=True)
    ld_preload = build_ld_preload()

    existing_ld_library_path = os.environ.get("LD_LIBRARY_PATH")
    if existing_ld_library_path:
        ld_library_path = f"{ld_library_path}:{existing_ld_library_path}"

    existing_ld_preload = os.environ.get("LD_PRELOAD")
    if existing_ld_preload:
        ld_preload = f"{ld_preload}:{existing_ld_preload}"

    os.environ["LD_LIBRARY_PATH"] = ld_library_path
    os.environ["LD_PRELOAD"] = ld_preload
    preload_nvidia_libraries()

    return {
        "LD_LIBRARY_PATH": ld_library_path,
        "LD_PRELOAD": ld_preload,
    }