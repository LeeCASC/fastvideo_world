"""CUDA detection must not depend solely on a working NVML library."""

from types import SimpleNamespace
from unittest.mock import patch

from fastvideo.platforms import cuda_platform_plugin


class NVMLError_LibRmVersionMismatch(Exception):
    pass


def test_cuda_platform_uses_runtime_when_nvml_version_mismatches():
    nvml = SimpleNamespace(nvmlInit=lambda: (_ for _ in ()).throw(NVMLError_LibRmVersionMismatch()))
    with patch("fastvideo.utils.import_pynvml", return_value=nvml), \
            patch("torch.cuda.is_available", return_value=True), \
            patch("torch.cuda.device_count", return_value=8):
        assert cuda_platform_plugin() == "fastvideo.platforms.cuda.CudaPlatform"


def test_cuda_platform_rejects_unusable_runtime_after_nvml_failure():
    nvml = SimpleNamespace(nvmlInit=lambda: (_ for _ in ()).throw(NVMLError_LibRmVersionMismatch()))
    with patch("fastvideo.utils.import_pynvml", return_value=nvml), \
            patch("torch.cuda.is_available", return_value=False):
        assert cuda_platform_plugin() is None
