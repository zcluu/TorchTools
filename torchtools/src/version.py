import torch
import pkg_resources
from packaging.version import Version


def get_torch_version() -> Version:
    try:
        if hasattr(torch, "__version__"):
            pkg_version = Version(torch.__version__)
        else:
            pkg_version = Version(pkg_resources.get_distribution("torch").version)
    except TypeError as e:
        raise TypeError("PyTorch version could not be detected automatically.") from e

    return pkg_version


def is_torch_version_geq_1_12() -> bool:
    return get_torch_version() >= Version("1.12.0")
