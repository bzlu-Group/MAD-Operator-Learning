"""Explicit trust boundary for the supported public checkpoint entry points."""
import warnings

from packaging.version import Version
import torch


def require_supported_runtime():
    if Version(str(torch.__version__).split("+", 1)[0]) < Version("2.15.0"):
        raise RuntimeError("Use the supported PyTorch environment in requirements-r4.txt; "
                           "the historical paper runtime is not a secure loading environment.")


def require_trusted_resume(trusted):
    require_supported_runtime()
    if not trusted:
        raise ValueError("Full resume checkpoints contain pickled Python/NumPy state. "
                         "Only for a checkpoint you created or independently trust, "
                         "pass --trust-checkpoint. Never use this flag for an unknown file.")
    warnings.warn("Trusted checkpoint mode permits pickle code execution. "
                  "Only load files whose source you independently trust.", RuntimeWarning)


def load_checkpoint(path, *, trusted=False):
    require_supported_runtime()
    if trusted:
        require_trusted_resume(True)
    # Never retry a rejected restricted load using unrestricted pickle.
    return torch.load(path, map_location="cpu", weights_only=not trusted)
