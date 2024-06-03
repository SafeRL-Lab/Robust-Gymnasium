"""Wrappers for vector environments."""
# pyright: reportUnsupportedDunderAll=false
import importlib

from robust_gymnasium.wrappers.vector.common import RecordEpisodeStatistics
from robust_gymnasium.wrappers.vector.dict_info_to_list import DictInfoToList
from robust_gymnasium.wrappers.vector.rendering import HumanRendering
from robust_gymnasium.wrappers.vector.stateful_observation import NormalizeObservation
from robust_gymnasium.wrappers.vector.stateful_reward import NormalizeReward
from robust_gymnasium.wrappers.vector.vectorize_action import (
    ClipAction,
    RescaleAction,
    TransformAction,
    VectorizeTransformAction,
)
from robust_gymnasium.wrappers.vector.vectorize_observation import (
    DtypeObservation,
    FilterObservation,
    FlattenObservation,
    GrayscaleObservation,
    RescaleObservation,
    ReshapeObservation,
    ResizeObservation,
    TransformObservation,
    VectorizeTransformObservation,
)
from robust_gymnasium.wrappers.vector.vectorize_reward import (
    ClipReward,
    TransformReward,
    VectorizeTransformReward,
)


__all__ = [
    # --- Vector only wrappers
    "VectorizeTransformObservation",
    "VectorizeTransformAction",
    "VectorizeTransformReward",
    "DictInfoToList",
    # --- Observation wrappers ---
    "TransformObservation",
    "FilterObservation",
    "FlattenObservation",
    "GrayscaleObservation",
    "ResizeObservation",
    "ReshapeObservation",
    "RescaleObservation",
    "DtypeObservation",
    "NormalizeObservation",
    # "RenderObservation",
    # "TimeAwareObservation",
    # "FrameStackObservation",
    # "DelayObservation",
    # --- Action Wrappers ---
    "TransformAction",
    "ClipAction",
    "RescaleAction",
    # --- Reward wrappers ---
    "TransformReward",
    "ClipReward",
    "NormalizeReward",
    # --- Common ---
    "RecordEpisodeStatistics",
    # --- Rendering ---
    # "RenderCollection",
    # "RecordVideo",
    "HumanRendering",
    # --- Conversion ---
    "JaxToNumpy",
    "JaxToTorch",
    "NumpyToTorch",
]


# As these wrappers requires `jax` or `torch`, they are loaded by runtime on users trying to access them
#   to avoid `import jax` or `import torch` on `import robust_gymnasium`.
_wrapper_to_class = {
    # data converters
    "JaxToNumpy": "jax_to_numpy",
    "JaxToTorch": "jax_to_torch",
    "NumpyToTorch": "numpy_to_torch",
}


def __getattr__(wrapper_name: str):
    """Load a wrapper by name.

    This optimizes the loading of robust_gymnasium wrappers by only loading the wrapper if it is used.
    Errors will be raised if the wrapper does not exist or if the version is not the latest.

    Args:
        wrapper_name: The name of a wrapper to load.

    Returns:
        The specified wrapper.

    Raises:
        AttributeError: If the wrapper does not exist.
        DeprecatedWrapper: If the version is not the latest.
    """
    # Check if the requested wrapper is in the _wrapper_to_class dictionary
    if wrapper_name in _wrapper_to_class:
        import_stmt = f"robust_gymnasium.wrappers.vector.{_wrapper_to_class[wrapper_name]}"
        module = importlib.import_module(import_stmt)
        return getattr(module, wrapper_name)

    raise AttributeError(f"module {__name__!r} has no attribute {wrapper_name!r}")
