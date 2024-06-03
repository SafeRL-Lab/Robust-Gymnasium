"""Root `__init__` of the robust_gymnasium module setting the `__all__` of robust_gymnasium modules."""
# isort: skip_file

from robust_gymnasium.core import (
    Env,
    Wrapper,
    ObservationWrapper,
    ActionWrapper,
    RewardWrapper,
)
from robust_gymnasium.spaces.space import Space
from robust_gymnasium.envs.registration import (
    make,
    spec,
    register,
    registry,
    pprint_registry,
    make_vec,
    VectorizeMode,
    register_envs,
)

# necessary for `envs.__init__` which registers all robust_gymnasium environments and loads plugins
from robust_gymnasium import envs
from robust_gymnasium import spaces, utils, vector, wrappers, error, logger, functional
from robust_gymnasium import configs

__all__ = [
    # core classes
    "Env",
    "Wrapper",
    "ObservationWrapper",
    "ActionWrapper",
    "RewardWrapper",
    "Space",
    # registration
    "make",
    "make_vec",
    "spec",
    "register",
    "registry",
    "VectorizeMode",
    "pprint_registry",
    "register_envs",
    # module folders
    "envs",
    "spaces",
    "utils",
    "vector",
    "wrappers",
    "error",
    "logger",
    "functional",
    # robust settings
    "configs"
]
__version__ = "0.1.0"


# Initializing pygame initializes audio connections through SDL. SDL uses alsa by default on all Linux systems
# SDL connecting to alsa frequently create these giant lists of warnings every time you import an environment using
#   pygame
# DSP is far more benign (and should probably be the default in SDL anyways)

import os
import sys

if sys.platform.startswith("linux"):
    os.environ["SDL_AUDIODRIVER"] = "dsp"

os.environ["PYGAME_HIDE_SUPPORT_PROMPT"] = "hide"

try:
    from farama_notifications import notifications

    if "robust_gymnasium" in notifications and __version__ in notifications["robust_gymnasium"]:
        print(notifications["robust_gymnasium"][__version__], file=sys.stderr)

except Exception:  # nosec
    pass
