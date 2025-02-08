import math
from dataclasses import MISSING

from omni.isaac.lab.managers import CommandTermCfg
from omni.isaac.lab.utils import configclass

from .gait_command_quad import GaitCommandQuad  # Import the GaitCommandQuad class


@configclass
class UniformGaitCommandCfgQuad(CommandTermCfg):
    """Configuration for the gait command generator."""

    class_type: type = GaitCommandQuad  # Specify the class type for dynamic instantiation

    @configclass
    class Ranges:
        """Uniform distribution ranges for the gait parameters."""

        frequencies: tuple[float, float] = MISSING
        """Range for gait frequencies [Hz]."""
        durations: tuple[float, float] = MISSING
        """Range for contact durations [0-1]."""
        offsets2: tuple[float, float] = MISSING
        """Range for phase offsets [0-1]."""
        offsets3: tuple[float, float] = MISSING
        """Range for phase offsets [0-1]."""
        offsets4: tuple[float, float] = MISSING
        """Range for phase offsets [0-1]."""


    ranges: Ranges = MISSING
    """Distribution ranges for the gait parameters."""

    resampling_time_range: tuple[float, float] = MISSING
    """Time interval for resampling the gait (in seconds)."""
