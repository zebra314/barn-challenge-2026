from dataclasses import dataclass
import numpy as np
from typing import Optional

@dataclass
class HallucinationConfig:
    """Parameters specific to hallucination/obstacle generation."""
    obstacle_density: float = 0.0
    noise_level: float = 0.0
    hallucination_type: str = "none"
    # Add other hallucination-specific params here
