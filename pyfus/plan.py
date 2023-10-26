from dataclasses import dataclass, field
import pyfus.bf as bf

@dataclass
class Plan:
    id: str = "plan"
    name: str = "Plan"
    description: str = ""
    pulse: bf.Pulse = field(default_factory=bf.Pulse)
    sequence: bf.Sequence = field(default_factory=bf.Sequence)
    focal_pattern: bf.FocalPattern = field(default_factory=bf.SingleFocus)
    sim_grid: bf.SimulationGrid = field(default_factory=bf.SimulationGrid)