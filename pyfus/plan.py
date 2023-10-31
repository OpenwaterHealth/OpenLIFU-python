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
    bf_plan: bf.BeamformingPlan = field(default_factory=bf.BeamformingPlan)
    param_constraints: dict = field(default_factory=dict)
    target_constraints: dict = field(default_factory=dict)
    analysis_options: dict = field(default_factory=dict)

    @staticmethod
    def from_dict(d):
        d["pulse"] = bf.Pulse.from_dict(d.get("pulse", {}))
        d["sequence"] = bf.Sequence.from_dict(d.get("sequence", {}))
        d["focal_pattern"] = bf.FocalPattern.from_dict(d.get("focal_pattern", {}))
        d["sim_grid"] = bf.SimulationGrid.from_dict(d.get("sim_grid", {}))
        d["bf_plan"] = bf.BeamformingPlan.from_dict(d.get("bf_plan", {}))
        return Plan(**d)



