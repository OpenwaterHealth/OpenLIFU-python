from dataclasses import dataclass, field
import pyfus.bf as bf
import json

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

    def to_dict(self):
        return {
            "id": self.id,
            "name": self.name,
            "description": self.description,
            "pulse": self.pulse.to_dict(),
            "sequence": self.sequence.to_dict(),
            "focal_pattern": self.focal_pattern.to_dict(),
            "sim_grid": self.sim_grid.to_dict(),
            "bf_plan": self.bf_plan.to_dict(),
            "param_constraints": self.param_constraints,
            "target_constraints": self.target_constraints,
            "analysis_options": self.analysis_options,
        }
    
    @staticmethod
    def from_file(filename):
        with open(filename, "r") as f:
            d = json.load(f)
        return Plan.from_dict(d)
    

