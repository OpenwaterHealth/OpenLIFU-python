import json
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Dict

import xarray as xa

from openlifu import bf, geo, seg, sim, xdc


@dataclass
class Protocol:
    id: str = "protocol"
    name: str = "Protocol"
    description: str = ""
    pulse: bf.Pulse = field(default_factory=bf.Pulse)
    sequence: bf.Sequence = field(default_factory=bf.Sequence)
    focal_pattern: bf.FocalPattern = field(default_factory=bf.SinglePoint)
    sim_setup: sim.SimSetup = field(default_factory=sim.SimSetup)
    delay_method: bf.DelayMethod = field(default_factory=bf.delay_methods.Direct)
    apod_method: bf.ApodizationMethod = field(default_factory=bf.apod_methods.Uniform)
    seg_method: seg.SegmentationMethod = field(default_factory=seg.seg_methods.Water)
    param_constraints: dict = field(default_factory=dict)
    target_constraints: dict = field(default_factory=dict)
    analysis_options: dict = field(default_factory=dict)

    @staticmethod
    def from_dict(d : Dict[str,Any]) -> "Protocol":
        d["pulse"] = bf.Pulse.from_dict(d.get("pulse", {}))
        d["sequence"] = bf.Sequence.from_dict(d.get("sequence", {}))
        d["focal_pattern"] = bf.FocalPattern.from_dict(d.get("focal_pattern", {}))
        d["sim_setup"] = sim.SimSetup.from_dict(d.get("sim_setup", {}))
        d["delay_method"] = bf.DelayMethod.from_dict(d.get("delay_method", {}))
        d["apod_method"] = bf.ApodizationMethod.from_dict(d.get("apod_method", {}))
        seg_method_dict = d.get("seg_method", {})
        if "materials" in d:
            seg_method_dict["materials"] = seg.Material.from_dict(d.pop("materials"))
        d["seg_method"] = seg.SegmentationMethod.from_dict(seg_method_dict)
        return Protocol(**d)

    def to_dict(self):
        return {
            "id": self.id,
            "name": self.name,
            "description": self.description,
            "pulse": self.pulse.to_dict(),
            "sequence": self.sequence.to_dict(),
            "focal_pattern": self.focal_pattern.to_dict(),
            "sim_setup": asdict(self.sim_setup),
            "delay_method": self.delay_method.to_dict(),
            "apod_method": self.apod_method.to_dict(),
            "seg_method": self.seg_method.to_dict(),
            "param_constraints": self.param_constraints,
            "target_constraints": self.target_constraints,
            "analysis_options": self.analysis_options,
        }

    @staticmethod
    def from_file(filename):
        with open(filename) as f:
            d = json.load(f)
        return Protocol.from_dict(d)

    def beamform(self, arr: xdc.Transducer, target:geo.Point, params: xa.Dataset):
        delays = self.delay_method.calc_delays(arr, target, params)
        apod = self.apod_method.calc_apodization(arr, target, params)
        return delays, apod

    @staticmethod
    def from_json(json_string : str) -> "Protocol":
        """Load a Protocol from a json string"""
        return Protocol.from_dict(json.loads(json_string))

    def to_json(self, compact:bool) -> str:
        """Serialize a Protocol to a json string

        Args:
            compact: if enabled then the string is compact (not pretty). Disable for pretty.

        Returns: A json string representing the complete Protocol object.
        """
        if compact:
            return json.dumps(self.to_dict(), separators=(',', ':'))
        else:
            return json.dumps(self.to_dict(), indent=4)

    def to_file(self, filename):
        """
        Save the protocol to a file

        :param filename: Name of the file
        """
        Path(filename).parent.mkdir(exist_ok=True)
        with open(filename, 'w') as file:
            file.write(self.to_json(compact=False))
