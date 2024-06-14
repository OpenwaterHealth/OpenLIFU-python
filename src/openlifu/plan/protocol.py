from dataclasses import dataclass, field, InitVar
from typing import List
from openlifu import bf, sim, seg, xdc, geo
import json
import xarray as xa

@dataclass
class Protocol:
    id: str = "protocol"
    name: str = "Protocol"
    description: str = ""
    pulse: bf.Pulse = field(default_factory=bf.Pulse)
    sequence: bf.Sequence = field(default_factory=bf.Sequence)
    focal_pattern: bf.FocalPattern = field(default_factory=bf.FocalPattern)
    sim_setup: sim.SimSetup = field(default_factory=sim.SimSetup)
    delay_method: bf.DelayMethod = field(default_factory=bf.delay_methods.Direct)
    apod_method: bf.ApodizationMethod = field(default_factory=bf.apod_methods.Uniform)
    seg_method: seg.SegmentationMethod = field(default_factory=seg.seg_methods.Water)
    param_constraints: dict = field(default_factory=dict)
    target_constraints: dict = field(default_factory=dict)
    analysis_options: dict = field(default_factory=dict)    

    @staticmethod
    def from_dict(d):
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
            "sim_setup": self.sim_setup.to_dict(),
            "delay_method": self.delay_method.to_dict(),
            "apod_method": self.apod_method.to_dict(),
            "seg_method": self.seg_method.to_dict(),
            "param_constraints": self.param_constraints,
            "target_constraints": self.target_constraints,
            "analysis_options": self.analysis_options,
        }
    
    @staticmethod
    def from_file(filename):
        with open(filename, "r") as f:
            d = json.load(f)
        return Protocol.from_dict(d)
    
    def beamform(self, arr: xdc.Transducer, target:geo.Point, params: xa.Dataset):
        delays = self.delay_method.calc_delays(arr, target, params)
        apod = self.apod_method.calc_apodization(arr, target, params)
        return delays, apod