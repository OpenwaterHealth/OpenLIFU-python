import json
from dataclasses import asdict, dataclass, field
from datetime import datetime
from pathlib import Path
from typing import TYPE_CHECKING, Any, Dict, Tuple

import numpy as np
import xarray as xa

from openlifu import bf, geo, seg, sim, xdc
from openlifu.geo import Point
from openlifu.plan.solution import Solution
from openlifu.sim import run_simulation
from openlifu.xdc import Transducer

if TYPE_CHECKING:
    from openlifu.db import Session


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
        Path(filename).parent.parent.mkdir(exist_ok=True)
        Path(filename).parent.mkdir(exist_ok=True)
        with open(filename, 'w') as file:
            file.write(self.to_json(compact=False))

    def calc_solution(
            self,
            transducer:Transducer,
            volume:xa.DataArray,
            target: Point,
            session:"Optional[Session]"=None, # useful in solution id
        ) -> Tuple[Solution, xa.DataArray, xa.DataArray]:
        params = self.seg_method.seg_params(volume)
        delays_to_stack : List[np.ndarray] = []
        apodizations_to_stack : List[np.ndarray] = []
        simulation_outputs_to_stack : List[xa.Dataset] = []
        target_pattern_points : List[Point] = self.focal_pattern.get_targets(target)
        for focus_point in target_pattern_points:
            delays, apodization = self.beamform(arr=transducer, target=focus_point, params=params)

            simulation_output_xarray, simulation_output_kwave = run_simulation(
                arr=transducer,
                params=params,
                delays=delays,
                apod= apodization,
                freq = self.pulse.frequency,
                cycles = np.max([np.round(self.pulse.duration * self.pulse.frequency), 20]),
                dt=self.sim_setup.dt,
                t_end=self.sim_setup.t_end,
                amplitude = 1,
                gpu = False
            )

            delays_to_stack.append(delays)
            apodizations_to_stack.append(apodization)
            simulation_outputs_to_stack.append(simulation_output_xarray)

        simulation_output_stacked = xa.concat(
            [
                sim.assign_coords(focal_point_index=i)
                for i,sim in enumerate(simulation_outputs_to_stack)
            ],
            dim='focal_point_index',
        )

        # Peak negative pressure volume, a simulation output. This is max-aggregated over all focus points.
        pnp_aggregated = simulation_output_stacked['p_min'].max(dim="focal_point_index")

        # Mean-aggregate the intensity over the focus points
        # TODO: Ensure this mean is weighted by the number of times each point is focused on, once openlifu supports hitting points different numbers of times
        intensity_aggregated = simulation_output_stacked['ita'].mean(dim="focal_point_index")

        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
        solution_id = timestamp
        if session is not None:
            solution_id = f"{session.id}_{solution_id}"
        solution =  Solution(
            id=solution_id,
            name=f"Solution {timestamp}",
            protocol_id=self.id,
            transducer_id=transducer.id,
            delays=np.stack(delays_to_stack, axis=0),
            apodizations=np.stack(apodizations_to_stack, axis=0),
            pulse=self.pulse, # TODO This pulse needs to be scaled via a port of scale_solution from matlab!!
            sequence=self.sequence, # TODO is it correct to set the sequence the same as the protocol's here?
            foci=target_pattern_points,
            target=target,
            simulation_result=simulation_output_stacked,
            approved=False,
            description= (
                f"A solution computed for the {self.name} protocol with transducer {transducer.name}"
                f" for subject volume [TODO]" # TODO put volume ID here if it is not None, once Sadhana's PR #123 is merged
                f" for target {target.id}."
                f" This solution was created for the session {session.id} for subject {session.subject_id}." if session is not None else ""
            )
        )
        return solution, pnp_aggregated, intensity_aggregated
