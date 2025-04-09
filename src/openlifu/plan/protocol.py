from __future__ import annotations

import json
import logging
import math
from copy import deepcopy
from dataclasses import asdict, dataclass, field
from datetime import datetime
from enum import Enum
from pathlib import Path
from typing import Annotated, Any, Dict, List, Tuple

import numpy as np
import xarray as xa

from openlifu import bf, geo, seg, sim, xdc
from openlifu.db.session import Session
from openlifu.geo import Point
from openlifu.plan.solution import Solution
from openlifu.plan.solution_analysis import SolutionAnalysis, SolutionAnalysisOptions
from openlifu.plan.target_constraints import TargetConstraints
from openlifu.sim import run_simulation
from openlifu.util.annotations import OpenLIFUFieldData
from openlifu.util.checkgpu import gpu_available
from openlifu.util.json import PYFUSEncoder
from openlifu.virtual_fit import VirtualFitOptions
from openlifu.xdc import Transducer

OnPulseMismatchAction = Enum("OnPulseMismatchAction", ["ERROR", "ROUND", "ROUNDUP", "ROUNDDOWN"])


@dataclass
class Protocol:
    id: Annotated[str, OpenLIFUFieldData("Protocol ID", "The unique identifier of the protocol")] = "protocol"
    """The unique identifier of the protocol"""

    name: Annotated[str, OpenLIFUFieldData("Protocol name", "The name of the protocol")] = "Protocol"
    """The name of the protocol"""

    description: Annotated[str, OpenLIFUFieldData("Protocol description", "A more detailed description of the protocol")] = ""
    """A more detailed description of the protocol"""

    pulse: Annotated[bf.Pulse, OpenLIFUFieldData("Pulse definition", "The pulse definition used in the protocol")] = field(default_factory=bf.Pulse)
    """The pulse definition used in the protocol"""

    sequence: Annotated[bf.Sequence, OpenLIFUFieldData("Pulse sequence", "The sequence of pulses used in the protocol")] = field(default_factory=bf.Sequence)
    """The sequence of pulses used in the protocol"""

    focal_pattern: Annotated[bf.FocalPattern, OpenLIFUFieldData("Focal pattern", "The focal pattern used in the protocol. By default, a single point is used")] = field(default_factory=bf.SinglePoint)
    """The focal pattern used in the protocol. By default, a single point is used"""

    sim_setup: Annotated[sim.SimSetup, OpenLIFUFieldData("Simulation setup", "Configuration options for using k-wave to simulate wave propagation")] = field(default_factory=sim.SimSetup)
    """Configuration options for using k-wave to simulate wave propagation"""

    delay_method: Annotated[bf.DelayMethod, OpenLIFUFieldData("Delay method", "The method used to calculate transmit delays. By default, delays are calculated using a nominal speed of sound")] = field(default_factory=bf.delay_methods.Direct)
    """The method used to calculate transmit delays. By default, delays are calculated using a nominal speed of sound"""

    apod_method: Annotated[bf.ApodizationMethod, OpenLIFUFieldData("Apodization method", "The method used to calculate transmit apodizations. By default, apodizations are uniform")] = field(default_factory=bf.apod_methods.Uniform)
    """The method used to calculate transmit apodizations. By default, apodizations are uniform"""

    seg_method: Annotated[seg.SegmentationMethod, OpenLIFUFieldData("Segmentation method", "The method used to segment the subject's MRI for delay calculation. By default, the entire field is assumed to be water")] = field(default_factory=seg.seg_methods.Water)
    """The method used to segment the subject's MRI for delay calculation. By default, the entire field is assumed to be water"""

    param_constraints: Annotated[dict, OpenLIFUFieldData("Parameter constraints", "The constraints on the analysis parameters. If computed parameters are outside of the ranges defined here, warnings or errors may be flagged to reject the solution")] = field(default_factory=dict)
    """The constraints on the analysis parameters. If computed parameters are outside of the ranges defined here, warnings or errors may be flagged to reject the solution"""

    target_constraints: Annotated[List[TargetConstraints], OpenLIFUFieldData("Target constraints", "The constraints on the target position. If the target is outside of the bounds defined here, warnings or errors may be flagged to reject the solution")] = field(default_factory=list)
    """The constraints on the target position. If the target is outside of the bounds defined here, warnings or errors may be flagged to reject the solution"""

    analysis_options: Annotated[SolutionAnalysisOptions, OpenLIFUFieldData("Analysis options", "Options to adjust solution analysis. By default, the analysis is configured with default options")] = field(default_factory=SolutionAnalysisOptions)
    """Options to adjust solution analysis. By default, the analysis is configured with default options"""

    virtual_fit_options: Annotated[VirtualFitOptions, OpenLIFUFieldData("Virtual fit options", "Configuration of the virtual fit algorithm")] = field(default_factory=VirtualFitOptions)
    """Configuration of the virtual fit algorithm"""

    def __post_init__(self):
        self.logger = logging.getLogger(__name__)

    @staticmethod
    def from_dict(d : Dict[str,Any]) -> Protocol:
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
        d['param_constraints'] = d.get("param_constraints", {})
        if "target_constraints" in d:
            d['target_constraints'] = [TargetConstraints.from_dict(d_tc) for d_tc in d.get("target_constraints", {})]
        if "virtual_fit_options" in d:
            d['virtual_fit_options'] = VirtualFitOptions.from_dict(d['virtual_fit_options'])
        if "analysis_options" in d:
            if "mainlobe_aspect_ratio" in d["analysis_options"]:
                d["analysis_options"]["mainlobe_aspect_ratio"] = tuple(d["analysis_options"]["mainlobe_aspect_ratio"])
            d['analysis_options'] = SolutionAnalysisOptions.from_dict(d.get("analysis_options"))
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
            "virtual_fit_options": self.virtual_fit_options.to_dict(),
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
    def from_json(json_string : str) -> Protocol:
        """Load a Protocol from a json string"""
        return Protocol.from_dict(json.loads(json_string))

    def to_json(self, compact:bool) -> str:
        """Serialize a Protocol to a json string

        Args:
            compact: if enabled then the string is compact (not pretty). Disable for pretty.

        Returns: A json string representing the complete Protocol object.
        """
        if compact:
            return json.dumps(self.to_dict(), separators=(',', ':'), cls=PYFUSEncoder)
        else:
            return json.dumps(self.to_dict(), indent=4, cls=PYFUSEncoder)

    def to_file(self, filename: str):
        """
        Save the protocol to a file

        Args:
            filename: Name of the file
        """
        Path(filename).parent.parent.mkdir(exist_ok=True)
        Path(filename).parent.mkdir(exist_ok=True)
        with open(filename, 'w') as file:
            file.write(self.to_json(compact=False))


    def check_target(self, target: Point):
        """
        Check if a target is within bounds, raising an exception if it isn't.

        Args:
            target: The geo.Point target to check.
        """
        if isinstance(target, list):
            raise ValueError(f"Input target {target} not supposed to be a list!")

        # check if target position is within target_constraints defined bounds.
        for target_constraint in self.target_constraints:
            pos = target.get_position(
                dim=target_constraint.dim,
                units=target_constraint.units
            )
            target_constraint.check_bounds(pos)

    def fix_pulse_mismatch(self, on_pulse_mismatch: OnPulseMismatchAction, foci: List[Point]):
        """Fix the protocol sequence pulse count in-place given a pulse_mismatch action."""
        if on_pulse_mismatch is OnPulseMismatchAction.ERROR:
            raise ValueError(f"Pulse Count {self.sequence.pulse_count} is not a multiple of the number of foci {len(foci)}")
        else:
            if on_pulse_mismatch is OnPulseMismatchAction.ROUND:
                self.sequence.pulse_count = round(self.sequence.pulse_count / len(foci)) * len(foci)
            elif on_pulse_mismatch is OnPulseMismatchAction.ROUNDUP:
                self.sequence.pulse_count = math.ceil(self.sequence.pulse_count / len(foci)) * len(foci)
            elif on_pulse_mismatch is OnPulseMismatchAction.ROUNDDOWN:
                self.sequence.pulse_count = math.floor(self.sequence.pulse_count / len(foci)) * len(foci)
            self.logger.warning(
                f"Pulse Count {self.sequence.pulse_count} is not a multiple of the number of foci {len(foci)}."
                f"Rounding to {self.sequence.pulse_count}."
            )

    def calc_solution(
        self,
        target: Point,
        transducer: Transducer,
        volume: xa.DataArray | None = None,
        session: Session | None = None,
        simulate: bool = True,
        scale: bool = True,
        sim_options: sim.SimSetup | None = None,
        analysis_options: SolutionAnalysisOptions | None = None,
        on_pulse_mismatch: OnPulseMismatchAction = OnPulseMismatchAction.ERROR,
        use_gpu: bool | None = None,
    ) -> Tuple[Solution, xa.DataArray, SolutionAnalysis]:
        """Calculate the solution and aggregated k-wave simulation outputs.

        Method that computes the delays and apodizations for each focus in the treatment plan,
        simulates the resulting pressure field to adjust transmit pressures to reach target pressures,
        and then analyzes the resulting pressure field to compute the resulting acoustic parameters.

        Args:
            target: The target Point.
                Target is expected to be in the simulation grid coordinates (lat, ele, ax).
            transducer: A Transducer item.
            volume: xa.DataArray
                The subject scan (Default: None).
                It is expected to be in the simulation grid coordinates (lat, ele, ax).
                If None, a default simulation grid will be used.
            session: db.Session
                A session used to define solution_id (Default: None).
            simulate: bool
                Enable solution simulation (Default: true).
            scale: bool
                Triggers solution and simulation scaling to the requested pressure (Default: true).
            sim_options : sim.SimSetup
                The options for the k-wave simulation (Default: self.sim_setup).
            analysis_options: plan.solution.SolutionAnalysisOptions
                The options for the solution analysis (Default: self.analysis_options).
            on_pulse_mismatch: plan.protocol.OnPulseMismatchAction
                An action to take if the number of pulses in the sequence does not match
                the number of foci (Default: OnPulseMismatchAction.ERROR).
            use_gpu: Whether to use GPU in the simulation. If not provided then a GPU will be used
                if available, with CPU as a fallback.

        Returns:
            solution: Solution
            simulation_result_aggregated: xa.Dataset
                If simulation is enabled, then this is the resulting aggregated
                output (max pressure and mean intensity over all foci).
            scaled_solution_analysis: SolutionAnalysis
                This is the resulting rescaled analysis, if scale is enabled.
        """
        if use_gpu is None:
            use_gpu = gpu_available()

        if sim_options is None:
            sim_options = self.sim_setup
        if analysis_options is None:
            analysis_options = self.analysis_options
        # check before if target is within bounds
        self.check_target(target)
        params = sim_options.setup_sim_scene(self.seg_method, volume=volume)

        delays_to_stack: List[np.ndarray] = []
        apodizations_to_stack: List[np.ndarray] = []
        simulation_outputs_to_stack: List[xa.Dataset] = []
        simulation_output_stacked: xa.Dataset = xa.Dataset()
        simulation_result_aggregated: xa.Dataset = xa.Dataset()
        scaled_solution_analysis: SolutionAnalysis = SolutionAnalysis()
        foci: List[Point] = self.focal_pattern.get_targets(target)
        simulation_cycles = np.min([np.round(self.pulse.duration * self.pulse.frequency), 20])

        # updating solution sequence if pulse mismatch
        if (self.sequence.pulse_count % len(foci)) != 0:
            self.fix_pulse_mismatch(on_pulse_mismatch, foci)
        # run simulation and aggregate the results
        for focus in foci:
            self.logger.info(f"Beamform for focus {focus}...")
            delays, apodization = self.beamform(arr=transducer, target=focus, params=params)
            simulation_output_xarray = None
            if simulate:
                self.logger.info(f"Simulate for focus {focus}...")
                simulation_output_xarray, _ = run_simulation(
                    arr=transducer,
                    params=params,
                    delays=delays,
                    apod= apodization,
                    freq = self.pulse.frequency,
                    cycles = simulation_cycles,
                    dt=sim_options.dt,
                    t_end=sim_options.t_end,
                    cfl=sim_options.cfl,
                    amplitude = self.pulse.amplitude,
                    gpu = use_gpu
                )
            delays_to_stack.append(delays)
            apodizations_to_stack.append(apodization)
            simulation_outputs_to_stack.append(simulation_output_xarray)
        if simulate:
            simulation_output_stacked = xa.concat(
                [
                    sim.assign_coords(focal_point_index=i)
                    for i, sim in enumerate(simulation_outputs_to_stack)
                ],
                dim='focal_point_index',
            )
        # instantiate and return the solution
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
            pulse=self.pulse,
            sequence=self.sequence,
            foci=foci,
            target=target,
            simulation_result=simulation_output_stacked,
            approved=False,
            description= (
                f"A solution computed for the {self.name} protocol with transducer {transducer.name}"
                f" for target {target.id}."
                f" This solution was created for the session {session.id} for subject {session.subject_id}." if session is not None else ""
            )
        )
        # optionally scale the solution with simulation result
        if scale:
            if not simulate:
                self.logger.error(msg=f"Cannot scale solution {solution.id} if simulation is not enabled!")
                raise ValueError(f"Cannot scale solution {solution.id} if simulation is not enabled!")
            self.logger.info(f"Scaling solution {solution.id}...")
            #TODO can analysis be an attribute of solution ?
            scaled_solution_analysis = solution.scale(transducer, self.focal_pattern, analysis_options=analysis_options)

        if simulate:
            # Finally the resulting pressure is max-aggregated and intensity is mean-aggregated, over all focus points .
            pnp_aggregated = solution.simulation_result['p_min'].max(dim="focal_point_index")
            ppp_aggregated = solution.simulation_result['p_max'].max(dim="focal_point_index")
            # TODO: Ensure this mean is weighted by the number of times each point is focused on, once openlifu supports hitting points different numbers of times
            intensity_aggregated = solution.simulation_result['intensity'].mean(dim="focal_point_index")
            simulation_result_aggregated = deepcopy(solution.simulation_result)
            simulation_result_aggregated = simulation_result_aggregated.drop_dims("focal_point_index")
            simulation_result_aggregated['p_min'] = pnp_aggregated
            simulation_result_aggregated['p_max'] = ppp_aggregated
            simulation_result_aggregated['intensity'] = intensity_aggregated

        return solution, simulation_result_aggregated, scaled_solution_analysis
