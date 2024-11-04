import base64
import json
from dataclasses import asdict, dataclass, field
from datetime import datetime
from pathlib import Path
from typing import List, Optional, Tuple

import numpy as np
import xarray as xa

from openlifu.bf import Pulse, Sequence, mask_focus
from openlifu.bf.mask_focus import MaskOp
from openlifu.geo import Point
from openlifu.io.dict_conversion import DictMixin
from openlifu.util.json import PYFUSEncoder
from openlifu.util.units import rescale_data_arr
from openlifu.xdc import Transducer


def _construct_nc_filepath_from_json_filepath(json_filepath:Path) -> Path:
    """Construct a default filepath to netCDF file given filepath to associated solution json file."""
    nc_filename = json_filepath.name.split(".")[0] + ".nc"
    nc_filepath = json_filepath.parent / nc_filename
    return nc_filepath


@dataclass
class SolutionAnalysis(DictMixin):
    mainlobe_pnp_MPa: list[float] = field(default_factory=list)
    mainlobe_isppa_Wcm2: list[float] = field(default_factory=list)
    mainlobe_ispta_mWcm2: list[float] = field(default_factory=list)
    beamwidth_lat_3dB_mm: list[float] = field(default_factory=list)
    beamwidth_ax_3dB_mm: list[float] = field(default_factory=list)
    beamwidth_lat_6dB_mm: list[float] = field(default_factory=list)
    beamwidth_ax_6dB_mm: list[float] = field(default_factory=list)
    sidelobe_pnp_MPa: list[float] = field(default_factory=list)
    sidelobe_isppa_Wcm2: list[float] = field(default_factory=list)
    global_pnp_MPa: list[float] = field(default_factory=list)
    global_isppa_Wcm2: list[float] = field(default_factory=list)
    p0_Pa: list[float] = field(default_factory=list)
    TIC: float = None
    power_W: float = None
    MI: float = None
    global_ispta_mWcm2: float = None


@dataclass
class SolutionAnalysisOptions(DictMixin):
    standoff_sound_speed: float = 1500.0
    standoff_density: float = 1000.0
    ref_sound_speed: float = 1500.0
    ref_density: float = 1000.0
    focus_diameter: float = 0.5
    mainlobe_aspect_ratio: Tuple[float, float, float] = (1., 1., 5.)
    mainlobe_radius: float = 2.5e-3
    beamwidth_radius: float = 5e-3
    sidelobe_radius: float = 3e-3
    sidelobe_zmin: float = 1e-3
    distance_units: str = "m"


@dataclass
class Solution:
    """
    A sonication solution resulting from beamforming and running a simulation.
    """
    id: str = "solution"  # the *solution* id, a concept that did not exist in the matlab software
    """ID of this solution"""

    name: str = "Solution"
    """Name of this solution"""

    protocol_id: Optional[str] = None  # this used to be called plan_id in the matlab code
    """ID of the protocol that was used when generating this solution"""

    transducer_id: Optional[str] = None
    """ID of the transducer that was used when generating this solution"""

    created_on: datetime = field(default_factory=datetime.now)
    """Solution creation time"""

    description: str = ""
    """Description of this solution"""

    delays: Optional[np.ndarray] = None
    """Vectors of time delays to steer the beam. Shape is (number of foci, number of transducer elements)."""

    apodizations: Optional[np.ndarray] = None
    """Vectors of apodizations to steer the beam. Shape is (number of foci, number of transducer elements)."""

    pulse: Pulse = field(default_factory=Pulse)
    """Pulse to send to the transducer when running sonication"""

    sequence: Sequence = field(default_factory=Sequence)
    """Pulse sequence to use when running sonication"""

    foci: List[Point] = field(default_factory=list)
    """Points that are focused on in this Solution due to the focal pattern around the target.
    Each item in this list is a unique point from the focal pattern, and the pulse sequence is
    what determines how many times each point will be used.
    """

    # there was "target_id" in the matlab software, but here we do not have the concept of a target ID.
    # I believe this was only needed in the matlab software because solutions were organized by target rather
    # than having their own unique solution ID. We do have unique solution IDs so it's possible we don't need
    # this target attribute at all here. Keeping it here for now just in case.
    target: Optional[Point] = None
    """The ultimate target of this sonication. This sonication solution is focused on one focal point
    in a pattern that is centered on this target."""

    # In the matlab code the simulation result was saved as a separate .mat file.
    # Here we include it as an xarray dataset.
    simulation_result: xa.Dataset = field(default_factory=xa.Dataset)
    """The xarray Dataset of simulation results"""

    approved: bool = False
    """Approval state of this solution as a sonication plan. `True` means the user has provided some
    kind of confirmation that the solution is safe and acceptable to be executed."""

    def num_foci(self) -> int:
        """Get the number of foci"""
        return len(self.foci)

    def analyze(self, transducer: Transducer, options: SolutionAnalysisOptions = SolutionAnalysisOptions()) -> SolutionAnalysis:
        """Analyzes the treatment solution.

        Args:
            transducer: A Transducer item.
            options: A struct for solution analysis options.

        Returns: A struct containing the results of the analysis.
        """
        solution_analysis = SolutionAnalysis()

        if transducer.id != self.transducer_id:
            # self.logger.error(f"Provided transducer id {transducer.id} does not match Solution transducer id: {self.transducer_id}")
            raise ValueError(f"Provided transducer id {transducer.id} does not match Solution transducer id: {self.transducer_id}")

        # dt = 1 / (self.pulse.frequency * 20)
        # t = self.pulse.calc_time(dt)
        # input_signal = self.pulse.calc_pulse(t)

        pnp = self.simulation_result.p_min
        ppp = self.simulation_result.p_max
        ita = self.simulation_result.ita

        if options.sidelobe_radius is np.nan:
            options.sidelobe_radius = options.mainlobe_radius

        pnp_MPa = rescale_data_arr(pnp, "MPa")

        # standoff_Z = options.standoff_density * 1500
        # c_tic = 40e-3  # W cm-1
        # A_cm = transducer.get_area("cm")
        # d_eq_cm = np.sqrt(4*A_cm / np.pi)
        # ele_sizes_cm2 = np.array([elem.get_area("cm") for elem in transducer.elements])

        # xyz = np.stack(np.meshgrid(*coords, indexing="xy"), axis=-1)  #TODO: if fus.Axis is defined, coords.ndgrid(dim="ax")
        # z_mask = xyz[..., -1] >= options.sidelobe_zmin  #TODO: probably wrong here, should be z{1}>=options.sidelobe_zmin;

        # intensity_Wcm2 = output.intensity.rescale_data("W/cm^2")
        # pulsetrain_dutycycle = self.get_pulsetrain_dutycycle()
        # treatment_dutycycle = self.get_treatment_dutycycle()
        # ita_mWcm2 = self.get_ita(output, units="W/cm^2")

        # power_W = np.zeros(self.num_foci())
        # TIC = np.zeros(self.num_foci())
        for focus_index in range(self.num_foci()):
            foc = self.foci[focus_index]
            # output_signal = []
            # output_signal = np.zeros((transducer.numelements(), len(input_signal)))
            # for i in range(transducer.numelements()):
            #     apod_signal = input_signal * self.apodizations[focus_index, i]
            #     output_signal[i] = transducer.elements[i].calc_output(apod_signal, dt)

            # p0_Pa = np.max(output_signal)

            # get focus region masks (for mainlobe, sidelobe and beamwidth)
            mainlobe_mask = mask_focus(
                self.simulation_result,  #TODO: Original code uses coords, but too complicated to maniplulate a Coordinates class
                foc,
                options.mainlobe_radius,
                mask_op=MaskOp.LESS_EQUAL,
                units=options.distance_units,
                aspect_ratio=options.mainlobe_aspect_ratio
            )
            # mask_options['operation'] = ">="
            # mask_options['zmin'] = options.sidelobe_zmin
            # sidelobe_mask = mask_focus(
            #     coords=coords,
            #     coords_units="m",  #TODO: currently hard-coded because sol.simulation_result.coords does not store units
            #     focus=foc,
            #     distance=options.sidelobe_radius,
            #     options=mask_options)
            # mask_options['operation'] = "<="
            # beamwidth_mask = mask_focus(
            #     coords=coords,
            #     coords_units="m",  #TODO: currently hard-coded because sol.simulation_result.coords does not store units
            #     focus=foc,
            #     distance=options.beamwidth_radius,
            #     options=mask_options)

            pk = np.max(pnp_MPa.data[focus_index] * mainlobe_mask)  #TODO: pnp_MPa supposed to be a list for each focus: pnp_MPa(focus_index)
            solution_analysis.mainlobe_pnp_MPa += [pk]

        #     thresh_m3dB = pk*10**(-3 / 20)
        #     thresh_m6dB = pk*10**(-6 / 20)
        #     beamwidth_options = {
        #         'dims': (0, 1),
        #         'mask': beamwidth_mask,
        #         'units': "mm"
        #     }
        #     bw3xy = get_beamwidth(
        #         pnp_MPa,  #TODO: pnp_MPa supposed to be a list for each focus: pnp_MPa(focus_index)
        #         coords_units="m",
        #         focus=foc,
        #         cutoff=thresh_m3dB,
        #         options=beamwidth_options
        #     )
        #     beamwidth_options['dims'] = (2)
        #     bw3z = get_beamwidth(
        #         pnp_MPa,  #TODO: pnp_MPa supposed to be a list for each focus: pnp_MPa(focus_index)
        #         coords_units="m",
        #         focus=foc,
        #         cutoff=thresh_m3dB,
        #         options=beamwidth_options
        #     )
        #     beamwidth_options['dims'] = (0, 1)
        #     bw6xy = get_beamwidth(
        #         pnp_MPa,  #TODO: pnp_MPa supposed to be a list for each focus: pnp_MPa(focus_index)
        #         coords_units="m",
        #         focus=foc,
        #         cutoff=thresh_m6dB,
        #         options=beamwidth_options
        #     )
        #     beamwidth_options['dims'] = (2)
        #     bw6z = get_beamwidth(
        #         pnp_MPa,  #TODO: pnp_MPa supposed to be a list for each focus: pnp_MPa(focus_index)
        #         coords_units="m",
        #         focus=foc,
        #         cutoff=thresh_m6dB,
        #         options=beamwidth_options
        #     )
        #     solution_analysis.mainlobe_isppa_Wcm2 += [np.max(intensity_Wcm2.data * mainlobe_mask)]
        #     solution_analysis.mainlobe_ispta_mWcm2 += [np.max(ita_mWcm2.data * mainlobe_mask)]
        #     solution_analysis.beamwidth_lat_3dB_mm += [bw3xy.beamwidth]
        #     solution_analysis.beamwidth_ax_3dB_mm += [bw3z.beamwidth]
        #     solution_analysis.beamwidth_lat_6dB_mm += [bw6xy.beamwidth]
        #     solution_analysis.beamwidth_ax_6dB_mm += [bw6z.beamwidth]

        #     solution_analysis.sidelobe_pnp_MPa += [np.max(pnp_MPa.data * sidelobe_mask)]
        #     solution_analysis.sidelobe_isppa_Wcm2 += [np.max(intensity_Wcm2.data * sidelobe_mask)]
        #     solution_analysis.global_pnp_MPa += [np.max(pnp_MPa.data * z_mask)]
        #     solution_analysis.global_isppa_Wcm2 += [np.max(intensity_Wcm2.data * z_mask)]
        #     i0_Wcm2 = (p0_Pa**2 / (2*standoff_Z)) * 1e-4
        #     i0ta_Wcm2 = i0_Wcm2*pulsetrain_dutycycle*treatment_dutycycle
        #     power_W[focus_index] = np.mean(np.sum(i0ta_Wcm2*ele_sizes_cm2*self.apodizations[focus_index, :]))
        #     TIC[focus_index] = c_tic*power_W[focus_index]/d_eq_cm
        #     solution_analysis.p0_Pa += [np.max(p0_Pa)]
        # solution_analysis.TIC = np.mean(TIC)
        # solution_analysis.power_W = np.mean(power_W)
        # solution_analysis.MI = solution_analysis.mainlobe_pnp_MPa/np.sqrt(self.pulse.frequency*1e-6)
        # ita_mWcm2 = self.get_ita(output)
        # solution_analysis.global_ispta_mWcm2 = np.max(ita_mWcm2.data*z_mask)

        return solution_analysis

    def get_pulsetrain_dutycycle(self) -> float:
        """
        Compute the pulsetrain dutycycle given a sequence.

        Returns:
            A float.
        """
        pulsetrain_dutycycle = min(1., self.pulse.duration / self.sequence.pulse_interval)

        return pulsetrain_dutycycle

    def get_treatment_dutycycle(self) -> float:
        """
        Compute the treatment dutycycle given a sequence.

        Returns:
            A float.
        """
        treatment_dutycycle = min(
            1, (self.sequence.pulse_count * self.sequence.pulse_interval) / self.sequence.pulse_train_interval
        )

        return treatment_dutycycle

    def get_ita(self, output: dict, units: str = "mW/cm^2") -> xa.DataArray:
        """
        Calculate the intensity-time-area product for a treatment solution.

        Args:
            output: A struct for simulation results from the treatment.
            units: str
                Target units. Default "mW/cm^2".

        Returns:
            A Solution instance with the calculated ita value.
        """
        intensity_scaled = output.intensity.rescale_data(units)
        pulsetrain_dutycycle = self.get_pulsetrain_dutycycle()
        treatment_dutycycle = self.get_treatment_dutycycle()
        pulse_seq = (np.arange(self.sequence.pulse_count) - 1) % self.num_foci() + 1
        counts = np.zeros((1, 1, 1, self.num_foci()))
        for i in range(self.num_foci()):
            counts[0, 0, 0, i] = np.sum(pulse_seq == (i+1))
        ita = intensity_scaled.copy(deep=True)
        isppa_avg = np.sum(np.expand_dims(ita.data, axis=-1) * counts, axis=-1) / np.sum(counts)
        ita.data = isppa_avg * pulsetrain_dutycycle * treatment_dutycycle

        return ita

    def to_json(self, include_simulation_data: bool, compact: bool) -> str:
        """Serialize a Solution to a json string

        Args:
            include_array_data: if enabled then large simulation data arrays are serialized somehow into the json,
                so that they can be recovered via `from_json` alone. otherwise they are excluded.
            compact: if enabled then the string is compact (not pretty). Disable for pretty.

        Returns: A json string representing the complete Solution object.
        """
        solution_dict = asdict(self)

        if include_simulation_data:
            # Serialize xarray dataset into a string
            solution_dict['simulation_result'] = base64.b64encode(self.simulation_result.to_netcdf(engine='scipy')).decode('utf-8')
        else:
            solution_dict.pop('simulation_result')

        if compact:
            return json.dumps(solution_dict, separators=(',', ':'), cls=PYFUSEncoder)
        else:
            return json.dumps(solution_dict, indent=4, cls=PYFUSEncoder)

    @staticmethod
    def from_json(json_string : str, simulation_result: Optional[xa.Dataset]=None) -> "Solution":
        """Load a Solution from a json string.

        Args:
            json_string: the json string defining the Solution
            simulation_result: the simulation result arrays to use. If the json string has this then it will
                be read from the json string and it should not be provided in this argument.

        Returns: The new Solution object.
        """
        solution_dict = json.loads(json_string)
        solution_dict["created_on"] = datetime.fromisoformat(solution_dict["created_on"])
        if solution_dict["delays"] is not None:
            solution_dict["delays"] = np.array(solution_dict["delays"])
        if solution_dict["apodizations"] is not None:
            solution_dict["apodizations"] = np.array(solution_dict["apodizations"], ndmin=2)
            solution_dict["apodizations"] = np.array(solution_dict["apodizations"], ndmin=2)
        solution_dict["pulse"] = Pulse.from_dict(solution_dict["pulse"])
        solution_dict["sequence"] = Sequence.from_dict(solution_dict["sequence"])
        solution_dict["foci"] = [
            Point.from_dict(focus_dict)
            for focus_dict in solution_dict["foci"]
        ]
        if solution_dict["target"] is not None:
            solution_dict["target"] = Point.from_dict(solution_dict["target"])

        if simulation_result is not None:
            if "simulation_result" in solution_dict:
                raise ValueError(
                    "A simulation result was provided while the json string already contains `simulation_result`. "
                    "Unclear which to use!"
                )
            solution_dict["simulation_result"] = simulation_result
        elif "simulation_result" in solution_dict:
            # Deserialize xarray dataset from string
            solution_dict["simulation_result"] = xa.open_dataset(
                base64.b64decode(
                    solution_dict["simulation_result"].encode('utf-8')
                ),
                engine='scipy',
            )

        return Solution(**solution_dict)

    def to_files(self, json_filepath:Path, nc_filepath:Optional[Path]=None) -> None:
        """Save the solution to json and netCDF files.

        json_filepath: where to save the json file with all data except the simulation results dataset
        nc_filepath: where to save the netCDF file containing the simulation results.
            If None then it will be saved in the same directory as the json file, with the same name but with
            the extension *.nc
        """
        if nc_filepath is None:
            nc_filepath = _construct_nc_filepath_from_json_filepath(json_filepath)
        json_filepath.parent.mkdir(parents=True, exist_ok=True)
        nc_filepath.parent.mkdir(parents=True, exist_ok=True)
        with json_filepath.open("w") as json_file:
            json_file.write(
                self.to_json(include_simulation_data=False, compact=False)
            )
        self.simulation_result.to_netcdf(nc_filepath, engine='h5netcdf')

    @staticmethod
    def from_files(json_filepath:Path, nc_filepath:Optional[Path]=None):
        """Read solution from json and netCDF files.

        json_filepath: solution json file location, containing all data except the simulation results dataset
        nc_filepath: netCDF file location, containing the simulation results.
            If None then it will be assumed to be saved in the same directory as the json file, with the same name but with
            the extension *.nc. This is the default saving behavior of Solution.to_files.
        """
        if nc_filepath is None:
            nc_filepath = _construct_nc_filepath_from_json_filepath(json_filepath)
        simulation_result = xa.open_dataset(nc_filepath, engine='h5netcdf').load()
        simulation_result.close() # this is needed to release the lock on the file so that it can be written to again
        return Solution.from_json(
            json_string = json_filepath.read_text(),
            simulation_result = simulation_result,
        )
