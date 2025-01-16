import json
from dataclasses import dataclass, field
from typing import Optional, Tuple

from openlifu.io.dict_conversion import DictMixin


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
    TIC: Optional[float] = None
    power_W: Optional[float] = None
    MI: Optional[float] = None
    global_ispta_mWcm2: Optional[float] = None

    @staticmethod
    def from_json(json_string : str) -> "SolutionAnalysis":
        """Load a SolutionAnalysis from a json string"""
        return SolutionAnalysis.from_dict(json.loads(json_string))

    def to_json(self, compact:bool) -> str:
        """Serialize a SolutionAnalysis to a json string

        Args:
            compact: if enabled then the string is compact (not pretty). Disable for pretty.

        Returns: A json string representing the complete SolutionAnalysis object.
        """
        if compact:
            return json.dumps(self.to_dict(), separators=(',', ':'))
        else:
            return json.dumps(self.to_dict(), indent=4)


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
