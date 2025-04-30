from __future__ import annotations

import pytest

from openlifu.plan.param_constraint import ParameterConstraint
from openlifu.plan.solution_analysis import SolutionAnalysis, SolutionAnalysisOptions

# ---- Tests for SolutionAnalysis ----

@pytest.fixture()
def example_solution_analysis() -> SolutionAnalysis:
    return SolutionAnalysis(
        mainlobe_pnp_MPa=[1.1, 1.2],
        mainlobe_isppa_Wcm2=[10.0, 12.0],
        mainlobe_ispta_mWcm2=[500.0, 520.0],
        beamwidth_lat_3dB_mm=[1.5, 1.6],
        beamwidth_ele_3dB_mm=[2.0, 2.1],
        beamwidth_ax_3dB_mm=[3.0, 3.1],
        beamwidth_lat_6dB_mm=[1.8, 1.9],
        beamwidth_ele_6dB_mm=[2.5, 2.6],
        beamwidth_ax_6dB_mm=[3.5, 3.6],
        sidelobe_pnp_MPa=[0.5, 0.6],
        sidelobe_isppa_Wcm2=[5.0, 5.5],
        global_pnp_MPa=[1.3],
        global_isppa_Wcm2=[13.0],
        p0_MPa=[1.0, 1.1],
        TIC=0.7,
        power_W=25.0,
        MI=1.2,
        global_ispta_mWcm2=540.0,
        param_constraints={
            "global_pnp_MPa": ParameterConstraint(
                operator="<=",
                warning_value=1.4,
                error_value=1.6
            )
        }
    )

def test_to_dict_from_dict_solution_analysis(example_solution_analysis: SolutionAnalysis):
    sa_dict = example_solution_analysis.to_dict()
    new_solution = SolutionAnalysis.from_dict(sa_dict)
    assert new_solution == example_solution_analysis

@pytest.mark.parametrize("compact", [True, False])
def test_serialize_deserialize_solution_analysis(example_solution_analysis: SolutionAnalysis, compact: bool):
    json_str = example_solution_analysis.to_json(compact)
    deserialized = SolutionAnalysis.from_json(json_str)
    assert deserialized == example_solution_analysis


# ---- Tests for SolutionAnalysisOptions ----


@pytest.fixture()
def example_solution_analysis_options() -> SolutionAnalysisOptions:
    return SolutionAnalysisOptions(
        standoff_sound_speed=1480.0,
        standoff_density=990.0,
        ref_sound_speed=1540.0,
        ref_density=1020.0,
        mainlobe_aspect_ratio=(1.0, 1.0, 4.0),
        mainlobe_radius=2.0e-3,
        beamwidth_radius=4.0e-3,
        sidelobe_radius=2.5e-3,
        sidelobe_zmin=0.5e-3,
        distance_units="mm",
        param_constraints={
            "mainlobe_radius": ParameterConstraint(
                operator=">=",
                warning_value=1.5e-3,
                error_value=1.0e-3
            )
        }
    )

def test_to_dict_from_dict_solution_analysis_options(example_solution_analysis_options: SolutionAnalysisOptions):
    options_dict = example_solution_analysis_options.to_dict()
    new_options = SolutionAnalysisOptions.from_dict(options_dict)
    assert new_options == example_solution_analysis_options
