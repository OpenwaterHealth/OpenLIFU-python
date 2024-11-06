from pathlib import Path

import numpy as np
import pytest
import xarray as xa

from openlifu import Protocol, Transducer
from openlifu.db import Session


@pytest.fixture()
def example_protocol() -> Protocol:
    return Protocol.from_file(Path(__file__).parent/'resources/example_db/protocols/example_protocol/example_protocol.json')

@pytest.fixture()
def example_transducer() -> Transducer:
    return Transducer.from_file(Path(__file__).parent/"resources/example_db/transducers/example_transducer/example_transducer.json")

@pytest.fixture()
def example_session() -> Session:
    return Session.from_file(Path(__file__).parent/"resources/example_db/subjects/example_subject/sessions/example_session/example_session.json")

@pytest.fixture()
def example_transducer_transform() -> np.ndarray:
    return np.array(
        [
            [-1,    0,	0,	0],
            [0,	0.05,	0.998749217771909,	-105],
            [0,	0.998749217771909,	-0.05,	5],
            [0,	0,	0,	1]
        ]
    )

@pytest.fixture()
def example_volume() -> xa.Dataset:
    return xa.Dataset()
    #TODO the following is in case we want to test a nifti input volume
    # loading a nifti file and then construction a xarray dataset from it
    # with open(Path(__file__).parent/"resources/example_db/subjects/example_subject/volumes/example_volume/mni.json") as f:
    #     nib_volume_metadata = json.load(f)
    # nib_volume = Nifti1Image.load(Path(__file__).parent/"resources/example_db/subjects/example_subject/volumes/example_volume"/nib_volume_metadata['data_filename'])
    # l_min, p_min, s_min = nib_volume.affine[:3, -1]
    # vol_shape = nib_volume.shape
    # l_max = affines.apply_affine(nib_volume.affine, [vol_shape[0]-1, 0, 0])[0]
    # p_max = affines.apply_affine(nib_volume.affine, [0, vol_shape[1]-1, 0])[1]
    # s_max = affines.apply_affine(nib_volume.affine, [0, 0, vol_shape[2]-1])[2]
    # volume = xa.Dataset(
    #     {
    #         'data': xa.DataArray(data=nib_volume.get_fdata(), dims=["L", "P", "S"])
    #     },
    #     coords={
    #         'L': xa.DataArray(dims=["L"], data=np.linspace(l_min, l_max, vol_shape[0]), attrs={'units': "mm"}),
    #         'P': xa.DataArray(dims=["P"], data=np.linspace(p_min, p_max, vol_shape[1]), attrs={'units': "mm"}),
    #         'S': xa.DataArray(dims=["S"], data=np.linspace(s_min, s_max, vol_shape[2]), attrs={'units': "mm"})
    #     },
    #     attrs={
    #         'units': "",
    #         'name': "Test Volume",
    #         'id': "volume_000",
    #         'affine': nib_volume.affine,
    #         'affine_units': "mm"
    #     }
    # )

@pytest.mark.parametrize("compact_representation", [True, False])
def test_serialize_deserialize_protocol(example_protocol : Protocol, compact_representation: bool):
    assert example_protocol.from_json(example_protocol.to_json(compact_representation)) == example_protocol

def test_default_protocol():
    """Ensure it is possible to construct a default protocol"""
    Protocol()

def test_check_target(example_protocol: Protocol, example_session: Session):
    """Ensure that the target can be correctly verified."""
    example_protocol.check_target(example_session.targets[0])

def test_calc_solution(
        example_protocol: Protocol,
        example_transducer: Transducer,
        example_session: Session,
        example_transducer_transform: np.ndarray
    ):
    """Make sure a solution can be calculated."""
    import logging
    from copy import deepcopy

    from openlifu.util.units import convert_transform

    logging.disable(logging.CRITICAL)
    target = deepcopy(example_session.targets[0])
    target.rescale("m")
    transducer_transform = convert_transform(example_transducer_transform, units="mm", tgt_units="m")
    target.transform(np.linalg.inv(transducer_transform))
    target.dims = ("lat", "ele", "ax")

    example_protocol.calc_solution(
        target,
        example_transducer,
        volume=None,
        session=example_session,
        simulate=False,
        scale=False
    )
