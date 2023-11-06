from pyfus import xdc
from pyfus.util.units import getunitconversion
import kwave
import kwave.data
from kwave.kgrid import kWaveGrid
from kwave.kmedium import kWaveMedium
from kwave.options.simulation_options import SimulationOptions
from kwave.ksensor import kSensor
from kwave.ksource import kSource
from kwave.kspaceFirstOrder3D import kspaceFirstOrder3D
from kwave.options.simulation_execution_options import SimulationExecutionOptions
from kwave.utils.kwave_array import kWaveArray
from crc import Crc32, Calculator
import json
from typing import List, Dict, Any, Tuple, Optional
import logging
import xarray as xa
import numpy as np

def get_kgrid(coords: xa.Coordinates, t_end = 0, dt = 0, sound_speed_ref=1500):
    scl = getunitconversion(coords['lat'].attrs['units'], 'm')
    sz = [len(coord) for coord in coords.values()]
    dx = [np.diff(coord)[0]*scl for coord in coords.values()]
    kgrid = kWaveGrid(sz, dx)
    if dt == 0 or t_end == 0:
        kgrid.makeTime(sound_speed_ref)
    else:
        Nt = round(t_end / dt)
        kgrid.setTime(Nt, dt)
    return kgrid

def get_karray(arr: xdc.Transducer, 
               bli_tolerance: float = 0.5,
               upsampling_rate: int = 5,
               translation: List[float] = [0.,0.,0.],
               rotation: List[float] = [0.,0.,0.]):
    karray = kWaveArray(bli_tolerance=bli_tolerance, upsampling_rate=upsampling_rate)
    for el in arr.elements:
        ele_pos = list(el.get_position(units="m"))
        ele_w, ele_l = el.get_size(units="m")
        ele_angle = list(el.get_angle(units="deg"))
        karray.add_rect_element(ele_pos, ele_w, ele_l, ele_angle)
    translation = kwave.data.Vector(translation)
    rotation = kwave.data.Vector(rotation)
    karray.set_array_position(translation, rotation)
    return karray

def get_medium(params: xa.Dataset):
    medium= kWaveMedium(sound_speed=params['sound_speed'].attrs['ref_value'], 
                       density=params['density'].attrs['ref_value'])
    return medium

def get_sensor(kgrid, record=['p_max','p_min']):
    sensor_mask = np.ones([kgrid.Nx, kgrid.Ny, kgrid.Nz])
    sensor = kSensor(sensor_mask, record=record)
    return sensor

def get_source(kgrid, karray, source_sig, grid_weights=None):
    source = kSource()
    logging.info("Getting binary mask")
    source.p_mask = karray.get_array_binary_mask(kgrid)
    logging.info("Getting distributed source signal")
    source.p = karray.get_distributed_source_signal(kgrid, source_sig, grid_weights=grid_weights)
    return source

def hash_array_kgrid(kgrid, karray):
    c = Calculator(Crc32.CRC32)
    d = {'x':kgrid.x_vec.tolist(),
        'y':kgrid.y_vec.tolist(),
        'z':kgrid.z_vec.tolist(),
        'transform': karray.array_transformation.tolist(),
        'BLI_tolerance': karray.bli_tolerance,
        'upsampling_rate': karray.upsampling_rate}
    check = c.checksum(bytes(json.dumps(d), 'utf-8'))
    return f'{check:x}'    

def run_simulation(arr: xdc.Transducer, 
                   params: xa.Dataset, 
                   delays: Optional[np.ndarray] = None,
                   apod: Optional[np.ndarray] = None,
                   freq: float = 1e6,
                   cycles: float = 20,
                   amplitude: float = 1,
                   dt: float = 0,
                   t_end: float = 0,
                   load_gridweights: bool = True,
                   save_gridweights: bool = True,
                   db = None):
    delays = delays if delays is not None else np.zeros(arr.numelements())
    apod = apod if apod is not None else np.ones(arr.numelements())
    kgrid = get_kgrid(params.coords, dt=dt, t_end=t_end)
    t = np.arange(0, cycles / freq, kgrid.dt)
    input_signal = amplitude * np.sin(2 * np.pi * freq * t)
    source_mat = arr.calc_output(input_signal, kgrid.dt, delays, apod)
    pcoords = params.coords['lat'].attrs['units']
    scl = getunitconversion(pcoords, 'm')
    array_offset =[-float(coord.mean())*scl for coord in params.coords.values()]
    karray = get_karray(arr, translation=array_offset)
    medium = get_medium(params)
    sensor = get_sensor(kgrid, record=['p_max', 'p_min'])
    grid_weights = None
    if load_gridweights and db is not None:
        h = hash_array_kgrid(kgrid, karray)
        available_hashes = db.get_gridweight_hashes(arr.id)
        if h in available_hashes:
            logging.info("Loading grid weights")
            grid_weights = db.load_gridweights(arr.id, h)
    if grid_weights is None:
        logging.info("Calculating grid weights")
        grid_weights = np.array([karray.get_element_grid_weights(kgrid, i) for i in range(karray.number_elements)])
    if save_gridweights and db is not None:
        logging.info("Saving grid weights")
        db.add_gridweights(arr.id, h, grid_weights, on_conflict='overwrite')
    source = get_source(kgrid, karray, source_mat, grid_weights=grid_weights)
    logging.info("Running simulation")
    simulation_options = SimulationOptions(
                            pml_auto=True,
                            pml_inside=False,
                            save_to_disk=True,
                            data_cast='single'
                        )
    execution_options = SimulationExecutionOptions(is_gpu_simulation=True)
    output = kspaceFirstOrder3D(kgrid=kgrid, 
                                source=source, 
                                sensor=sensor, 
                                medium=medium, 
                                simulation_options=simulation_options,
                                execution_options=execution_options)
    logging.info('Simulation Complete')
    sz = list(params.coords.sizes.values())
    p_max = xa.DataArray(output['p_max'].reshape(sz, order='F'),
                         coords=params.coords,
                         name='p_max', 
                         attrs={'units':'Pa', 'long_name':'PPP'})
    p_min = xa.DataArray(-1*output['p_min'].reshape(sz, order='F'),
                         coords=params.coords,
                         name='p_min', 
                         attrs={'units':'Pa', 'long_name':'PNP'})
    Z = params['density'].data*params['sound_speed'].data
    intensity = xa.DataArray(1e-4*output['p_min'].reshape(sz, order='F')**2/(2*Z),
                         coords=params.coords,
                         name='I', 
                         attrs={'units':'W/cm^2', 'long_name':'Intensity'})
    ds = xa.Dataset({'p_max':p_max, 'p_min':p_min, 'ita':intensity})
    return ds, output
