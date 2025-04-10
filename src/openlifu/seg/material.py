from __future__ import annotations

from dataclasses import dataclass
from typing import Annotated

from openlifu.util.annotations import OpenLIFUFieldData

PARAM_INFO = {"sound_speed":{"id":"sound_speed",
                               "name": "Speed of Sound",
                               "units": "m/s"},
                "density":{"id":"density",
                           "name": "Density",
                           "units": "kg/m^3"},
                "attenuation":{"id":"attenuation",
                               "name": "Attenuation",
                               "units": "dB/cm/MHz"},
                "specific_heat":{"id":"specific_heat",
                                 "name": "Specific Heat",
                                 "units": "J/kg/K"},
                "thermal_conductivity":{"id":"thermal_conductivity",
                                        "name": "Thermal Conductivity",
                                        "units": "W/m/K"}}
@dataclass
class Material:
    id: Annotated[str, OpenLIFUFieldData("Material ID", "The unique identifier of the material")] = "material"
    """The unique identifier of the material"""

    name: Annotated[str, OpenLIFUFieldData("Material name", "Name for the material")] = "Material"
    """Name for the material"""

    sound_speed: Annotated[float, OpenLIFUFieldData("Sound speed (m/s)", "Speed of sound in the material (m/s)")] = 1500.0  # m/s
    """Speed of sound in the material (m/s)"""

    density: Annotated[float, OpenLIFUFieldData("Density (kg/m^3)", "Mass density of the material (kg/m^3)")] = 1000.0  # kg/m^3
    """Mass density of the material (kg/m^3)"""

    attenuation: Annotated[float, OpenLIFUFieldData("Attenuation (dB/cm/MHz)", "Ultrasound attenuation in the material (dB/cm/MHz)")] = 0.0  # dB/cm/MHz
    """Ultrasound attenuation in the material (dB/cm/MHz)"""

    specific_heat: Annotated[float, OpenLIFUFieldData("Specific heat (J/kg/K)", "Specific heat capacity of the material (J/kg/K)")] = 4182.0  # J/kg/K
    """Specific heat capacity of the material (J/kg/K)"""

    thermal_conductivity: Annotated[float, OpenLIFUFieldData("Thermal conductivity (W/m/K)", "Thermal conductivity of the material (W/m/K)")] = 0.598  # W/m/K
    """Thermal conductivity of the material (W/m/K)"""

    @classmethod
    def param_info(cls, param_id: str):
        if param_id not in PARAM_INFO:
            raise ValueError(f"Parameter {param_id} not found.")
        return PARAM_INFO[param_id]

    def get_param(self, param_id: str):
        if param_id not in PARAM_INFO:
            raise ValueError(f"Parameter {param_id} not found.")
        return self.__getattribute__(param_id)

    @staticmethod
    def get_materials(material_id="all", as_dict=True):
        material_id = ("water", "tissue", "skull", "air", "standoff") if material_id == "all" else material_id
        if isinstance(material_id, (list, tuple)):
            materials = {m: Material.get_materials(m, as_dict=False) for m in material_id}
        elif material_id in MATERIALS:
            materials = MATERIALS[material_id]
        else:
            raise ValueError(f"Material {material_id} not found.")
        if as_dict:
            return {materials.id: materials}
        else:
            return materials

    @staticmethod
    def from_dict(d):
        if isinstance(d, (list, tuple)):
            return {dd['id']: Material.from_dict(dd) for dd in d}
        elif isinstance(d, str):
            return Material.get_materials(d, as_dict=False)
        elif isinstance(d, Material):
            return d
        else:
            return Material(**d)


WATER = Material(id="water",
                 name="water",
                 sound_speed=1500.0,
                 density=1000.0,
                 attenuation=0.0,
                 specific_heat=4182.0,
                 thermal_conductivity=0.598)

TISSUE = Material(id="tissue",
                  name="tissue",
                  sound_speed=1540.0,
                  density=1000.0,
                  attenuation=0.0,
                  specific_heat=3600.0,
                  thermal_conductivity=0.5)

SKULL = Material(id="skull",
                 name="skull",
                 sound_speed=4080.0,
                 density=1900.0,
                 attenuation=0.0,
                 specific_heat=1100.0,
                 thermal_conductivity=0.3)

AIR = Material(id="air",
               name="air",
               sound_speed=344.0,
               density=1.25,
               attenuation=0.0,
               specific_heat=1012.0,
               thermal_conductivity=0.025)

STANDOFF = Material(id="standoff",
                    name="standoff",
                    sound_speed=1420.0,
                    density=1000.0,
                    attenuation=1.0,
                    specific_heat=4182.0,
                    thermal_conductivity=0.598)

MATERIALS = {"water": WATER,
             "tissue": TISSUE,
             "skull": SKULL,
             "air": AIR,
             "standoff": STANDOFF}
