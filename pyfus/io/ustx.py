from dataclasses import dataclass, field
from typing import Tuple, Optional, List, Dict
from pyfus.util.units import getunitconversion
import pyfus.io.tx7332 as tx7332
from pyfus.io.tx7332 import PulseProfile
import numpy as np
import logging

NUM_TRANSMITTERS = 2
NUM_CHANNELS = tx7332.NUM_CHANNELS
@dataclass
class DelayProfile:
    profile: int
    delays: List[float]
    apodizations: List[int] = field(default_factory=lambda: [1]*NUM_CHANNELS*NUM_TRANSMITTERS)
    units: str = 's'

    def __post_init__(self):
        if self.profile not in tx7332.DELAY_PROFILES:
            raise ValueError(f"Invalid Profile {self.profile}")
        if len(self.delays) != NUM_CHANNELS*NUM_TRANSMITTERS:
            raise ValueError(f"Delays list must have {NUM_CHANNELS*NUM_TRANSMITTERS} elements")
        if len(self.apodizations) != NUM_CHANNELS*NUM_TRANSMITTERS:
            raise ValueError(f"Apodizations list must have {NUM_CHANNELS*NUM_TRANSMITTERS} elements")

ProfileOpts = tx7332.ProfileOpts

@dataclass
class TxModule:
    address: int = 0x0
    bf_clk: int = tx7332.DEFAULT_CLK_FREQ
    delay_profiles: List[DelayProfile] = field(default_factory=list)
    pulse_profiles: List[PulseProfile] = field(default_factory=list)
    active_delay_profile: Optional[int] = None
    active_pulse_profile: Optional[int] = None

    def __post_init__(self):
        self.transmitters = tuple([tx7332.TX7332Registers(bf_clk=self.bf_clk) for _ in range(NUM_TRANSMITTERS)])

    def add_pulse_profile(self, p: PulseProfile, activate: Optional[bool]=None):
        """
        Add a pulse profile

        :param p: Pulse profile
        :param activate: Activate the pulse profile
        """
        profiles = [p.index for p in self.pulse_profiles]
        if p.index in profiles:
            i = profiles.index(p.index)
            self.pulse_profiles[i] = p
        else:
            self.pulse_profiles.append(p)
        if activate is None:
            activate = self.active_pulse_profile is None
        if activate:
            self.active_pulse_profile = p.index
        for tx in self.transmitters:
            tx.add_pulse_profile(p, activate = activate)        
        
    def add_delay_profile(self, p: DelayProfile, activate: Optional[bool]=None):
        """
        Add a delay profile
        
        :param p: Delay profile
        :param activate: Activate the delay profile
        """
        profiles = [p.index for p in self.delay_profiles]
        if p.index in profiles:
            i = profiles.index(p.index)
            self.delay_profiles[i] = p
        else:
            self.delay_profiles.append(p)
        if activate is None:
            activate = self.active_delay_profile is None
        if activate:
            self.active_delay_profile = p.index
        for i, tx in enumerate(self.transmitters):
            start_channel = i*NUM_CHANNELS
            indices = np.arange(start_channel, start_channel+NUM_CHANNELS, dtype=int)
            tx_delays = np.array(p.delays)[indices].tolist()
            tx_apodizations = np.array(p.apodizations)[indices].tolist()
            txp = tx7332.DelayProfile(p.profile, tx_delays, tx_apodizations, p.units)
            tx.add_delay_profile(txp, activate = activate)

    def remove_delay_profile(self, index:int):
        """
        Remove a delay profile

        :param index: Delay profile number
        """
        profiles = [p.index for p in self.delay_profiles]
        if index not in profiles:
            raise ValueError(f"Delay profile {index} not found")
        i = profiles.index(index)
        del self.delay_profiles[i]
        if self.active_delay_profile == index:
            self.active_delay_profile = None
        for tx in self.transmitters:
            tx.remove_delay_profile(index)

    def remove_pulse_profile(self, index:int):
        """
        Remove a pulse profile
        
        :param index: Pulse profile number
        """
        profiles = [p.index for p in self.pulse_profiles]
        if index not in profiles:
            raise ValueError(f"Pulse profile {index} not found")
        i = profiles.index(index)
        del self.pulse_profiles[i]
        if self.active_pulse_profile == index:
            self.active_pulse_profile = None
        for tx in self.transmitters:
            tx.remove_pulse_profile(index)

    def get_delay_profile(self, index:Optional[int]=None) -> DelayProfile:
        """
        Retrieve a delay profile

        :param index: Delay profile number
        :return: Delay profile
        """
        if index is None:
            index = self.active_delay_profile
        profiles = [p.index for p in self.delay_profiles]
        if index not in profiles:
            raise ValueError(f"Delay profile {index} not found")
        i = profiles.index(index)
        return self.delay_profiles[i]        
    
    def get_pulse_profile(self, index:Optional[int]=None) -> PulseProfile:
        """
        Retrieve a pulse profile

        :param index: Pulse profile number
        :return: Pulse profile
        """
        if index is None:
            index = self.active_pulse_profile
        profiles = [p.index for p in self.pulse_profiles]
        if index not in profiles:
            raise ValueError(f"Pulse profile {index} not found")
        i = profiles.index(index)
        return self.pulse_profiles[i]
    
    def activate_delay_profile(self, index:int=1):
        """
        Activates a delay profile
        
        :param profile: Delay profile number
        """
        for tx in self.transmitters:
            tx.activate_delay_profile(index)  
        self.active_delay_profile = index

    def activate_pulse_profile(self, index:int=1):
        """
        Activates a pulse profile
        
        :param profile: Pulse profile number
        """
        for tx in self.transmitters:
            tx.activate_pulse_profile(index)
        self.active_pulse_profile = index

    def recompute_delay_profiles(self):
        """
        Recompute the delay profiles
        """
        for tx in self.transmitters:
            indices = [p.index for p in tx.delay_profiles]
            for index in indices:
                tx.remove_delay_profile(index)
            for dp in self.delay_profiles:
                tx.add_delay_profile(dp, activate = dp.index == self.active_delay_profile)

    def recompute_pulse_profiles(self):
        """
        Recompute the pulse profiles
        """
        for tx in self.transmitters:
            indices = [p.index for p in tx.pulse_profiles]
            for index in indices:
                tx.remove_pulse_profile(index)
            for pp in self.pulse_profiles:
                tx.add_pulse_profile(pp, activate = pp.index == self.active_pulse_profile)

    def get_registers(self, profiles: ProfileOpts = "set", recompute: bool = False) -> List[Dict[int,int]]:
        """
        Get the registers for all transmitters

        :param profiles: Profile options
        :param recompute: Recompute the registers
        :return: List of registers for each transmitter
        """
        if recompute:
            self.recompute_delay_profiles()
            self.recompute_pulse_profiles()
        return [tx.get_registers(profiles) for tx in self.transmitters]
    
    def get_delay_control_registers(self, index:Optional[int]=None) -> List[Dict[int,int]]:
        """
        Get the delay control registers for all transmitters

        :param index: Delay profile number
        :return: List of delay control registers for each transmitter
        """
        if index is None:
            index = self.active_delay_profile
        return [tx.get_delay_control_registers(index) for tx in self.transmitters]
    
    def get_pulse_control_registers(self, index:Optional[int]=None) -> List[Dict[int,int]]:
        """
        Get the pulse control registers for all transmitters

        :param index: Pulse profile number
        :return: List of pulse control registers for each transmitter
        """
        if index is None:
            index = self.active_pulse_profile
        return [tx.get_pulse_control_registers(index) for tx in self.transmitters]
    
    def get_delay_data_registers(self, index:Optional[int]=None) -> List[Dict[int,int]]:
        """
        Get the delay data registers for all transmitters

        :param index: Delay profile number
        :return: List of delay data registers for each transmitter
        """
        if index is None:
            index = self.active_delay_profile
        return [tx.get_delay_data_registers(index) for tx in self.transmitters]
    
    def get_pulse_data_registers(self, index:Optional[int]=None) -> List[Dict[int,int]]:
        """
        Get the pulse data registers for all transmitters

        :param index: Pulse profile number
        :return: List of pulse data registers for each transmitter
        """
        if index is None:
            index = self.active_pulse_profile
        return [tx.get_pulse_data_registers(index) for tx in self.transmitters]