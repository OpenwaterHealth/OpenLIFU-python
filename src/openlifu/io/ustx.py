from dataclasses import dataclass, field
from typing import Tuple, Optional, List, Dict, Literal
from openlifu.util.units import getunitconversion
import numpy as np
import logging

NUM_TRANSMITTERS = 2
ADDRESS_GLOBAL_MODE = 0x0
ADDRESS_STANDBY = 0x1
ADDRESS_DYNPWR_2 = 0x6
ADDRESS_LDO_PWR_1 = 0xB
ADDRESS_TRSW_TURNOFF = 0xC
ADDRESS_DYNPWR_1 = 0xF
ADDRESS_LDO_PWR_2 = 0x14
ADDRESS_TRSW_TURNON = 0x15
ADDRESS_DELAY_SEL = 0x16
ADDRESS_PATTERN_MODE = 0x18
ADDRESS_PATTERN_REPEAT = 0x19
ADDRESS_PATTERN_SEL_G2 = 0x1E
ADDRESS_PATTERN_SEL_G1 = 0x1F
ADDRESS_TRSW = 0x1A
ADDRESS_APODIZATION = 0x1B
ADDRESSES_GLOBAL = [ADDRESS_GLOBAL_MODE,
                    ADDRESS_STANDBY,
                    ADDRESS_DYNPWR_2,
                    ADDRESS_LDO_PWR_1,
                    ADDRESS_TRSW_TURNOFF,
                    ADDRESS_DYNPWR_1,
                    ADDRESS_LDO_PWR_2,
                    ADDRESS_TRSW_TURNON,
                    ADDRESS_DELAY_SEL, 
                    ADDRESS_PATTERN_MODE, 
                    ADDRESS_PATTERN_REPEAT,
                    ADDRESS_PATTERN_SEL_G1,
                    ADDRESS_PATTERN_SEL_G2,
                    ADDRESS_TRSW,
                    ADDRESS_APODIZATION]
ADDRESSES_DELAY_DATA = [i for i in range(0x20, 0x11F+1)]
ADDRESSES_PATTERN_DATA = [i for i in range(0x120, 0x19F+1)]
ADDRESSES = ADDRESSES_GLOBAL + ADDRESSES_DELAY_DATA + ADDRESSES_PATTERN_DATA
NUM_CHANNELS = 32
MAX_REGISTER = 0x19F
REGISTER_BYTES = 4
REGISTER_WIDTH = REGISTER_BYTES*8
DELAY_ORDER = [[32, 30],
               [28, 26],
               [24, 22],
               [20, 18],
               [31, 29],
               [27, 25],
               [23, 21],
               [19, 17],
               [16, 14],
               [12, 10],
               [8, 6],
               [4, 2],
               [15, 13],
               [11, 9],
               [7, 5],
               [3, 1]]
DELAY_CHANNEL_MAP = {}
for row, channels in enumerate(DELAY_ORDER):
    for i, channel in enumerate(channels):
        DELAY_CHANNEL_MAP[channel] = {'row': row, 'lsb': 16*(1-i)}
DELAY_PROFILE_OFFSET = 16
VALID_DELAY_PROFILES = [i for i in range(1, 17)]
DELAY_WIDTH = 13
APODIZATION_CHANNEL_ORDER = [17, 19, 21, 23, 25, 27, 29, 31, 18, 20, 22, 24, 26, 28, 30, 32, 1, 3, 5, 7, 9, 11, 13, 15, 2, 4, 6, 8, 10, 12, 14, 16]
DEFAULT_PATTERN_DUTY_CYCLE = 0.66
PATTERN_PROFILE_OFFSET = 4
NUM_PATTERN_PROFILES = 32
VALID_PATTERN_PROFILES = [i for i in range(1, NUM_PATTERN_PROFILES+1)]
MAX_PATTERN_PERIODS = 16
PATTERN_PERIOD_ORDER = [[1, 2, 3, 4],
                 [5, 6, 7, 8],
                 [9, 10, 11, 12],
                 [13, 14, 15, 16]]
PATTERN_LENGTH_WIDTH = 5
MAX_PATTERN_PERIOD_LENGTH = 30
PATTERN_LEVEL_WIDTH = 3
PATTERN_MAP = {}
for row, periods in enumerate(PATTERN_PERIOD_ORDER):
    for i, period in enumerate(periods):
        PATTERN_MAP[period] = {'row': row, 'lsb_lvl': i*(PATTERN_LEVEL_WIDTH+PATTERN_LENGTH_WIDTH), 'lsb_period': i*(PATTERN_LENGTH_WIDTH+PATTERN_LEVEL_WIDTH)+PATTERN_LEVEL_WIDTH}
MAX_REPEAT = 2**5-1
MAX_ELASTIC_REPEAT = 2**16-1
DEFAULT_TAIL_COUNT = 29
DEFAULT_CLK_FREQ = 64e6
ProfileOpts = Literal['active', 'configured', 'all']

def get_delay_location(channel:int, profile:int=1):
    """
    Gets the address and least significant bit of a delay
    
    :param channel: Channel number
    :param profile: Delay profile number
    :returns: Register address and least significant bit of the delay location
    """
    if channel not in DELAY_CHANNEL_MAP:
        raise ValueError(f"Invalid channel {channel}.")
    channel_map = DELAY_CHANNEL_MAP[channel]
    if profile not in VALID_DELAY_PROFILES:
        raise ValueError(f"Invalid Profile {profile}")
    address = ADDRESSES_DELAY_DATA[0] + (profile-1) * DELAY_PROFILE_OFFSET + channel_map['row']
    lsb = channel_map['lsb']
    return address, lsb

def set_register_value(reg_value:int, value:int, lsb:int=0, width: Optional[int]=None):
    """
    Sets the value of a parameter in a register integer

    :param reg_value: Register value
    :param value: New value of the parameter
    :param lsb: Least significant bit of the parameter
    :param width: Width of the parameter (bits)
    :returns: New register value
    """
    if width is None:
        width = REGISTER_WIDTH - lsb
    mask = (1 << width) - 1
    if value < 0 or value > mask:
        raise ValueError(f"Value {value} does not fit in {width} bits")
    return (reg_value & ~(mask << lsb)) | ((int(value) & mask) << lsb)

def get_register_value(reg_value:int, lsb:int=0, width: Optional[int]=None):
    """
    Extracts the value of a parameter from a register integer

    :param reg_value: Register value
    :param lsb: Least significant bit of the parameter
    :param width: Width of the parameter (bits)
    :returns: Value of the parameter
    """
    if width is None:
        width = REGISTER_WIDTH - lsb
    mask = (1 << width) - 1
    return (reg_value >> lsb) & mask

def calc_pulse_pattern(frequency:float, duty_cycle:float=DEFAULT_PATTERN_DUTY_CYCLE, bf_clk:float=DEFAULT_CLK_FREQ):
    """
    Calculates the pattern for a given frequency and duty cycle

    The pattern is calculated to represent a single cycle of a pulse with the specified frequency and duty cycle.
    If the pattern requires more than 16 periods, the clock divider is increased to reduce the period length.

    :param frequency: Frequency of the pattern in Hz
    :param duty_cycle: Duty cycle of the pattern
    :param bf_clk: Clock frequency of the BF system in Hz
    :returns: Tuple of lists of levels and lengths, and the clock divider setting
    """
    clk_div_n = 0
    while clk_div_n < 6:        
        clk_n = bf_clk / (2**clk_div_n)
        period_samples = int(clk_n / frequency)
        first_half_period_samples = int(period_samples / 2)
        second_half_period_samples = period_samples - first_half_period_samples
        first_on_samples = int(first_half_period_samples * duty_cycle)
        if first_on_samples < 2:
            logging.warning(f"Duty cycle too short. Setting to minimum of 2 samples")
            first_on_samples = 2
        first_off_samples = first_half_period_samples - first_on_samples
        second_on_samples = max(2, int(second_half_period_samples * duty_cycle))
        if second_on_samples < 2:
            logging.warning(f"Duty cycle too short. Setting to minimum of 2 samples")
            second_on_samples = 2
        second_off_samples = second_half_period_samples - second_on_samples
        if first_off_samples > 0 and first_off_samples < 2:
            logging.warn
            first_off_samples = 0
            first_on_samples = first_half_period_samples
        if second_off_samples > 0 and first_off_samples < 2:
            second_off_samples = 0
            second_on_samples = second_half_period_samples
        levels = [1, 0, -1, 0]
        per_lengths = []
        per_levels = []
        for i, samples in enumerate([first_on_samples, first_off_samples, second_on_samples, second_off_samples]):
            while samples > 0:
                if samples > MAX_PATTERN_PERIOD_LENGTH+2:
                    if samples == MAX_PATTERN_PERIOD_LENGTH+3:
                        per_lengths.append(MAX_PATTERN_PERIOD_LENGTH-1)
                        samples -= (MAX_PATTERN_PERIOD_LENGTH+1)
                    else:
                        per_lengths.append(MAX_PATTERN_PERIOD_LENGTH)
                        samples -= (MAX_PATTERN_PERIOD_LENGTH+2)
                    per_levels.append(levels[i])    
                else:
                    per_lengths.append(samples-2)
                    per_levels.append(levels[i])
                    samples = 0
        if len(per_levels) <= MAX_PATTERN_PERIODS:
            t = (np.arange(np.sum(np.array(per_lengths)+2))*(1/clk_n)).tolist()
            y = np.concatenate([[yi]*(ni+2) for yi,ni in zip(per_levels, per_lengths)]).tolist()
            pattern = {'levels': per_levels, 
                        'lengths': per_lengths, 
                        'clk_div_n': clk_div_n,
                        't': t,
                        'y': y}
            return pattern
        else:
            clk_div_n += 1
    raise ValueError(f"Pattern requires too many periods ({len(per_levels)} > {MAX_PATTERN_PERIODS})")

def get_pattern_location(period:int, profile:int=1):
    """
    Gets the address and least significant bit of a pattern period

    :param period: Pattern period number
    :param profile: Pattern profile number
    :returns: Register address and least significant bit of the pattern period location
    """
    if period not in PATTERN_MAP:
        raise ValueError(f"Invalid period {period}.")
    if profile not in VALID_PATTERN_PROFILES:
        raise ValueError(f"Invalid profile {profile}.")
    address = ADDRESSES_PATTERN_DATA[0] + (profile-1) * PATTERN_PROFILE_OFFSET + PATTERN_MAP[period]['row']
    lsb_lvl = PATTERN_MAP[period]['lsb_lvl']
    lsb_period = PATTERN_MAP[period]['lsb_period']
    return address, lsb_lvl, lsb_period

def print_regs(d):
    for addr, val in sorted(d.items()):
        if isinstance(val, list):
            for i, v in enumerate(val):
                print(f'0x{addr:X}[+{i:d}]:x{v:08X}')
        else:
            print(f'0x{addr:X}:x{val:08X}')

def pack_registers(regs, pack_single:bool=False):
    """
    Packs registers into contiguous blocks
    
    :param regs: Dictionary of registers
    :param pack_single: Pack single registers into arrays. Default True. 
    :returns: Dictionary of packed registers.
    """
    addresses = sorted(regs.keys())
    if len(addresses) == 0:
        return {}
    last_addr = -255
    burst_addr = -255
    packed = {}
    for addr in addresses:
        if addr == last_addr+1 and burst_addr in packed:
            packed[burst_addr].append(regs[addr])
        else:
            packed[addr] = [regs[addr]]
            burst_addr = addr
        last_addr = addr
    if not pack_single:
        for addr, val in packed.items():
            if len(val) == 1:
                packed[addr] = val[0]
    return packed

def swap_byte_order(regs):
    """
    Swaps the byte order of the registers
    
    :param regs: Dictionary of registers
    :returns: Dictionary of registers with swapped byte order
    """
    swapped = {}
    for addr, val in regs.items():
        if isinstance(val, list):
            swapped[addr] = [int.from_bytes(v.to_bytes(REGISTER_BYTES, 'big'), 'little') for v in val]
        else:
            swapped[addr] = int.from_bytes(val.to_bytes(REGISTER_BYTES, 'big'), 'little')
    return swapped

@dataclass
class DelayProfile:
    profile: int
    delays: List[float]
    apodizations: Optional[List[int]] = None
    units: str = 's'

    def __post_init__(self):
        self.num_elements = len(self.delays)
        if self.apodizations is None:
            self.apodizations = [1]*self.num_elements
        if len(self.apodizations) != self.num_elements:
            raise ValueError(f"Apodizations list must have {self.num_elements} elements")
        if self.profile not in VALID_DELAY_PROFILES:
            raise ValueError(f"Invalid Profile {self.profile}")
            
@dataclass
class PulseProfile:
    profile: int
    frequency: float
    cycles: int
    duty_cycle: float=DEFAULT_PATTERN_DUTY_CYCLE
    tail_count: int=DEFAULT_TAIL_COUNT
    invert: bool=False
    
    def __post_init__(self):
        if self.profile not in VALID_PATTERN_PROFILES:
            raise ValueError(f"Invalid profile {self.profile}.")
        
@dataclass
class Tx7332Registers:
    bf_clk: float = DEFAULT_CLK_FREQ
    _delay_profiles_list: List[DelayProfile] = field(default_factory=list)
    _pulse_profiles_list: List[PulseProfile] = field(default_factory=list)
    active_delay_profile: Optional[int] = None
    active_pulse_profile: Optional[int] = None

    def __post_init__(self):
        delay_profiles = self.configured_delay_profiles()
        if len(delay_profiles) != len(set(delay_profiles)):
            raise ValueError(f"Duplicate delay profiles found")
        if self.active_delay_profile is not None:
            if self.active_delay_profile not in delay_profiles:
                raise ValueError(f"Delay profile {self.active_delay_profile} not found")
        pulse_profiles = self.configured_pulse_profiles()
        if len(pulse_profiles) != len(set(pulse_profiles)):
            raise ValueError(f"Duplicate pulse profiles found")
        if self.active_pulse_profile is not None:
            if self.active_pulse_profile not in pulse_profiles:
                raise ValueError(f"Pulse profile {self.active_pulse_profile} not found")

    def add_delay_profile(self, delay_profile: DelayProfile, activate: Optional[bool]=None):
        if delay_profile.num_elements != NUM_CHANNELS:
            raise ValueError(f"Delay profile must have {NUM_CHANNELS} elements")
        profiles = self.configured_delay_profiles()
        if delay_profile.profile in profiles:
            i = profiles.index(delay_profile.profile)
            self._delay_profiles_list[i] = delay_profile
        else:
            self._delay_profiles_list.append(delay_profile)
        if activate is None:
            activate = self.active_delay_profile is None
        if activate:
            self.active_delay_profile = delay_profile.profile

    def add_pulse_profile(self, pulse_profile: PulseProfile, activate: Optional[bool]=None):
        profiles = self.configured_pulse_profiles()
        if pulse_profile.profile in profiles:
            i = profiles.index(pulse_profile.profile)
            self._pulse_profiles_list[i] = pulse_profile
        else:
            self._pulse_profiles_list.append(pulse_profile)
        if activate is None:
            activate = self.active_pulse_profile is None
        if activate:
            self.active_pulse_profile = pulse_profile.profile

    def remove_delay_profile(self, profile:int):
        profiles = self.configured_delay_profiles()
        if profile not in profiles:
            raise ValueError(f"Delay profile {profile} not found")
        profile = profiles.index(profile)
        del self._delay_profiles_list[profile]
        if self.active_delay_profile == profile:
            self.active_delay_profile = None

    def remove_pulse_profile(self, profile:int):
        profiles = self.configured_pulse_profiles()
        if profile not in profiles:
            raise ValueError(f"Pulse profile {profile} not found")
        profile = profiles.index(profile)
        del self._pulse_profiles_list[profile]
        if self.active_pulse_profile == profile:
            self.active_pulse_profile = None

    def get_delay_profile(self, profile: Optional[int]=None) -> DelayProfile:
        if profile is None:
            profile = self.active_delay_profile
        profiles = self.configured_delay_profiles()
        if profile not in profiles:
            raise ValueError(f"Delay profile {profile} not found")
        profile = profiles.index(profile)
        return self._delay_profiles_list[profile]

    def configured_delay_profiles(self) -> List[int]:
        return [p.profile for p in self._delay_profiles_list]

    def get_pulse_profile(self, profile: Optional[int]=None) -> PulseProfile:
        if profile is None:
            profile = self.active_pulse_profile
        profiles = self.configured_pulse_profiles()
        if profile not in profiles:
            raise ValueError(f"Pulse profile {profile} not found")
        profile = profiles.index(profile)
        return self._pulse_profiles_list[profile]
    
    def configured_pulse_profiles(self) -> List[int]:
        return [p.profile for p in self._pulse_profiles_list]
    
    def activate_delay_profile(self, profile:int):
        if profile not in self.configured_delay_profiles():
            raise ValueError(f"Delay profile {profile} not configured")
        self.active_delay_profile = profile

    def activate_pulse_profile(self, profile:int):
        if profile not in self.configured_pulse_profiles():
            raise ValueError(f"Pulse profile {profile} not configured")
        self.active_pulse_profile = profile

    def get_delay_control_registers(self, profile: Optional[int]=None) -> Dict[int,int]:
        if profile is None:
            profile = self.active_delay_profile
        delay_profile = self.get_delay_profile(profile)
        apod_register = 0
        for i, apod in enumerate(delay_profile.apodizations):
            apod_register = set_register_value(apod_register, 1-apod, lsb=i, width=1)
        delay_sel_register = 0
        delay_sel_register = set_register_value(delay_sel_register, delay_profile.profile-1, lsb=12, width=4)
        delay_sel_register = set_register_value(delay_sel_register, delay_profile.profile-1, lsb=28, width=4)
        return {ADDRESS_DELAY_SEL: delay_sel_register, 
                ADDRESS_APODIZATION: apod_register}
    
    def get_pulse_control_registers(self, profile: Optional[int]=None) -> Dict[int,int]:
        if profile is None:
            profile = self.active_pulse_profile
        pulse_profile = self.get_pulse_profile(profile)
        if pulse_profile.profile not in VALID_PATTERN_PROFILES:
            raise ValueError(f"Invalid profile {pulse_profile.profile}.")
        pattern = calc_pulse_pattern(pulse_profile.frequency, pulse_profile.duty_cycle, bf_clk=self.bf_clk)
        clk_div_n = pattern['clk_div_n']
        clk_div = 2**clk_div_n
        clk_n = self.bf_clk / clk_div
        cycles = pulse_profile.cycles
        if cycles > (MAX_REPEAT+1):
            # Use elastic repeat
            pulse_duration_samples = cycles * self.bf_clk / pulse_profile.frequency
            repeat = 0
            elastic_repeat = int(pulse_duration_samples / 16)
            period_samples = int(clk_n / pulse_profile.frequency)
            cycles = 16*elastic_repeat / period_samples
            y = pattern['y']*int(cycles+1)
            y = y[:(16*elastic_repeat)]
            y = y + ([0]*pulse_profile.tail_count)
            t = np.arange(len(y))*(1/clk_n)
            elastic_mode = 1
            if elastic_repeat > MAX_ELASTIC_REPEAT:
                raise ValueError(f"Pattern duration too long for elastic repeat")
        else:
            repeat = cycles-1
            elastic_repeat = 0        
            elastic_mode = 0
            y = pattern['y']*(repeat+1)
            y = np.array(y + [0]*pulse_profile.tail_count)
        reg_mode =  0x02000003
        reg_mode = set_register_value(reg_mode, clk_div_n, lsb=3, width=3)
        reg_mode = set_register_value(reg_mode, int(pulse_profile.invert), lsb=6, width=1)
        reg_repeat = 0
        reg_repeat = set_register_value(reg_repeat, repeat, lsb=1, width=5)
        reg_repeat = set_register_value(reg_repeat, pulse_profile.tail_count, lsb=6, width=5)
        reg_repeat = set_register_value(reg_repeat, elastic_mode, lsb=11, width=1)
        reg_repeat = set_register_value(reg_repeat, elastic_repeat, lsb=12, width=16)
        reg_pat_sel = 0
        reg_pat_sel = set_register_value(reg_pat_sel, pulse_profile.profile-1, lsb=0, width=6)        
        registers = {ADDRESS_PATTERN_MODE: reg_mode,
                     ADDRESS_PATTERN_REPEAT: reg_repeat,
                     ADDRESS_PATTERN_SEL_G1: reg_pat_sel,
                     ADDRESS_PATTERN_SEL_G2: reg_pat_sel}
        return registers

    def get_delay_data_registers(self, profile: Optional[int]=None, pack: bool=False, pack_single: bool=False) -> Dict[int,int]:
        if profile is None:
            profile = self.active_delay_profile
        delay_profile = self.get_delay_profile(profile)
        data_registers = {}
        for channel in range(1, NUM_CHANNELS+1):
            address, lsb = get_delay_location(channel, delay_profile.profile)
            if address not in data_registers:
                data_registers[address] = 0
            delay_value = int(delay_profile.delays[channel-1] * getunitconversion(delay_profile.units, 's') * self.bf_clk)
            data_registers[address] = set_register_value(data_registers[address], delay_value, lsb=lsb, width=DELAY_WIDTH)
        if pack:
            data_registers = pack_registers(data_registers, pack_single=pack_single)
        return data_registers    
    
    def get_pulse_data_registers(self, profile: Optional[int]=None, pack: bool=False, pack_single: bool=False) -> Dict[int,int]:
        if profile is None:
            profile = self.active_pulse_profile
        pulse_profile = self.get_pulse_profile(profile)
        data_registers = {}
        pattern = calc_pulse_pattern(pulse_profile.frequency, pulse_profile.duty_cycle, bf_clk=self.bf_clk)
        levels = pattern['levels']
        lengths = pattern['lengths']
        nperiods = len(levels)
        level_lut = {-1: 0b01, 0: 0b00, 1: 0b10}
        for i, (level, length) in enumerate(zip(levels, lengths)):
            address, lsb_lvl, lsb_length = get_pattern_location(i+1, pulse_profile.profile)
            if address not in data_registers:
                data_registers[address] = 0
            data_registers[address] = set_register_value(data_registers[address], level_lut[level], lsb=lsb_lvl, width=PATTERN_LEVEL_WIDTH)
            data_registers[address] = set_register_value(data_registers[address], length, lsb=lsb_length, width=PATTERN_LENGTH_WIDTH)
        if nperiods< MAX_PATTERN_PERIODS:
            address, lsb_lvl, lsb_length = get_pattern_location(nperiods+1, pulse_profile.profile)
            if address not in data_registers:
                data_registers[address] = 0
            data_registers[address] = set_register_value(data_registers[address], 0b111, lsb=lsb_lvl, width=PATTERN_LEVEL_WIDTH)
            data_registers[address] = set_register_value(data_registers[address], 0, lsb=lsb_length, width=PATTERN_LENGTH_WIDTH)
        if pack:
            data_registers = pack_registers(data_registers, pack_single=pack_single)
        return data_registers

    def get_registers(self, profiles: ProfileOpts = "configured", pack: bool=False, pack_single: bool=False) -> Dict[int,int]:
        if len(self._delay_profiles_list) == 0:
            raise ValueError(f"No delay profiles have been configured")
        if len(self._pulse_profiles_list) == 0:
            raise ValueError(f"No pulse profiles have been configured")
        if self.active_delay_profile is None:
            raise ValueError(f"No delay profile activated")
        if self.active_pulse_profile is None:
            raise ValueError(f"No pulse profile activated")
        registers = {addr:0x0 for addr in ADDRESSES_GLOBAL}        
        registers.update(self.get_delay_control_registers())
        registers.update(self.get_pulse_control_registers())
        if profiles == "active":
            delay_data = self.get_delay_data_registers(pack=pack, pack_single=pack_single)
            pulse_data = self.get_pulse_data_registers(pack=pack, pack_single=pack_single)    
        else:
            if profiles == "all":
                delay_data = {addr:0x0 for addr in ADDRESSES_DELAY_DATA}
                pulse_data = {addr:0x0 for addr in ADDRESSES_PATTERN_DATA}
            else:
                delay_data = {}
                pulse_data = {}
            for delay_profile in self._delay_profiles_list:
                delay_data.update(self.get_delay_data_registers(profile=delay_profile.profile, pack=pack, pack_single=pack_single))
            for pulse_profile in self._pulse_profiles_list:
                pulse_data.update(self.get_pulse_data_registers(profile=pulse_profile.profile, pack=pack, pack_single=pack_single))
        if pack:
            delay_data = pack_registers(delay_data, pack_single=pack_single)
            pulse_data = pack_registers(pulse_data, pack_single=pack_single)
        registers.update(delay_data)
        registers.update(pulse_data)
        return registers

@dataclass
class TxModule:
    i2c_addr: int = 0x0
    bf_clk: int = DEFAULT_CLK_FREQ
    _delay_profiles_list: List[DelayProfile] = field(default_factory=list)
    _pulse_profiles_list: List[PulseProfile] = field(default_factory=list)
    active_delay_profile: Optional[int] = None
    active_pulse_profile: Optional[int] = None
    num_transmitters: int = NUM_TRANSMITTERS

    def __post_init__(self):
        self.transmitters = tuple([Tx7332Registers(bf_clk=self.bf_clk) for _ in range(self.num_transmitters)])
        
    def add_pulse_profile(self, pulse_profile: PulseProfile, activate: Optional[bool]=None):
        """
        Add a pulse profile

        :param p: Pulse profile
        :param activate: Activate the pulse profile
        """
        profiles = self.configured_pulse_profiles()
        if pulse_profile.profile in profiles:
            i = profiles.index(pulse_profile.profile)
            self._pulse_profiles_list[i] = pulse_profile
        else:
            self._pulse_profiles_list.append(pulse_profile)
        if activate is None:
            activate = self.active_pulse_profile is None
        if activate:
            self.active_pulse_profile = pulse_profile.profile
        for tx in self.transmitters:
            tx.add_pulse_profile(pulse_profile, activate = activate)        
        
    def add_delay_profile(self, delay_profile: DelayProfile, activate: Optional[bool]=None):
        """
        Add a delay profile
        
        :param p: Delay profile
        :param activate: Activate the delay profile
        """
        if delay_profile.num_elements != NUM_CHANNELS*self.num_transmitters:
            raise ValueError(f"Delay profile must have {NUM_CHANNELS*self.num_transmitters} elements")
        profiles = self._configured_delay_profiles()
        if delay_profile.profile in profiles:
            i = profiles.index(delay_profile.profile)
            self._delay_profiles_list[i] = delay_profile
        else:
            self._delay_profiles_list.append(delay_profile)
        if activate is None:
            activate = self.active_delay_profile is None
        if activate:
            self.active_delay_profile = delay_profile.profile
        for i, tx in enumerate(self.transmitters):
            start_channel = i*NUM_CHANNELS
            profiles = np.arange(start_channel, start_channel+NUM_CHANNELS, dtype=int)
            tx_delays = np.array(delay_profile.delays)[profiles].tolist()
            tx_apodizations = np.array(delay_profile.apodizations)[profiles].tolist()
            txp = DelayProfile(delay_profile.profile, tx_delays, tx_apodizations, delay_profile.units)
            tx.add_delay_profile(txp, activate = activate)

    def remove_delay_profile(self, profile:int):
        """
        Remove a delay profile

        :param profile: Delay profile number
        """
        profiles = self._configured_delay_profiles()
        if profile not in profiles:
            raise ValueError(f"Delay profile {profile} not found")
        i = profiles.index(profile)
        del self._delay_profiles_list[i]
        if self.active_delay_profile == profile:
            self.active_delay_profile = None
        for tx in self.transmitters:
            tx.remove_delay_profile(profile)

    def remove_pulse_profile(self, profile:int):
        """
        Remove a pulse profile
        
        :param profile: Pulse profile number
        """
        profiles = self.configured_pulse_profiles()
        if profile not in profiles:
            raise ValueError(f"Pulse profile {profile} not found")
        i = profiles.index(profile)
        del self._pulse_profiles_list[i]
        if self.active_pulse_profile == profile:
            self.active_pulse_profile = None
        for tx in self.transmitters:
            tx.remove_pulse_profile(profile)

    def get_delay_profile(self, profile:Optional[int]=None) -> DelayProfile:
        """
        Retrieve a delay profile

        :param profile: Delay profile number
        :return: Delay profile
        """
        if profile is None:
            profile = self.active_delay_profile
        profiles = self._configured_delay_profiles()
        if profile not in profiles:
            raise ValueError(f"Delay profile {profile} not found")
        i = profiles.index(profile)
        return self._delay_profiles_list[i]        
    
    def _configured_delay_profiles(self) -> List[int]:
        """
        Get the configured delay profiles
        
        :return: List of delay profiles
        """
        return [p.profile for p in self._delay_profiles_list]

    def get_pulse_profile(self, profile:Optional[int]=None) -> PulseProfile:
        """
        Retrieve a pulse profile

        :param profile: Pulse profile number
        :return: Pulse profile
        """
        if profile is None:
            profile = self.active_pulse_profile
        profiles = self.configured_pulse_profiles()
        if profile not in profiles:
            raise ValueError(f"Pulse profile {profile} not found")
        i = profiles.index(profile)
        return self._pulse_profiles_list[i]
    
    def configured_pulse_profiles(self) -> List[int]:
        """
        Get the configured pulse profiles
        
        :return: List of pulse profiles
        """
        return [p.profile for p in self._pulse_profiles_list]

    def activate_delay_profile(self, profile:int=1):
        """
        Activates a delay profile
        
        :param profile: Delay profile number
        """
        for tx in self.transmitters:
            tx.activate_delay_profile(profile)  
        self.active_delay_profile = profile

    def activate_pulse_profile(self, profile:int=1):
        """
        Activates a pulse profile
        
        :param profile: Pulse profile number
        """
        for tx in self.transmitters:
            tx.activate_pulse_profile(profile)
        self.active_pulse_profile = profile

    def recompute_delay_profiles(self):
        """
        Recompute the delay profiles
        """
        for tx in self.transmitters:
            profiles = tx.configured_delay_profiles()
            for profile in profiles:
                tx.remove_delay_profile(profile)
        for dp in self._delay_profiles_list:
            self.add_delay_profile(dp, activate = dp.profile == self.active_delay_profile)

    def recompute_pulse_profiles(self):
        """
        Recompute the pulse profiles
        """
        for tx in self.transmitters:
            profiles = tx.configured_pulse_profiles()
            for profile in profiles:
                tx.remove_pulse_profile(profile)
            for pp in self._pulse_profiles_list:
                tx.add_pulse_profile(pp, activate = pp.profile == self.active_pulse_profile)

    def get_registers(self, profiles: ProfileOpts = "configured", recompute: bool = False, pack: bool=False, pack_single:bool=False) -> List[Dict[int,int]]:
        """
        Get the registers for all transmitters

        :param profiles: Profile options
        :param recompute: Recompute the registers
        :return: List of registers for each transmitter
        """
        if recompute:
            self.recompute_delay_profiles()
            self.recompute_pulse_profiles()
        return [tx.get_registers(profiles, pack=pack, pack_single=pack_single) for tx in self.transmitters]
    
    def get_delay_control_registers(self, profile:Optional[int]=None) -> List[Dict[int,int]]:
        """
        Get the delay control registers for all transmitters

        :param profile: Delay profile number
        :return: List of delay control registers for each transmitter
        """
        if profile is None:
            profile = self.active_delay_profile
        return [tx.get_delay_control_registers(profile) for tx in self.transmitters]
    
    def get_pulse_control_registers(self, profile:Optional[int]=None) -> List[Dict[int,int]]:
        """
        Get the pulse control registers for all transmitters

        :param profile: Pulse profile number
        :return: List of pulse control registers for each transmitter
        """
        if profile is None:
            profile = self.active_pulse_profile
        return [tx.get_pulse_control_registers(profile) for tx in self.transmitters]
    
    def get_delay_data_registers(self, profile:Optional[int]=None, pack: bool=False, pack_single: bool=False) -> List[Dict[int,int]]:
        """
        Get the delay data registers for all transmitters

        :param profile: Delay profile number
        :return: List of delay data registers for each transmitter
        """
        if profile is None:
            profile = self.active_delay_profile
        return [tx.get_delay_data_registers(profile, pack=pack, pack_single=pack_single) for tx in self.transmitters]
    
    def get_pulse_data_registers(self, profile:Optional[int]=None, pack: bool=False, pack_single: bool=False) -> List[Dict[int,int]]:
        """
        Get the pulse data registers for all transmitters

        :param profile: Pulse profile number
        :return: List of pulse data registers for each transmitter
        """
        if profile is None:
            profile = self.active_pulse_profile
        return [tx.get_pulse_data_registers(profile, pack=pack, pack_single=pack_single) for tx in self.transmitters]
    
@dataclass
class TxArray:
    i2c_addresses: Tuple[int] = (0x0,)
    bf_clk: int = DEFAULT_CLK_FREQ
    modules: Dict = field(default_factory=dict)
    _delay_profiles_list: List[DelayProfile] = field(default_factory=list)
    _pulse_profiles_list: List[PulseProfile] = field(default_factory=list)
    active_delay_profile: Optional[int] = None
    active_pulse_profile: Optional[int] = None
    num_transmitters: int = NUM_TRANSMITTERS

    def __post_init__(self):
        if len(set(self.i2c_addresses)) != len(self.i2c_addresses):
            raise ValueError(f"Duplicate I2C addresses found")
        self.modules = {addr:TxModule(i2c_addr=addr, bf_clk=self.bf_clk, num_transmitters=self.num_transmitters) for addr in self.i2c_addresses}
        self.num_modules = len(self.modules)

    def add_pulse_profile(self, pulse_profile: PulseProfile, activate: Optional[bool]=None):
        """
        Add a pulse profile

        :param p: Pulse profile
        :param activate: Activate the pulse profile
        """
        profiles = self.configured_pulse_profiles()
        if pulse_profile.profile in profiles:
            i = profiles.index(pulse_profile.profile)
            self._pulse_profiles_list[i] = pulse_profile
        else:
            self._pulse_profiles_list.append(pulse_profile)
        if activate is None:
            activate = self.active_pulse_profile is None
        if activate:
            self.active_pulse_profile = pulse_profile.profile
        for module in self.modules.values():
            module.add_pulse_profile(pulse_profile, activate)
        
    def add_delay_profile(self, delay_profile: DelayProfile, activate: Optional[bool]=None):
        """
        Add a delay profile
        
        :param p: Delay profile
        :param activate: Activate the delay profile
        """
        if delay_profile.num_elements != NUM_CHANNELS*self.num_transmitters*self.num_modules:
            raise ValueError(f"Delay profile must have {NUM_CHANNELS*self.num_transmitters*self.num_modules} elements")
        profiles = self.configured_delay_profiles()
        if delay_profile.profile in profiles:
            i = profiles.index(delay_profile.profile)
            self._delay_profiles_list[i] = delay_profile
        else:
            self._delay_profiles_list.append(delay_profile)
        if activate is None:
            activate = self.active_delay_profile is None
        if activate:
            self.active_delay_profile = delay_profile.profile
        for i, module in enumerate(self.modules.values()):
            start_channel = i*NUM_CHANNELS*module.num_transmitters
            profiles = np.arange(start_channel, start_channel+NUM_CHANNELS*module.num_transmitters, dtype=int)
            module_delays = np.array(delay_profile.delays)[profiles].tolist()
            module_apodizations = np.array(delay_profile.apodizations)[profiles].tolist()
            modulep = DelayProfile(delay_profile.profile, module_delays, module_apodizations, delay_profile.units)
            module.add_delay_profile(modulep, activate = activate)

    def remove_pulse_profile(self, profile:int):
        """
        Remove a pulse profile
        
        :param profile: Pulse profile number
        """
        profiles = self.configured_pulse_profiles()
        if profile not in profiles:
            raise ValueError(f"Pulse profile {profile} not found")
        i = profiles.index(profile)
        del self._pulse_profiles_list[i]
        if self.active_pulse_profile == profile:
            self.active_pulse_profile = None
        for module in self.modules.values():
            module.remove_pulse_profile(profile)

    def remove_delay_profile(self, profile:int):
        """
        Remove a delay profile

        :param profile: Delay profile number
        """
        profiles = self.configured_delay_profiles()
        if profile not in profiles:
            raise ValueError(f"Delay profile {profile} not found")
        i = profiles.index(profile)
        del self._delay_profiles_list[i]
        if self.active_delay_profile == profile:
            self.active_delay_profile = None
        for module in self.modules.values():
            module.remove_delay_profile(profile)
        
    def get_pulse_profile(self, profile:Optional[int]=None) -> PulseProfile:
        """
        Retrieve a pulse profile

        :param profile: Pulse profile number
        :return: Pulse profile
        """
        if profile is None:
            profile = self.active_pulse_profile
        profiles = self.configured_pulse_profiles()
        if profile not in profiles:
            raise ValueError(f"Pulse profile {profile} not found")
        i = profiles.index(profile)
        return self._pulse_profiles_list[i]
    
    def configured_pulse_profiles(self) -> List[int]:
        """
        Get the configured pulse profiles
        
        :return: List of pulse profiles
        """
        return [p.profile for p in self._pulse_profiles_list]

    def get_delay_profile(self, profile:Optional[int]=None) -> DelayProfile:
        """
        Retrieve a delay profile

        :param profile: Delay profile number
        :return: Delay profile
        """
        if profile is None:
            profile = self.active_delay_profile
        profiles = self.configured_delay_profiles()
        if profile not in profiles:
            raise ValueError(f"Delay profile {profile} not found")
        i = profiles.index(profile)
        return self._delay_profiles_list[i]
    
    def configured_delay_profiles(self) -> List[int]:
        """
        Get the configured delay profiles
        
        :return: List of delay profiles
        """
        return [p.profile for p in self._delay_profiles_list]

    def activate_pulse_profile(self, profile:int=1):
        """
        Activates a pulse profile
        
        :param profile: Pulse profile number
        """
        for module in self.modules.values():
            module.activate_pulse_profile(profile)
        self.active_pulse_profile = profile
    
    def activate_delay_profile(self, profile:int=1):
        """
        Activates a delay profile
        
        :param profile: Delay profile number
        """
        for module in self.modules.values():
            module.activate_delay_profile(profile)  
        self.active_delay_profile = profile

    def recompute_pulse_profiles(self):
        """
        Recompute the pulse profiles
        """
        for module in self.modules.values():
            profiles = module.configured_pulse_profiles()
            for profile in profiles:
                module.remove_pulse_profile(profile)
            for pulse_profile in self._pulse_profiles_list:
                module.add_pulse_profile(pulse_profile, activate = pulse_profile.profile == self.active_pulse_profile)

    def recompute_delay_profiles(self):
        """
        Recompute the delay profiles
        """
        for module in self.modules.values():
            profiles = module.configured_delay_profiles()
            for profile in profiles:
                module.remove_delay_profile(profile)
        for delay_profile in self._delay_profiles_list:
            self.add_delay_profile(delay_profile, activate = delay_profile.profile == self.active_delay_profile)

    def get_registers(self, profiles: ProfileOpts = "configured", recompute: bool = False, pack: bool=False, pack_single: bool=False) -> Dict[int, List[Dict[int,int]]]:
        """
        Get the registers for all modules

        :param profiles: Profile options
        :param recompute: Recompute the registers
        :return: Dictionary of registers for each module
        """
        if recompute:
            self.recompute_delay_profiles()
            self.recompute_pulse_profiles()
        return {addr:module.get_registers(profiles, pack=pack, pack_single=pack_single) for addr, module in self.modules.items()}
    
    def get_delay_control_registers(self, profile:Optional[int]=None) -> Dict[int, List[Dict[int,int]]]:
        """
        Get the delay control registers for all modules

        :param profile: Delay profile number
        :return: Dictionary of delay control registers for each module
        """
        if profile is None:
            profile = self.active_delay_profile
        return {addr:module.get_delay_control_registers(profile) for addr, module in self.modules.items()}
    
    def get_pulse_control_registers(self, profile:Optional[int]=None) -> Dict[int, List[Dict[int,int]]]:
        """
        Get the pulse control registers for all modules

        :param profile: Pulse profile number
        :return: Dictionary of pulse control registers for each module
        """
        if profile is None:
            profile = self.active_pulse_profile
        return {addr:module.get_pulse_control_registers(profile) for addr, module in self.modules.items()}
    
    def get_delay_data_registers(self, profile:Optional[int]=None, pack: bool=False, pack_single: bool=False) -> Dict[int, List[Dict[int,int]]]:
        """
        Get the delay data registers for all modules

        :param profile: Delay profile number
        :return: Dictionary of delay data registers for each module
        """
        if profile is None:
            profile = self.active_delay_profile
        return {addr:module.get_delay_data_registers(profile, pack=pack, pack_single=pack_single) for addr, module in self.modules.items()}
    
    def get_pulse_data_registers(self, profile:Optional[int]=None, pack: bool=False, pack_single: bool=False) -> Dict[int, List[Dict[int,int]]]:
        """
        Get the pulse data registers for all modules

        :param profile: Pulse profile number
        :return: Dictionary of pulse data registers for each module
        """
        if profile is None:
            profile = self.active_pulse_profile
        return {addr:module.get_pulse_data_registers(profile, pack=pack, pack_single=pack_single) for addr, module in self.modules.items()}
    
