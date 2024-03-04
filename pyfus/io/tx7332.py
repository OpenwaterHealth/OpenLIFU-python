from dataclasses import dataclass, field
from typing import Tuple, Optional, List, Dict
from pyfus.util.units import getunitconversion
import numpy as np
import logging

ADDRESS_PATTERN_MODE = 0x18
ADDRESS_PATTERN_REPEAT = 0x19
ADDRESS_DELAY_SEL = 0x16
ADDRESSES_GLOBAL = [0x0, 0x1, 0x6, 0xB, 0xC, 0xF, 0x14, 0x15, ADDRESS_DELAY_SEL, ADDRESS_PATTERN_MODE, ADDRESS_PATTERN_REPEAT]
ADDRESSES_PATTERN_SEL = [0x1E, 0x1F]
ADDRESS_TRSW = 0x1A
ADDRESS_APODIZATION = 0x1B
ADDRESSES_APODIZATION = [ADDRESS_TRSW, ADDRESS_APODIZATION]
ADDRESSES_DELAY_DATA = [i for i in range(0x20, 0x11F+1)]
ADDRESSES_PATTERN_DATA = [i for i in range(0x120, 0x19F+1)]
ADDRESSES = ADDRESSES_GLOBAL + ADDRESSES_PATTERN_SEL + ADDRESSES_APODIZATION + ADDRESSES_DELAY_DATA + ADDRESSES_PATTERN_DATA

NUM_CHANNELS = 32
MAX_REGISTER = 0x19F
REGISTER_WIDTH = 32
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
DELAY_PROFILES = [i for i in range(1, 17)]
DELAY_WIDTH = 13
APODIZATION_CHANNEL_ORDER = [17, 19, 21, 23, 25, 27, 29, 31, 18, 20, 22, 24, 26, 28, 30, 32, 1, 3, 5, 7, 9, 11, 13, 15, 2, 4, 6, 8, 10, 12, 14, 16]
DEFAULT_PATTERN_DUTY_CYCLE = 0.66
PATTERN_PROFILE_OFFSET = 4
NUM_PATTERN_PROFILES = 32
PATTERN_PROFILES = [i for i in range(1, NUM_PATTERN_PROFILES+1)]
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

class TX7332:
    def __init__(self, bf_clk:float=64e6):
        """
        Initializes the TX7332 object with the specified clock frequency.

        :param bf_clk: Clock frequency of the BF system in Hz
        """
        self.registers = {addr:0 for addr in ADDRESSES}
        self.bf_clk = bf_clk
        self.pulse_profiles = {}
        self.pulse_profile = None
        self.delay_profiles = {}
        self.delay_profile = None

    @staticmethod
    def _get_register_value(reg_value:int, lsb:int=0, width: Optional[int]=None):
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
    
    def get_register(self, address:int, lsb:int=0, width: Optional[int]=None):
        """
        Extracts the value of a parameter from a register

        :param address: Address of the register
        :param lsb: Least significant bit of the parameter
        :param width: Width of the parameter (bits)
        :returns: Value of the parameter
        """
        if address not in ADDRESSES:
            raise ValueError(f"Invalid address {address}.")
        return self._get_register_value(self.registers[address], lsb, width)
    
    def get_bit(self, address:int, bit:int):
        """
        Extracts the value of a bit from a register

        :param address: Address of the register
        :param bit: Bit number
        :returns: Value of the bit
        """
        return self.get_register(address, lsb=bit, width=1)
    
    @staticmethod
    def _set_register_value(reg_value:int, value:int, lsb:int=0, width: Optional[int]=None):
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
        return (reg_value & ~(mask << lsb)) | ((value & mask) << lsb)

    def set_register(self, address, value:int, lsb:int=0, width: Optional[int]=None):
        """
        Sets the value of a parameter in a register

        :param address: Address of the register
        :param value: New value of the parameter
        :param lsb: Least significant bit of the parameter
        :param width: Width of the parameter (bits)
        """
        if address not in ADDRESSES:
            raise ValueError(f"Invalid address {address}.")
        self.registers[address] = self._set_register_value(self.registers[address], value, lsb, width)

    def set_bit(self, address, bit:int, value:int):
        """
        Sets the value of a bit in a register

        :param address: Address of the register
        :param bit: Bit number
        :param value: New value of the bit
        """
        if value not in [0, 1]:
            raise ValueError(f"Value must be 0 or 1")
        self.set_register(address, value, lsb=bit, width=1)

    @staticmethod
    def _get_delay_location(channel:int, profile:int=1):
        """
        Gets the address and least significant bit of a delay
        
        :param channel: Channel number
        :param profile: Delay profile number
        :returns: Register address and least significant bit of the delay location
        """
        if channel not in DELAY_CHANNEL_MAP:
            raise ValueError(f"Invalid channel {channel}.")
        channel_map = DELAY_CHANNEL_MAP[channel]
        if profile not in DELAY_PROFILES:
            raise ValueError(f"Invalid Profile {profile}")
        address = ADDRESSES_DELAY_DATA[0] + (profile-1) * DELAY_PROFILE_OFFSET + channel_map['row']
        lsb = channel_map['lsb']
        return address, lsb

    def _get_delay_value(self, channel:int, profile:int=1):
        """
        Gets the value of a delay from its register

        :param channel: Channel number
        :param profile: Delay profile number
        :returns: Value of the delay in the register
        """
        address, lsb = self._get_delay_location(channel, profile)
        return self.get_register(address, lsb=lsb, width=DELAY_WIDTH)    
    
    def _set_delay_value(self, channel:int, value:int, profile:int=1):
        """
        Sets the value for a delay in a register

        :param channel: Channel number
        :param value: New value of the delay
        :param profile: Delay profile number
        """
        address, lsb = self._get_delay_location(channel, profile)
        self.set_register(address, value, lsb=lsb, width=DELAY_WIDTH)
    
    def get_delay_registers(self, profile:int=1):
        """
        Gets the values of all delay registers for a profile
        
        :param profile: Delay profile number
        :returns: Dictionary of delay register addresses and values
        """
        if profile not in DELAY_PROFILES:
            raise ValueError(f"Invalid Profile {profile}")
        return {address: self.registers[address] for address in range(ADDRESSES_DELAY_DATA[0] + (profile-1) * DELAY_PROFILE_OFFSET, ADDRESSES_DELAY_DATA[0] + profile * DELAY_PROFILE_OFFSET)}

    def get_delay(self, channel, profile:int=1, units='s'):
        """
        Gets the value of a delay from the registers

        :param channel: Channel number
        :param profile: Delay profile number
        :param units: Units of the delay value. Default is 's'
        :returns: Value of the delay (units)
        """
        return self._get_delay_value(channel, profile) / self.bf_clk * getunitconversion('s', units)

    def set_delay(self, channel, delay, profile:int=1, units='s'):
        """
        Sets the value of a delay in the registers
        
        :param channel: Channel number
        :param delay: New value of the delay
        :param profile: Delay profile number
        :param units: Units of the delay value. Default is 's'
        """
        self._set_delay_value(channel, int(delay * getunitconversion(units, 's') * self.bf_clk), profile)

    def set_delay_profile(self, delays:List[float], apodizations:Optional[List[int]]=None,  profile:int=1, units='s'):
        """
        Sets the values of all delays and apodizations for a profile

        :param delays: List of delay values
        :param apodizations: List of apodization values
        :param profile: Delay profile number
        :param units: Units of the delay values. Default is 's'
        """
        if profile not in DELAY_PROFILES:
            raise ValueError(f"Invalid Profile {profile}")
        if len(delays) != NUM_CHANNELS:
            raise ValueError(f"Delays list must have {NUM_CHANNELS} elements")
        if apodizations is None:
            apodizations = [1]*NUM_CHANNELS
        if len(apodizations) != NUM_CHANNELS:
            raise ValueError(f"Apodizations list must have {NUM_CHANNELS} elements")
        for i, delay in enumerate(delays):
            self.set_delay(i+1, delay, profile, units)
        apod_register = 0
        for i, apod in enumerate(apodizations):
            apod_register = self._set_register_value(apod_register, 1-apod, lsb=i, width=1)
        delay_sel_register = 0
        delay_sel_register = self._set_register_value(delay_sel_register, profile-1, lsb=12, width=4)
        delay_sel_register = self._set_register_value(delay_sel_register, profile-1, lsb=28, width=4)
        registers = {ADDRESS_DELAY_SEL: delay_sel_register, ADDRESS_APODIZATION: apod_register}
        delay_registers = self.get_delay_registers(profile)
        actual_delays = self.get_delays(profile, units)
        self.delay_profiles[profile] = {
            'delays': actual_delays, 
            'apodizations': apodizations,
            'units': units,
            'registers': registers,
            'delay_registers': delay_registers}
        if profile == self.delay_profile or self.delay_profile is None:
            self.activate_delay_profile(profile)

    def get_delays(self, profile:int=1, units='s'):
        """
        Gets the values of all delays for a profile
        
        :param profile: Delay profile number
        :param units: Units of the delay values. Default is 's'
        :returns: List of delay values (units)
        """
        return [self.get_delay(channel, profile, units) for channel in range(1, NUM_CHANNELS+1)]

    def get_delay_profile(self, profile:int=1):
        """
        Retrieve a delay profile

        :param profile: Delay profile number
        :returns: Dictionary of delay profile information
        """
        if profile not in DELAY_PROFILES:
            raise ValueError(f"Invalid Profile {profile}")
        return self.delay_profiles[profile]
    
    def activate_delay_profile(self, profile:int):
        """
        Activates a delay profile

        The delay profile is activated by setting the delay select and apodization registers to the values stored in the profile.
        :param profile: Delay profile number
        """
        if profile not in DELAY_PROFILES:
            raise ValueError(f"Invalid Profile {profile}")
        self.delay_profile = profile
        for addr, value in self.delay_profiles[profile]['registers'].items():
            self.set_register(addr, value)

    def get_apodization(self, channel:int):
        """
        Gets the value of an apodization from the register

        :param channel: Channel number
        :returns: Value of the apodization
        """
        if channel not in APODIZATION_CHANNEL_ORDER:
            raise ValueError(f"Invalid channel {channel}")
        return 1-self.get_bit(ADDRESS_APODIZATION, bit=APODIZATION_CHANNEL_ORDER.index(channel))
    
    def get_apodizations_register(self):
        """
        Gets the value of the apodization register
        """
        return self.get_register(ADDRESS_APODIZATION)

    def set_apodization(self, channel, apod: int):
        """
        Sets the value of an apodization in the register

        :param channel: Channel number
        :param apod: New value of the apodization
        """
        if apod not in [0, 1]:
            raise ValueError(f"Apodization value must be 0 or 1")
        if channel not in APODIZATION_CHANNEL_ORDER:
            raise ValueError(f"Invalid channel {channel}")
        self.set_bit(ADDRESS_APODIZATION, bit=APODIZATION_CHANNEL_ORDER.index(channel), value=1-apod)

    def set_apodizations(self, apodizations: List[int]):
        """
        Sets the values of all apodizations in the register

        :param apodizations: List of apodization values
        """
        if len(apodizations) != NUM_CHANNELS:
            raise ValueError(f"Apodizations list must have {NUM_CHANNELS} elements")
        for i, apod in enumerate(apodizations):
            self.set_apodization(i+1, apod)

    def get_apodizations(self):
        """
        Gets the values of all apodizations from the register

        :returns: List of apodization values
        """
        return [self.get_apodization(channel) for channel in range(1, NUM_CHANNELS+1)]
    
    @staticmethod
    def _get_pattern_location(period:int, profile:int=1):
        """
        Gets the address and least significant bit of a pattern period

        :param period: Pattern period number
        :param profile: Pattern profile number
        :returns: Register address and least significant bit of the pattern period location
        """
        if period not in PATTERN_MAP:
            raise ValueError(f"Invalid period {period}.")
        if profile not in PATTERN_PROFILES:
            raise ValueError(f"Invalid profile {profile}.")
        address = ADDRESSES_PATTERN_DATA[0] + (profile-1) * PATTERN_PROFILE_OFFSET + PATTERN_MAP[period]['row']
        lsb_lvl = PATTERN_MAP[period]['lsb_lvl']
        lsb_period = PATTERN_MAP[period]['lsb_period']
        return address, lsb_lvl, lsb_period
    
    def _get_pattern_value(self, period:int, profile:int=1):
        """
        Gets the value of a pattern period (level, length) from its register

        :param period: Pattern period number
        :param profile: Pattern profile number
        :returns: Value of the pattern period in the register
        """
        address, lsb_lvl, lsb_length = self._get_pattern_location(period, profile)
        level = self.get_register(address, lsb=lsb_lvl, width=PATTERN_LEVEL_WIDTH)
        length = self.get_register(address, lsb=lsb_length, width=PATTERN_LENGTH_WIDTH)
        return level, length
    
    def _set_pattern_value(self, period:int, level:int, length:int, profile:int=1):
        """
        Sets the value for a pattern period in a register

        :param period: Pattern period number
        :param level: Level of the pattern period
        :param length: Length of the pattern period
        :param profile: Pattern profile number
        """
        address, lsb_lvl, lsb_length = self._get_pattern_location(period, profile)
        self.set_register(address, level, lsb=lsb_lvl, width=PATTERN_LEVEL_WIDTH)
        self.set_register(address, length, lsb=lsb_length, width=PATTERN_LENGTH_WIDTH)

    def calc_pulse_pattern(self, frequency:float, duty_cycle:float=DEFAULT_PATTERN_DUTY_CYCLE):
        """
        Calculates the pattern for a given frequency and duty cycle

        The pattern is calculated to represent a single cycle of a pulse with the specified frequency and duty cycle.
        If the pattern requires more than 16 periods, the clock divider is increased to reduce the period length.

        :param frequency: Frequency of the pattern in Hz
        :param duty_cycle: Duty cycle of the pattern
        :returns: Tuple of lists of levels and lengths, and the clock divider setting
        """
        clk_div_n = 0
        while clk_div_n < 6:        
            clk_n = self.bf_clk / (2**clk_div_n)
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
                return per_levels, per_lengths, clk_div_n
            else:
                clk_div_n += 1
        raise ValueError(f"Pattern requires too many periods ({len(per_levels)} > {MAX_PATTERN_PERIODS})")
    
    def set_pulse_profile(self, 
                          frequency:float, 
                          cycles:int, 
                          profile:int = 1, 
                          duty_cycle:float=DEFAULT_PATTERN_DUTY_CYCLE, 
                          tail_count:int=DEFAULT_TAIL_COUNT,
                          invert:bool=False):
        """
        Sets a pulse pattern profile
        
        :param frequency: Frequency of the pulse in Hz
        :param cycles: Number of pulse cycles
        :param profile: Pattern profile number
        :param duty_cycle: Duty cycle of the pulse
        :param tail_count: Number of tail counts
        :param invert: Invert the pulse
        """
        if profile not in PATTERN_PROFILES:
            raise ValueError(f"Invalid profile {profile}.")
        levels, lengths, clk_div_n = self.calc_pulse_pattern(frequency, duty_cycle)
        nperiods = len(levels)
        clk_div = 2**clk_div_n
        clk_n = self.bf_clk / clk_div
        level_lut = {-1: 0b01, 0: 0b00, 1: 0b10}
        for i, (level, length) in enumerate(zip(levels, lengths)):
            self._set_pattern_value(period=i+1, level=level_lut[level], length=length, profile=profile)
        if nperiods< MAX_PATTERN_PERIODS:
            self._set_pattern_value(period=nperiods+1, level=0b111, length=0, profile=profile)
        if cycles > (MAX_REPEAT+1):
            # Use elastic repeat
            pulse_duration_samples = cycles * self.bf_clk / frequency
            repeat = 1
            elastic_repeat = int(pulse_duration_samples / 16)
            period_samples = int(clk_n / frequency)
            cycles = 16*elastic_repeat * (1 / self.bf_clk) / period_samples
            elastic_mode = 1
            if elastic_repeat > MAX_ELASTIC_REPEAT:
                raise ValueError(f"Pattern duration too long for elastic repeat")
        else:
            repeat = cycles-1
            elastic_repeat = 0        
            elastic_mode = 0
        reg_mode =  0x02000003
        reg_mode = self._set_register_value(reg_mode, clk_div_n, lsb=3, width=3)
        reg_mode = self._set_register_value(reg_mode, int(invert), lsb=6, width=1)
        reg_repeat = 0
        reg_repeat = self._set_register_value(reg_repeat, repeat, lsb=1, width=5)
        reg_repeat = self._set_register_value(reg_repeat, tail_count, lsb=6, width=5)
        reg_repeat = self._set_register_value(reg_repeat, elastic_mode, lsb=11, width=1)
        reg_repeat = self._set_register_value(reg_repeat, elastic_repeat, lsb=12, width=16)
        registers = {ADDRESS_PATTERN_MODE: reg_mode,
                     ADDRESS_PATTERN_REPEAT: reg_repeat}
        reg_pat_sel = 0
        reg_pat_sel = self._set_register_value(reg_pat_sel, profile-1, lsb=0, width=6)
        registers[ADDRESSES_PATTERN_SEL[0]] = reg_pat_sel
        registers[ADDRESSES_PATTERN_SEL[1]] = reg_pat_sel
        pattern_registers = {}
        for period in range(1, MAX_PATTERN_PERIODS+1):
            address, _, _ = self._get_pattern_location(period, profile)
            if address not in pattern_registers:
                pattern_registers[address] = self.get_register(address)

        self.pulse_profiles[profile] = {
            'frequency': frequency,
            'duty_cycle': duty_cycle,
            'cycles': cycles,
            'tail_count': tail_count,
            'invert': invert,
            'clk_div': clk_div,
            'registers': registers,
            'pattern_registers': pattern_registers}

        if profile == self.pulse_profile or self.pulse_profile is None:
            self.activate_pulse_profile(profile)
            
    def get_pulse_profile(self, profile:int=1):
        """
        Retrieve a pulse profile
        
        :param profile: Pulse profile number
        :returns: Dictionary of pulse profile information
        """
        if profile not in PATTERN_PROFILES:
            raise ValueError(f"Invalid profile {profile}.")
        return self.pulse_profiles[profile]
    
    def activate_pulse_profile(self, profile:int):
        """
        Activates a pulse profile

        The pulse profile is activated by setting the pattern mode and repeat registers to the values stored in the profile.
        :param profile: Pulse profile number
        """
        if profile not in PATTERN_PROFILES:
            raise ValueError(f"Invalid profile {profile}.")
        self.pulse_profile = profile
        for addr, value in self.pulse_profiles[profile]['registers'].items():
            self.set_register(addr, value)
