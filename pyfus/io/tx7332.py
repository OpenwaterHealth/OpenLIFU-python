from dataclasses import dataclass, field
from typing import Tuple, Optional, List
from pyfus.util.units import getunitconversion
import numpy as np

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
DELAY_PROFILES_START = 0x20
DELAY_PROFILE_OFFSET = 16
MIN_DELAY_PROFILE = 1
MAX_DELAY_PROFILE = 16
DELAY_WIDTH = 13

APODIZATION_ADDRESS=0x1B
APODIZATION_CHANNEL_ORDER = [17, 19, 21, 23, 25, 27, 29, 31, 18, 20, 22, 24, 26, 28, 30, 32, 1, 3, 5, 7, 9, 11, 13, 15, 2, 4, 6, 8, 10, 12, 14, 16]

PATTERN_PROFILES_START = 0x120
PATTERN_PROFILE_OFFSET = 4
MIN_PATTERN_PROFILE = 1
MAX_PATTERN_PROFILE = 32
MAX_PATTERN_PERIOD = 16
PATTERN_ORDER = [[1, 2, 3, 4],
                 [5, 6, 7, 8],
                 [9, 10, 11, 12],
                 [13, 14, 15, 16]]
PATTERN_LENGTH_WIDTH = 5
PATTERN_LENGTH_MIN = 0
PATTERN_LENGTH_MAX = 30
PATTERN_LEVEL_WIDTH = 3
PATTERN_MAP = {}
for row, periods in enumerate(PATTERN_ORDER):
    for i, period in enumerate(periods):
        PATTERN_MAP[period] = {'row': row, 'lsb_lvl': i*(PATTERN_LEVEL_WIDTH+PATTERN_LENGTH_WIDTH), 'lsb_length': i*(PATTERN_LENGTH_WIDTH+PATTERN_LEVEL_WIDTH)+PATTERN_LEVEL_WIDTH}

@dataclass
class TX7332:
    _regs: List[int] = field(default_factory=lambda: [0]*MAX_REGISTER, init=False, repr=False)
    clk_freq: int = 64e6
    
    def get_register(self, address:int, lsb:int=0, width: Optional[int]=None):
        if width is None:
            width = REGISTER_WIDTH - lsb
        mask = (1 << width) - 1
        return (self._regs[address] >> lsb) & mask
    
    def get_bit(self, address:int, bit:int):
        return self.get_register(address, lsb=bit, width=1)
    
    def set_register(self, address, value:int, lsb:int=0, width: Optional[int]=None):
        if width is None:
            width = REGISTER_WIDTH - lsb
        mask = (1 << width) - 1
        if value < 0 or value > mask:
            raise ValueError(f"Value {value} does not fit in {width} bits")
        self._regs[address] = (self._regs[address] & ~(mask << lsb)) | ((value & mask) << lsb)

    def set_bit(self, address, bit:int, value:int):
        if value not in [0, 1]:
            raise ValueError(f"Value must be 0 or 1")
        self.set_register(address, value, lsb=bit, width=1)

    @staticmethod
    def _get_delay_location(channel:int, profile:int=1):
        if channel not in DELAY_CHANNEL_MAP:
            raise ValueError(f"Invalid channel {channel}.")
        channel_map = DELAY_CHANNEL_MAP[channel]
        if profile < MIN_DELAY_PROFILE or profile > MAX_DELAY_PROFILE:
            raise ValueError(f"Invalid Profile {profile}")
        address = DELAY_PROFILES_START + (profile-1) * DELAY_PROFILE_OFFSET + channel_map['row']
        lsb = channel_map['lsb']
        return address, lsb

    def _get_delay_value(self, channel:int, profile:int=1):
        address, lsb = self._get_delay_location(channel, profile)
        return self.get_register(address, lsb=lsb, width=DELAY_WIDTH)    
    
    def _set_delay_value(self, channel:int, value:int, profile:int=1):
        address, lsb = self._get_delay_location(channel, profile)
        self.set_register(address, value, lsb=lsb, width=DELAY_WIDTH)

    def get_delay(self, channel, profile:int=1, units='s'):
        return self._get_delay_value(channel, profile) / self.clk_freq * getunitconversion('s', units)

    def set_delay(self, channel, delay, profile:int=1, units='s'):
        self._set_delay_value(channel, int(delay * getunitconversion(units, 's') * self.clk_freq), profile)

    def get_apodization(self, channel:int):
        if channel not in APODIZATION_CHANNEL_ORDER:
            raise ValueError(f"Invalid channel {channel}")
        return 1-self.get_bit(APODIZATION_ADDRESS, bit=APODIZATION_CHANNEL_ORDER.index(channel))
    
    def get_apodizations_register(self):
        return self.get_register(APODIZATION_ADDRESS)

    def set_apodization(self, channel, apod: int):
        if apod not in [0, 1]:
            raise ValueError(f"Apodization value must be 0 or 1")
        if channel not in APODIZATION_CHANNEL_ORDER:
            raise ValueError(f"Invalid channel {channel}")
        self.set_bit(APODIZATION_ADDRESS, bit=APODIZATION_CHANNEL_ORDER.index(channel), value=1-apod)

    def set_apodizations(self, apodizations: List[int]):
        if len(apodizations) != 32:
            raise ValueError(f"Apodizations list must have 32 elements")
        for i, apod in enumerate(apodizations):
            self.set_apodization(i+1, apod)

    def get_apodizations(self):
        return [self.get_apodization(channel) for channel in range(1, 33)]
    
    @staticmethod
    def _get_pattern_location(period:int, profile:int=1):
        if period not in PATTERN_MAP:
            raise ValueError(f"Invalid period {period}.")
        if profile < MIN_PATTERN_PROFILE or profile > MAX_PATTERN_PROFILE:
            raise ValueError(f"Invalid profile {profile}.")
        address = PATTERN_PROFILES_START + (profile-1) * PATTERN_PROFILE_OFFSET + PATTERN_MAP[period]['row']
        lsb_lvl = PATTERN_MAP[period]['lsb_lvl']
        lsb_period = PATTERN_MAP[period]['lsb_period']
        return address, lsb_lvl, lsb_period
    
    def _get_pattern_value(self, period:int, profile:int=1):
        address, lsb_lvl, lsb_length = self._get_pattern_location(period, profile)
        level = self.get_register(address, lsb=lsb_lvl, width=PATTERN_LEVEL_WIDTH)
        length = self.get_register(address, lsb=lsb_length, width=PATTERN_LENGTH_WIDTH)
        return level, length
    
    def _set_pattern_value(self, period:int, level:int, length:int, profile:int=1):
        address, lsb_lvl, lsb_length = self._get_pattern_location(period, profile)
        self.set_register(address, level, lsb=lsb_lvl, width=PATTERN_LEVEL_WIDTH)
        self.set_register(address, length, lsb=lsb_length, width=PATTERN_LENGTH_WIDTH)

    def calc_pattern(self, freq:float, duty_cycle:float=0.66):
        period_samples = int(self.clk_freq / freq)
        if period_samples < 2 or period_samples > (PATTERN_LENGTH_MAX * MAX_PATTERN_PERIOD):
            raise ValueError(f"Frequency {freq} is out of range")
        first_half_period_samples = int(period_samples / 2)
        second_half_period_samples = period_samples - first_half_period_samples
        first_on_samples = int(first_half_period_samples * duty_cycle)
        first_off_samples = first_half_period_samples - first_on_samples
        second_on_samples = int(second_half_period_samples * duty_cycle)
        second_off_samples = second_half_period_samples - second_on_samples
        period = 1
        levels = [1, 0, -1, 0]
        per_lengths = []
        per_levels = []
        for i, samples in enumerate([first_on_samples, first_off_samples, second_on_samples, second_off_samples]):
            while samples > 0:
                if samples > PATTERN_LENGTH_MAX+4:
                    per_lengths.append(PATTERN_LENGTH_MAX)
                    per_levels.append(levels[i])
                    samples -= (PATTERN_LENGTH_MAX+2)
                else:
                    per_lengths.append(samples-2)
                    per_levels.append(levels[i])
                    samples = 0
        if len(per_levels) > MAX_PATTERN_PERIOD:
            raise ValueError(f"Pattern is too long")
        return per_levels, per_lengths