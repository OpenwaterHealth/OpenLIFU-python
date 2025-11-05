import pyvisa as visa
import matplotlib.pyplot as plt
import numpy as np
# import csv
# import time
# import serial
# import re
import math

class Rigol():

    # total display time window on osc. in nanoseconds
    def __init__(self, channels=1, window=1000, data_length=1200, volt_multiplier=(1, )):
        self.channels = channels
        rm = visa.ResourceManager()
        self.inst = rm.open_resource(
            'USB0::0x1AB1::0x04CE::DS1ZA232605458::INSTR')
        self.inst.write(':wav:form ascii')
        self.window = window
        self.data_length = data_length
        self.volt_multiplier = volt_multiplier
        self.data_array = np.empty((0, channels))
        self.time_array = np.empty((0))
        self.waveform_length = float(self.inst.query(
            ':ACQuire:SRATe?')) * float(self.inst.query(':TIM:MAIN:SCAL?')) * 12
        self.data_to_send = dict(data=None, running=True)

    def getFreq(self, channel=1):
        self.inst.write(f':meas:sour chan{channel}')
        return float(self.inst.query(':MEAS:ITEM? FREQ'))

    def acquireOnce(self, channel=1):
        self.inst.write(f':meas:sour chan{channel}')
        return float(self.inst.query(':MEAS:ITEM? VPP'))

    def StoreData(self, x, y, z, data, pos, points, value):
        data[pos, 0] = value * 1000
        data[pos, 1] = x
        data[pos, 2] = y
        data[pos, 3] = z
        data[pos, 4] = value * 1000
        data[pos, 5] = data[pos, 4] / math.sqrt(2)
        data[pos, 6] = data[pos, 5] / 327
        data[pos, 7] = data[pos, 6] * math.sqrt(2)
        return data

    def SaveHydrophoneCSV(self, data, headers, filename=None):
        if filename is None:
            filename = 'Hydrophone_Data.csv'
        np.savetxt(filename, data, delimiter=',', header=headers)

