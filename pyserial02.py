#berhasil
#!/usr/bin/env python3
import serial
import time
import numpy as np

ser = serial.Serial('COM4',9600)

while ser :
    ser.reset_input_buffer()
    line = ser.readline().decode('utf-8').rstrip()
    print(line)
    # input pressure value (bar)
    val_bar = 0.06
    bit0 = val_bar * 256 / 1
    bit_ard = np.clip(round(bit0, 2), 0, 255)
    ser.write(bytes(str(bit_ard),"utf-8"))
    print("success", bit_ard)