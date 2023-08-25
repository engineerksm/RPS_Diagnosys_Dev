#!/usr/bin/env python3

import time
from libreVNA import libreVNA

# Create the control instance
vna = libreVNA('localhost', 19542)

# Quick connection check (should print "LibreVNA-GUI")
print(vna.query("*IDN?"))

# Make sure we are connecting to a device (just to be sure, with default settings the LibreVNA-GUI auto-connects)
vna.cmd(":DEV:CONN")
dev = vna.query(":DEV:CONN?")
if dev == "Not connected":
    print("Not connected to any device, aborting")
    exit(-1)
else:
    print("Connected to " + dev)

# Simple generator demo

# switch to generator
# vna.cmd(":DEV:MODE GEN")
# switch to VNA
vna.cmd(":DEV:MODE VNA")

# set the output level
# vna.cmd(":GEN:LVL -20")
sweep_freq = vna.query(":VNA:SWEEP?")
vna.cmd(":VNA:SWEEP " + sweep_freq)

# set initial frequency and enable port 1
# print("Generating signal with 1GHz at port 1")
# vna.cmd(":GEN:FREQ 1000000000")
# vna.cmd(":GEN:PORT 1")
start_freq = vna.query(":VNA:FREQ:START?")
vna.cmd(":VNA:FREQ:START " + start_freq)
center_freq = vna.query(":VNA:FREQ:CENT?")
vna.cmd(":VNA:FREQ:CENT " + center_freq)
stop_freq = vna.query(":VNA:FREQ:STOP?")
vna.cmd(":VNA:FREQ:STOP " + stop_freq)


try:
    trace_list = vna.query(":VNA:TRAC:LIST?")
    print(trace_list)   # S11, S12, S21, S22

    while True:
        time.sleep(2)
        # print("Setting frequency to 1.5GHz")
        # vna.cmd(":GEN:FREQ 1500000000")
        print("Frequency sweep data at " + trace_list[0])
        trace_data_s11 = vna.query(":VNA:TRAC:DATA? " + trace_list[0])
        time.sleep(5)
        # print("Setting frequency to 1.0GHz")
        # vna.cmd(":GEN:FREQ 1000000000")
        print("Frequency sweep data at " + trace_list[1])
        trace_data_s12 = vna.query(":VNA:TRAC:DATA? " + trace_list[1])
        time.sleep(5)

except KeyboardInterrupt:
    # turn off generator
    vna.cmd(":GEN:PORT 0")
    exit(0)

