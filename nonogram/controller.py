from com.android.monkeyrunner import MonkeyRunner, MonkeyDevice
import os
import sys
import subprocess
from time import sleep

print('Connecting...')
device = MonkeyRunner.waitForConnection()

SINGLE_RUN = False

if SINGLE_RUN:
    iterations = 1
else:
    iterations = 1000

for i in range(iterations):
    if not SINGLE_RUN:
        device.touch(540, 420, "DOWN_AND_UP")
        sleep(4.0)

    result = device.takeSnapshot()
    script_dir = os.path.dirname(__file__)
    result.writeToFile(os.path.join(script_dir, 'screenshot.png'), 'png')

    print('Solving...')
    proc = subprocess.Popen(['python', os.path.join(script_dir, "solver.py")])
    proc.wait()

    print('Entering solution...')
    touches_file = os.path.join(script_dir, "touch_locations.txt")
    f = open(touches_file, 'r')
    lines = f.readlines()
    for line in lines:
        x, y = [int(s) for s in line.split(' ')]
        device.touch(x, y, "DOWN_AND_UP")

    f.close()

    os.remove(touches_file)
    sleep(0.5)
    device.touch(540, 2240, "DOWN_AND_UP")  # Continue
    sleep(0.5)
    device.touch(730, 1850, "DOWN_AND_UP")  # OK
    sleep(0.75)
    device.touch(540, 2240, "DOWN_AND_UP")  # Continue
    sleep(0.5)
    # sometimes first press doesn't go through
    device.touch(540, 2240, "DOWN_AND_UP")  # Continue
    sleep(0.5)
