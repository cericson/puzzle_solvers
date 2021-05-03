from com.android.monkeyrunner import MonkeyRunner, MonkeyDevice
import os
import subprocess
from time import sleep

HEX_TILE_DIAM = 156
HEX_COL_HEIGHTS = [4, 5, 6, 7, 6, 5, 4]
HEX_LOWER_EDGE_COORDS = [
    [135, 1696],
    [271, 1774],
    [406, 1852],
    [542, 1930],
    [676, 1852],
    [811, 1774],
    [945, 1696],
]

print('Connecting...')
device = MonkeyRunner.waitForConnection()

while True:
    print('Loading puzzle image...')
    result = device.takeSnapshot()
    script_dir = os.path.dirname(__file__)
    result.writeToFile(os.path.join(script_dir, 'screenshot.png'), 'png')

    print('Solving...')
    proc = subprocess.Popen(['python', os.path.join(script_dir, "solver.py")])
    proc.wait()

    f = open(os.path.join(script_dir, "presses.txt"), 'r')
    presses = [int(c) for c in f.read()]
    f.close()

    location_index = 0
    for point, col_height in zip(HEX_LOWER_EDGE_COORDS[:-1], HEX_COL_HEIGHTS[:-1]):
        x, y = point
        for i in range(col_height):
            for _ in range(presses[location_index]):
                device.touch(x, int(y - (i + 0.5) * HEX_TILE_DIAM), "DOWN_AND_UP")
                sleep(0.2)
            location_index += 1
    sleep(0.5)
    device.touch(550, 2100, "DOWN_AND_UP")
    sleep(0.5)
