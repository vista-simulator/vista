import numpy as np

"""
(RCW, RCCW, TR, TL)

Lidar
- IL: 
    1. (0/3, 0/3)
    2. (1/3, 0/3)
    3. (1/3, 0/3)
    4. (1/3, 0/3)
    5. (0/3, 2/3)
- GPL: 
    1. (3/3, 1/3) 
    2. (3/3, 3/3) 
    3. (3/3, 2/3) 
    4. (3/3, 2/3)
    5. (2/3, 3/3)

event
- IL: 
    1. (0/3, 0/3)
    2. (0/3, 0/3)
    3. (0/3, 0/3)
    4. 
    5. 
- GPL: 
    1. (3/3, 3/3) 
    2. (2/3, 3/3) 
    3. (2/3, 3/3) 
    4. (3/3, 3/3)
    5. (2/3, 3/3)

RGB
- IL: 
    1. (1/3, 0/3) 
    2. (1/3, 0/3)
    3. (0/3, 1/3)
    4. (2/3, 0/3)
    5. (1/3, 2/3)
- GPL: 
    1. (3/3, 3/3)
    2. (3/3, 1/3)
    3. (2/3, 2/3)
    4. (3/3, 2/3)
    5. (3/3, 3/3)
"""

rgb_il = [[1, 1, 0, 2, 1], [0, 0, 1, 0, 2]]
rgb_gpl = [[3, 3, 2, 3, 3], [3, 3, 2, 2, 3]]

lidar_il = [[0, 1, 1, 1, 0], [0, 0, 0, 0, 2]]
lidar_gpl = [[3, 3, 3, 3, 2], [1, 3, 2, 2, 3]]

event_il = [[0, 0, 0, 0, 0], [0, 0, 0, 0, 0]]
event_gpl = [[3, 2, 2, 3, 2], [3, 3, 3, 3, 3]]

def compute(data):
    data = [vv / 3. for v in data for vv in v]
    return np.mean(data), np.std(data)

print(compute(rgb_il))
print(compute(rgb_gpl))
print(compute(lidar_il))
print(compute(lidar_gpl))
print(compute(event_il))
print(compute(event_gpl))