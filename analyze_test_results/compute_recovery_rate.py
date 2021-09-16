import numpy as np

# (RCW, RCCW, TR, TL)
rgb_il = [[1, 1, 0, 2, 1], [0, 0, 1, 0, 2]]
rgb_gpl = [[3, 3, 2, 3, 3], [3, 3, 2, 2, 3]]

lidar_il = [[0, 1, 1, 1, 0], [0, 0, 0, 0, 2]]
lidar_gpl = [[3, 3, 3, 3, 2], [1, 3, 2, 2, 3]]

event_il = [[0, 0, 0, 0, 0], [0, 0, 0, 0, 0]]
event_gpl = [[3, 2, 2, 3, 2], [3, 3, 3, 3, 3]]

def compute(data):
    # data = [vv / 3. for v in data for vv in v]
    data = np.array(data).sum(0) / 6.
    # import pdb; pdb.set_trace()
    return np.mean(data), np.std(data)

print(compute(rgb_il))
print(compute(rgb_gpl))
print(compute(lidar_il))
print(compute(lidar_gpl))
print(compute(event_il))
print(compute(event_gpl))