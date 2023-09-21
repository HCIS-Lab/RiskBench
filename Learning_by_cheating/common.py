import numpy as np

COLOR = np.uint8([
        (  0,   0,   0),    # 0 unlabeled  # 0
        (105, 105, 105),    # 1 ROAD
        (255, 255, 255),    # 2 ROAD LINE
        (252, 175, 62),    # 3 VEHICLES
        (233, 185, 110),    # 4 PEDESTRIANS
        (50, 50, 50),    # 5 OBSTACLES
        (138, 226, 52),    # 6 AGENT

        ])

# COLOR = np.uint8([
#         (  0,   0,   0),    # unlabeled  # 0
#         (220,  20,  60),    # ped 
#         (157, 234,  50),    # road line
#         (128,  64, 128),    # road
#         (244,  35, 232),    # sidewalk
#         (  0,   0, 142),    # car
#         (255,   0,   0),    # red light
#         (255, 255,   0),    # yellow light
#         (  0, 255,   0),    # green light
#         (125,   125,   125),    # obstacle # 9
#         ])
