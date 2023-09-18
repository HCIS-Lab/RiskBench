import carla
import numpy as np
from carla import Transform, Location, Rotation


def get_transform(np_transform):
    transform = carla.Transform(Location(np_transform[0], np_transform[1], np_transform[2]),
                                Rotation(np_transform[3], np_transform[4], np_transform[5]))
    return transform


def read_control(path='control.npy'):
    """ param:

    """
    control = np.load(path)
    control_list = []
    init_transform = control[0]
    init_transform = carla.Transform(Location(x=control[0][0], y=control[0][1], z=control[0][2]+1),
                                     Rotation(pitch=control[0][3], yaw=control[0][4], roll=control[0][5]))
    for i in range(1, len(control)):
        control_list.append(carla.VehicleControl(float(control[i][0]), float(control[i][1]), float(control[i][2]), bool(control[i][3]),
                                                 bool(control[i][4]), bool(control[i][5]), int(control[i][6])))

    return init_transform, control_list


def read_transform(path='control.npy'):
    """ param:

    """
    transform_npy = np.load(path)
    transform_list = []
    for i, transform in enumerate(transform_npy):
        if i == 0:
            transform_list.append(carla.Transform(Location(x=transform[0], y=transform[1], z=transform[2]+1),
                                                  Rotation(pitch=transform[3], yaw=transform[4], roll=transform[5])))
        else:
            transform_list.append(carla.Transform(Location(x=transform[0], y=transform[1], z=transform[2]),
                                                  Rotation(pitch=transform[3], yaw=transform[4], roll=transform[5])))

    return transform_list

def read_ped_control(path='control.npy'):
    """ param:

    """
    control_npy = np.load(path)
    control_list = []
    for i, control in enumerate(control_npy):
        control_list.append(carla.WalkerControl(carla.Vector3D(x=control[0], y=control[1], z=control[2]+1),
                                              float(control[3]), bool(control[4])))
    return control_list


def read_velocity(path='velocity.npy'):
    velocity_npy = np.load(path)
    velocity_list = []
    for velocity in velocity_npy:
        v = (velocity[0]**2 + velocity[1]**2 + velocity[2]**2)**0.5
        velocity_list.append(v)
        # velocity_list.append(carla.Vector3D(x=velocity[0], y=velocity[1], z=velocity[2]))

    return velocity_list