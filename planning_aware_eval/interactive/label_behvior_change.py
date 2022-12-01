import numpy as np
import json


def main(json_file):
    f = open(json_file+'/ego_data.json')
    data = json.load(f)
    f.close()

    frames = []
    velocity = []
    steer = []
    brake = []
    throttle = []
    acc = []

    change = None
    steer_in_row = 0

    for frame in data:
        if 'speed' in data[frame] and 'control' in data[frame] and 'imu' in data[frame]:

            frames.append(frame)
            velocity.append(data[frame]['speed']['constant'])
            steer.append(data[frame]['control']['steer'])
            brake.append(data[frame]['control']['brake'])
            throttle.append(data[frame]['control']['throttle'])

            # acc_x = data[frame]['imu']['accelerometer_x']
            # acc_y = data[frame]['imu']['accelerometer_y']
            # acc.append((acc_x**2+acc_y**2)**0.5)

            print(frame)
            print('velocity:', data[frame]['speed']['constant'])
            print('steer:', data[frame]['control']['steer'])
            print('brake:', data[frame]['control']['brake'])
            print('throttle: ', data[frame]['control']['throttle'])
            print()

            if brake[-1] != 0 or (len(throttle) > 2 and throttle[-2] > 0 and throttle[-1] == 0):
                change = frame
                break

            if abs(steer[-1]) > 0.07:
                steer_in_row += 1
                change = frame
                if steer_in_row > 2:
                    break
            else:
                steer_in_row = 0

    n_frame = len(frames)
    print('behavior change at frame: ', change)

    f = open(json_file+'/behavior_change.txt', 'w')
    f.write(str(change))
    f.close()


if __name__ == '__main__':
    main('3_i-2_0_f_sr')
