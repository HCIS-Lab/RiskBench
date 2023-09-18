import numpy as np
import math
import cv2
import pandas as pd
import time

class KalmanFilter:

    kf = cv2.KalmanFilter(4, 2)
    kf.measurementMatrix = np.array([[1, 0, 0, 0], [0, 1, 0, 0]], np.float32)
    kf.transitionMatrix = np.array(
        [[1, 0, 1, 0], [0, 1, 0, 1], [0, 0, 1, 0], [0, 0, 0, 1]], np.float32)

    def Estimate(self, coordX, coordY):
        ''' This function estimates the position of the object'''
        measured = np.array([[np.float32(coordX)], [np.float32(coordY)]])
        self.kf.correct(measured)
        predicted = self.kf.predict()
        return predicted

def angle_vectors(v1, v2):
    """ Returns angle between two vectors.  """
    v1_u = v1 / np.linalg.norm(v1)
    v2_u = v2 / np.linalg.norm(v2)
    angle = np.arccos(np.clip(np.dot(v1_u, v2_u), -1.0, 1.0))
    if math.isnan(angle):
        return 0.0
    else:
        return angle






def kf_inference(specific_frame, vehicle_list, variant_ego_id, pedestrian_id_list, vehicle_id_list):

    start_time = time.time()

    vehicle_length = 4.7
    vehicle_width = 2
    pedestrian_length = 0.8
    pedestrian_width = 0.8
    agent_area = [[vehicle_length, vehicle_width],
                [pedestrian_length, pedestrian_width]]
    future_len = 20
    risky_vehicle_list = []
    



    for val_vehicle_num in range(len(vehicle_list)):
        now_id = vehicle_list[val_vehicle_num].TRACK_ID[0]
        if int(now_id) == int(variant_ego_id):
            x = vehicle_list[val_vehicle_num].X.to_numpy()
            y = vehicle_list[val_vehicle_num].Y.to_numpy()
            kf = KalmanFilter()
            x_pred = []
            y_pred = []
            count = 0
            for x_now, y_now in zip(x[0:0 + 20], y[0:0 + 20]):
                count += 1
                if count <= 20:
                    pred = kf.Estimate(x_now, y_now)
                    x_pred.append(pred[0])
                    y_pred.append(pred[1])
            for temp in range(future_len):
                pred = kf.Estimate(
                    x_pred[temp + 19], y_pred[temp + 19])
                x_pred.append(pred[0])
                y_pred.append(pred[1])
            ego_prediction = np.zeros((len(x_pred), 2))
            for pred_t in range(len(x_pred) - 21):
                real_pred_x = x_pred[pred_t + 20]
                real_pred_x_next = x_pred[pred_t + 21]
                real_pred_y = y_pred[pred_t + 20]
                real_pred_y_next = y_pred[pred_t + 21]
                ego_prediction[pred_t][0] = real_pred_x
                ego_prediction[pred_t][1] = real_pred_y
                if pred_t == int(len(x_pred) - 22):
                    ego_prediction[pred_t +
                                    1][0] = real_pred_x_next
                    ego_prediction[pred_t +
                                    1][1] = real_pred_y_next


    for val_vehicle_num in range(len(vehicle_list)):
        vl = vehicle_list[val_vehicle_num].to_numpy()
        x = vehicle_list[val_vehicle_num].X.to_numpy()
        y = vehicle_list[val_vehicle_num].Y.to_numpy()
        x = x.astype(float)
        y = y.astype(float)
        kf = KalmanFilter()
        x_pred = []
        y_pred = []
        count = 0
        for x_now, y_now in zip(x[0:0 + 20], y[0:0 + 20]):
            count += 1
            if count <= 20:
                pred = kf.Estimate(x_now, y_now)
                x_pred.append(pred[0])
                y_pred.append(pred[1])
        for temp in range(future_len):
            pred = kf.Estimate(
                x_pred[temp + 19], y_pred[temp + 19])
            x_pred.append(pred[0])
            y_pred.append(pred[1])
        now_id = int(vehicle_list[val_vehicle_num].TRACK_ID[0])


        agent_type = 0

        if str(int(now_id)) in pedestrian_id_list:
            agent_type = 1
        elif str(int(now_id)) in vehicle_id_list:
            agent_type = 0

            
        for pred_t in range(0, future_len - 1, 3):
            if int(now_id) == int(variant_ego_id):
                continue
            real_pred_x = x_pred[pred_t + 20]
            real_pred_x_next = x_pred[pred_t + 21]
            real_pred_y = y_pred[pred_t + 20]
            real_pred_y_next = y_pred[pred_t + 21]
            if True:
                center_distance_vector = [
                    real_pred_x - ego_prediction[pred_t][0], real_pred_y - ego_prediction[pred_t][1]]
                vehicle_vector_square = math.sqrt(
                    (real_pred_x_next - real_pred_x)**2 + (real_pred_y_next - real_pred_y)**2)
                if vehicle_vector_square == 0:
                    vehicle_vector_square = 2147483647
                vehicle_cos = (real_pred_x_next -
                                real_pred_x) / vehicle_vector_square
                vehicle_sin = (real_pred_y_next -
                                real_pred_y) / vehicle_vector_square
                vehicle_axisX_1 = vehicle_cos * \
                    agent_area[agent_type][0] / 2
                vehicle_axisY_1 = vehicle_sin * \
                    agent_area[agent_type][0] / 2
                vehicle_axisX_2 = vehicle_sin * \
                    agent_area[agent_type][1] / 2
                vehicle_axisY_2 = - vehicle_cos * \
                    agent_area[agent_type][1] / 2
                ego_vector_square = math.sqrt((ego_prediction[pred_t + 1][0] - ego_prediction[pred_t][0])**2 + (
                    ego_prediction[pred_t + 1][1] - ego_prediction[pred_t][1])**2)
                ego_cos = (
                    ego_prediction[pred_t + 1][0] - ego_prediction[pred_t][0]) / ego_vector_square
                ego_sin = (
                    ego_prediction[pred_t + 1][1] - ego_prediction[pred_t][1]) / ego_vector_square
                ego_axisX_1 = ego_cos * \
                    agent_area[agent_type][0] / 2
                ego_axisY_1 = ego_sin * \
                    agent_area[agent_type][0] / 2
                ego_axisX_2 = ego_sin * \
                    agent_area[agent_type][1] / 2
                ego_axisY_2 = - ego_cos * \
                    agent_area[agent_type][1] / 2
                if abs(center_distance_vector[0] * vehicle_cos + center_distance_vector[1] * vehicle_cos) <=\
                        abs(ego_axisX_1 * vehicle_cos + ego_axisY_1 * vehicle_sin) + abs(ego_axisX_2 * vehicle_cos + ego_axisY_2 * vehicle_sin) + agent_area[agent_type][0] / 2 and \
                        abs(center_distance_vector[0] * vehicle_sin - center_distance_vector[1] * vehicle_cos) <=\
                        abs(ego_axisX_1 * vehicle_cos - ego_axisY_1 * vehicle_cos) + abs(ego_axisX_2 * vehicle_sin - ego_axisY_2 * vehicle_cos) + agent_area[agent_type][1] / 2 and \
                        abs(center_distance_vector[0] * ego_cos + center_distance_vector[1] * ego_sin) <=\
                        abs(vehicle_axisX_1 * ego_cos + vehicle_axisY_1 * ego_sin) + abs(vehicle_axisX_2 * ego_cos + vehicle_axisY_2 * ego_sin) + agent_area[agent_type][0] / 2 and \
                        abs(center_distance_vector[0] * ego_sin - center_distance_vector[1] * ego_cos) <=\
                        abs(vehicle_axisX_1 * ego_cos - vehicle_axisY_1 * ego_cos) + abs(vehicle_axisX_2 * ego_sin + vehicle_axisY_2 * ego_cos) + agent_area[agent_type][1] / 2:
                    risky_vehicle_list.append(
                        [str(specific_frame), int(now_id)])
    file_d = {}

    for frame, id in risky_vehicle_list:
        if frame in file_d:
            if str(id) in file_d[frame]:
                continue
            else:
                file_d[frame].append(str(id))
        else:
            file_d[frame] = [str(id)]
    # check all prediction result
    # print(file_d)
    if str(specific_frame) in file_d:
        risky_id = file_d[str(specific_frame)]
    else:
        risky_id = []
    return risky_id
    


    #end_time = time.time()
    #print( 'Frame:', specific_frame,  ' risky_id:',  risky_id, 'time', start_time-end_time)


