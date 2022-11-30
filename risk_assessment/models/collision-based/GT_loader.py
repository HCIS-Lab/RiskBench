import json

root = 'GT_loader/'
collision_data = root + 'new_collision_GT.json'
interactive_data = root + 'new_interactive_GT.json'
obstacle_data = root + 'new_obstacle_GT.json'
non_interactive_data = root + 'new_noninteractive_GT.json'

def jsonToDict(file_path):
    with open(file_path) as f:
        data = json.load(f)
        f.close()
        return data

def getGTframe(scenario_type, scenario_id, weather):
    '''
    parameters:
        scenario_type(str): 'collision', 'interactive', 'obstacle'
        scneario_id(str): ex. '7_t3-1_0_b_f_l_1_0'
        weathe(str)r: ex. 'ClearNoon_high_'
    return:
        int: gt_start, gt_end, actor_id
    
    example:
        GT_frame = getGTframe('collision', '10_i-1_1_c_r_l_0', 'ClearSunset_low_')
    '''
    try:
        if scenario_type == 'collision':
            data = jsonToDict(collision_data)
            frame_id = data[scenario_id][weather]
            gt_start = frame_id["gt_start_frame"]
            gt_end = frame_id["gt_end_frame"]
            for _, id in frame_id.items():
                return int(gt_start), int(gt_end), id
        
        if scenario_type == 'interactive':
            data = jsonToDict(interactive_data)
            frame_id = data[scenario_id][weather]
            gt_start = frame_id["gt_start_frame"]
            gt_end = frame_id["gt_end_frame"]
            for _, id in frame_id.items():
                return int(gt_start), int(gt_end), id
        
        if scenario_type == 'obstacle':
            data = jsonToDict(obstacle_data)
            return int(data[scenario_id][weather]['gt_start_frame']),int(data[scenario_id][weather]['gt_end_frame']),int(data[scenario_id][weather]['nearest_obstacle_id'])
    except:
        return None,None,None
#if __name__ == '__main__':
    #print(getGTframe('collision', '10_i-1_1_c_r_l_0', 'ClearSunset_low_'))

    #print(getGTframe('interactive', '5_i-1_0_m_l_f_1_0', 'ClearSunset_mid_'))

    #print(getGTframe('obstacle', '5_t1-2_1_r_sr', 'ClearNoon_high_'))
