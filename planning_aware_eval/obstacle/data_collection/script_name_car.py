import os 

# scenario_type = "interactive"

#scenario_type = "non-interactive"

scenario_type = "car_obstacle"



dir_list = os.listdir(f"./{scenario_type}")




for name in dir_list:
    
    
    with open(f"./{scenario_type}/{name}/start_frame.txt") as f:
        start_frame = int(f.readline())
    with open(f"./{scenario_type}/{name}/obstacle_info.txt") as f:
        m = f.readline().split("\n")[0]
        m = m.split(" ")
        x = float(m[0])
        y = float(m[1])
    
    var_dir = os.listdir(f"./{scenario_type}/"+name+"/variant_scenario")
    for var in var_dir:
        s = var.split("_")
        
        
        #print("python data_generator_randomseed.py --scenario_id " + name + " --scenario_type obstacle --map Town10HD_opt --weather " + s[0] + " --random_actors "+ s[1] +" --replay --no_save")
    
        print(f"{name} {s[0]} {s[1]} {start_frame} {x} {y}")
    
    # var_list
