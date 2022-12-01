import os 

# scenario_type = "interactive"

#scenario_type = "non-interactive"

scenario_type = "obstacle"



dir_list = os.listdir(f"./{scenario_type}")




for name in dir_list:
    
    with open(f"./{scenario_type}/{name}/start_frame.txt") as f:
        start_frame = int(f.readline())


    var_dir = os.listdir(f"./{scenario_type}/"+name+"/variant_scenario")
    for var in var_dir:
        s = var.split("_")
        
        
        print(f"{name} {s[0]} {s[1]} {start_frame} 0.0 0.0")
    
    # var_list
