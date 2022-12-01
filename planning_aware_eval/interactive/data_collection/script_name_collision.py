import os 

# scenario_type = "interactive"

#scenario_type = "non-interactive"

scenario_type = "collision"



dir_list = os.listdir(f"./{scenario_type}")


for name in dir_list:
    f = open(f"./{scenario_type}/{name}/target_point.txt")
    line = f.readline().split(" ")
    target_x = float(line[0])
    target_y = float(line[1])
    # print(target_x, target_y)
    
    var_dir = os.listdir(f"./{scenario_type}/"+name+"/variant_scenario")

    for var in var_dir:
        s = var.split("_")
        with open(f"./{scenario_type}/{name}/variant_scenario/{s[0]}_{s[1]}_/interactive_frame.txt") as f:
                start_frame = int(f.readline())
        
        print(f"{name} {s[0]} {s[1]} {target_x} {target_y} {start_frame}")
    
    # var_list
