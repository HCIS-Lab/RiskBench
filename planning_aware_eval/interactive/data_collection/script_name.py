import os 

import random
scenario_type = "interactive"


weather=['ClearNoon','CloudyNoon','WetNoon','WetCloudyNoon','MidRainyNoon','HardRainNoon','SoftRainNoon',
	'ClearSunset','CloudySunset','WetSunset','WetCloudySunset','MidRainSunset','HardRainSunset','SoftRainSunset',
	'ClearNight','CloudyNight','WetNight','WetCloudyNight','MidRainyNight','HardRainNight','SoftRainNight']
random_actor=['low','mid','high']

dir_list = os.listdir(f"./{scenario_type}")


# print(random.choice(weather))
# print(random.randint(0, 10000000))


for name in dir_list:
    for _ in range(10):
        with open(f"./{scenario_type}/{name}/interactive_frame.txt") as f:
            start_frame = int(f.readline())
        print(f"{name} {random.choice(weather)} {random.choice(random_actor)} {start_frame} {random.randint(0, 10000000)}")

    # var_dir = os.listdir(f"./{scenario_type}/"+name+"/variant_scenario")




    # for var in var_dir:
    #     s = var.split("_")
        
    #     with open(f"./{scenario_type}/{name}/interactive_frame.txt") as f:
    #             start_frame = int(f.readline())

    #     # with open(f"./{scenario_type}/{name}/variant_scenario/{s[0]}_{s[1]}_/interactive_frame.txt") as f:
    #     #         start_frame = int(f.readline())

    #     print(f"{name} {s[0]} {s[1]} {start_frame} ")
    