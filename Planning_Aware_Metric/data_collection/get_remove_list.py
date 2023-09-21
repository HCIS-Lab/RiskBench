import os
# Find bad variant scenario 
# You need to manually remove it 

if __name__ =="__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument( "-s",
                        "--scenario_type",
                        type=str,
                        default="interactive", # "obstacle", "non-interactive", "collision"
                        )

    args = parser.parse_args()
    scenario = args.scenario_type
    if os.path.exists(f"./{scenario}"):
        basic_scenario_list = sorted(os.listdir(f"./{scenario}"))

        for basic_scenario in basic_scenario_list:
            
            town_id = basic_scenario.split("_")[0]
            if town_id[0] =="A" or town_id[0] =="B":
                town = town_id
            elif town_id[:2] == "10":
                town = "Town10HD"
            else:
                town = f"Town0{town_id[0]}"
            variant_scenario_list = sorted(os.listdir(f"./{scenario}/{basic_scenario}/variant_scenario"))
            for variant_scenario in variant_scenario_list:

            

                weather = variant_scenario.split("_")[0]
                actor = variant_scenario.split("_")[1]

                # check random_seed exists? 
                if not os.path.exists(f"./{scenario}/{basic_scenario}/variant_scenario/{variant_scenario}/seed.txt"):
                    print(f"rm -r ./{scenario}/{basic_scenario}/variant_scenario/{variant_scenario}")
                
     
        
                    
