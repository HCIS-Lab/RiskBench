import os
# Generate name list for collecting data
# useage: python get_name_list > name.txt

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
    if os.path.exists(f"./data_collection/{scenario}"):
        basic_scenario_list = sorted(os.listdir(f"./data_collection/{scenario}"))

        for basic_scenario in basic_scenario_list:
            
            town_id = basic_scenario.split("_")[0]
            if town_id[0] =="A" or town_id[0] =="B":
                town = town_id
            elif town_id[:2] == "10":
                town = "Town10HD"
            else:
                town = f"Town0{town_id[0]}"
            variant_scenario_list = sorted(os.listdir(f"./data_collection/{scenario}/{basic_scenario}/variant_scenario"))
            for variant_scenario in variant_scenario_list:

                weather = variant_scenario.split("_")[0]
                actor = variant_scenario.split("_")[1]
                with open(f"./data_collection/{scenario}/{basic_scenario}/variant_scenario/{variant_scenario}/seed.txt") as f:
                    random_seed = int(f.readline())
                print(scenario, basic_scenario, town, weather, actor, random_seed)
                break
     
        
                    
