import numpy as np
import json
import os
import csv
from collections import OrderedDict

root_dir = "/path/to/RiskBench_Dataset"
data_type = ["interactive", "non-interactive", "obstacle", "collision"][:]


def main(_type):

	data_root = os.path.join(root_dir, _type)

	for basic in sorted(os.listdir(data_root)):

		basic_path = os.path.join(data_root, basic, "variant_scenario")

		for variant in sorted(os.listdir(basic_path)):

			print(_type, basic, variant)

			save_path = f"../datasets/state/{_type}/{basic}_{variant}.json"
			state_dict = OrderedDict()

			variant_dir = os.path.join(basic_path, variant)

			bbox_path = os.path.join(variant_dir, "bbox.json")
			bbox_info = json.load(open(bbox_path))

			static_path = os.path.join(variant_dir, "actor_attribute.json")
			static_info = json.load(open(static_path))["obstacle"]

			if _type == "obstacle":
				obstacle_path = os.path.join(variant_dir, "obstacle_info.json")
				obstacle_info = json.load(open(obstacle_path))
			

			N = len(os.listdir(variant_dir+"/ego_data"))

			for frame in range(1, N+1):

				state_dict[str(frame)] = OrderedDict()

				ego_data_path = os.path.join(variant_dir, "ego_data", f"{frame:08d}.json")
				ego_data = json.load(open(ego_data_path))
				compass = ego_data["compass"]
				ego_x = ego_data["location"]["x"]
				ego_y = ego_data["location"]["y"]

				actor_data_path = os.path.join(variant_dir, "actors_data", f"{frame:08d}.json")
				actors_data = json.load(open(actor_data_path))
				actors_data.update(static_info)

				if _type == "obstacle":
					actors_data.update(obstacle_info)

				bboxes = bbox_info[f"{frame:08d}"]

				for actor_id in bboxes:
					
					if str(int(actor_id)%65536) in actors_data:
						src_id = str(int(actor_id)%65536)
					elif str(int(actor_id)%65536+65536) in actors_data:
						src_id = str(int(actor_id)%65536+65536)
					else:
						continue
					
					x = actors_data[src_id]["location"]["x"]
					y = actors_data[src_id]["location"]["y"]

					# clockwise
					theta = compass*np.pi/180.0
					R = np.array([[np.cos(theta), np.sin(theta)],
								[np.sin(theta), -np.cos(theta)]])
				
					new_actor_vec = related_vector([x, y], R=R, ego_loc=[ego_x, ego_y])
					state_dict[str(frame)][int(actor_id)] = new_actor_vec


			with open(save_path, "w") as f:
				json.dump(state_dict, f, indent=4)



def related_vector(points, R=None, ego_loc=[0., 0.]):


    points = np.array(points)-np.array(ego_loc)
    related_vec = R.T@points

    return related_vec.tolist()


def change_key():
	
	data_type = ["interactive", "non-interactive", "obstacle", "collision"][:]
	json_root = "../state"

	for json_file in os.listdir(json_root):
		json_path = json_root+"/"+json_file
		src_json = json.load(open(json_path))
		new_json = OrderedDict()

		for key in src_json:
			# if "waywaybao_cs10" in key:
			tokens = key.split('/')
			_type = tokens[6]
			basic = tokens[7]
			variant = tokens[9]

			new_json[_type+"_"+basic+"_"+variant] = src_json[key].copy()

			print(_type+"_"+basic+"_"+variant)

		with open(json_path, "w") as f:
			json.dump(new_json, f, indent=4)


if __name__ == "__main__":

	for _type in data_type[:2]:
	    main(_type)

    # change_key()
	