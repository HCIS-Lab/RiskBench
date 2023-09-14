from argparse import ArgumentParser
from dataset.common import iterate_dataset, parse_scenario_id

one_hot = {'r': 1, 'sl': 2, 'f': 3, 'gi': 4, 'l': 5, 'gr': 6, 'u': 7, 'sr': 8,'er': 9}

def get_parser():
    parser = ArgumentParser()
    parser.add_argument("--model_path",type=str,default='logs')
    parser.add_argument("--batch",type=int,default=10)
    parser.add_argument("--epoch",type=int,default=10)
    parser.add_argument("--lr",type=float,default=1e-3)
    parser.add_argument("--wd",type=float,default=1e-2)
    parser.add_argument("--root",type=str,default='../dataset/')
    parser.add_argument("--load_first",action='store_true',default=False)
    parser.add_argument("--intention",action='store_true',default=False)
    parser.add_argument("--supervised",action='store_true',default=False)
    parser.add_argument("--state",action='store_true',default=False)

    args = parser.parse_args()
    return args

def get_one_hot(s_type,s_id):
    assert isinstance(s_id,list)
    intention = []
    for type,id in zip(s_type,s_id):
        # index = one_hot.get(parse_scenario_id(type,id)['ego_intention'],0)
        index = one_hot[parse_scenario_id(type,id)['ego_intention']]
        out = [1 if i== (index-1) else 0 for i in range(9)]
        intention.append(out)
    return intention

if __name__ == '__main__':
    root = "/home/hcis-s19/Desktop/dataset"
    result = iterate_dataset(root,types=['collision','interactive','non-interactive','obstacle'])
    print(result)