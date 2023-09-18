


state_names = [
            " 0  - No mask",
            " 1  - Ground Truth",
            " 2  - Random",
            " 3  - Nearest",
            " 4  - AutoPilot",
            " 5  - RSS",
            " 6  - Kalman Filter",
            " 7  - Social-GAN",
            " 8  - MANTRA",
            " 9  - DSA-RNN",
            " 10 - RRL",
            " 11 - BC single-stage",
            " 12 - BC two-stage",
            # " 13 - BC two-stage w/ intention",
    ]


def analysis(name, i, D_orig, not_use):
    
        
    count_min_distance = 0
    counter = 0
    #name = 'GT.txt'
    collision_counter = 0
    
    #for i in range(6)
    
    f = open(name)
    for line in f.readlines():
        
        s = line.split("\n")[0].split(" ")        
        if s[0] not in not_use:
            is_collision = s[1]
            # avg_distance = float(s[2])
            min_distance = float(s[3])
            if is_collision == "True":
                collision_counter+=1
                min_distance = 0.0
            count_min_distance += min_distance
            counter += 1

    # print("counter: ", counter)
    f.close
    x = count_min_distance/counter
    # print("avg min distance", x)
    if D_orig == None:
        D_orig = x 
    
    x = (-(x-D_orig)/D_orig )
    if x == -0.0:
        x = 0.0
    y = float(collision_counter/counter) * 100
    print(f"{state_names[i]:33} {x:5.2f} {y:5.1f}")
    print("---------------------------------------------")
    
    return D_orig

#  # new 0, 1, 3, 6, 9, 11

if __name__ == '__main__':

    print("run obstacle data analysis")
    
    s1 = "IR"
    s2 = "CR"
    print(f"-------------------------------- {s1} ---- {s2 } ( %) ---")
    
    # new 0, 1, 3, 6, 9, 11
    names = [
        [ "./test_result_0/result_interactive.txt",  0 ],  ##  No mask
        [ "./test_result_1/result_interactive.txt",  1 ],  ## Ground Truth
        [ "./test_result_2/result_interactive.txt",  2 ], # Random
        [ "./test_result_3/result_interactive.txt",  3 ],  ## Nearest 
        # [ "./test_result_4/result_interactive.txt",  4 ],  # autopilot
        [ "./test_result_5/result_interactive.txt",  5 ],  # RSS
        [ "./test_result_6/result_interactive.txt",  6 ],  # Kalman Filter
        [ "./test_result_7/result_interactive.txt",  7 ],  ## Social-GAN
        [ "./test_result_8/result_interactive.txt",  8 ],  # MANTRA
        [ "./test_result_9/result_interactive.txt",  9 ],  # DSA-RNN
        [ "./test_result_10/result_interactive.txt", 10 ],  ## DSA-RNN-Supervised # RRL
        [ "./test_result_11/result_interactive.txt", 11 ], # BC single-stage
        [ "./test_result_12/result_interactive.txt", 12 ],  ##  BC two-stage
        # [ "./test_result_13/result_interactive.txt", 13 ]  ##  BC two-stage w/ intention
    ]
    
    # find gt and no_mask collision_list 
    
    not_use = []
    for index in range(2):
        f = open(names[index][0])
        for line in f.readlines():
            s = line.split("\n")[0].split(" ")
            if s[0] not in not_use:
                is_collision = s[1]
                if is_collision == "True":
                    not_use.append(s[0])
    
    D_orig = None
    for n, i in names:
        D_orig = analysis(n, i, D_orig, not_use)
