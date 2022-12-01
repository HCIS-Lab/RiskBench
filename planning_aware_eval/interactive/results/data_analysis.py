
def analysis(name):
    count_min_distance = 0
    counter = 0
    #name = 'GT.txt'
    collision_counter = 0
    f = open(name)
    for line in f.readlines():
        
        s = line.split("\n")[0].split(" ")
        #print(s)

        is_collision = s[1]
        avg_distance = float(s[2])
        min_distance = float(s[3])
        if is_collision == "True":
            collision_counter+=1
            min_distance = 0.0


        counter += 1
       
        count_min_distance += min_distance

    print(counter)
    f.close
    x = count_min_distance/counter
    
    
    print("you need to modify orig_D according to No mask Result")
    orig_D = 6.2384
    x = -(x-orig_D)/orig_D
    
    y = float(collision_counter/counter)
    name_ = name.split(".")[0]
    print(f"{name_:12} {x:.4f} {y:5.4f}")
    print("------------------------------")

    
if __name__ == '__main__':

    print("run interactive data analysis")

    names = [
        "./test_result_0/result_interactive.txt", 
        "./test_result_1/result_interactive.txt", 
        "./test_result_2/result_interactive.txt", 
        "./test_result_3/result_interactive.txt", 
        "./test_result_4/result_interactive.txt", 
        "./test_result_5/result_interactive.txt", 
        "./test_result_6/result_interactive.txt", 
        "./test_result_7/result_interactive.txt", 
        "./test_result_8/result_interactive.txt", 
        "./test_result_9/result_interactive.txt", 
        "./test_result_10/result_interactive.txt", 
        "./test_result_11/result_interactive.txt"
    ]

    for n in names:
        analysis(n)

