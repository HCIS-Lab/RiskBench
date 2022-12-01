
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
    orig_D = 3.1788
    x = -(x-orig_D)/orig_D
    
    y = float(collision_counter/counter)
    name_ = name.split(".")[0]
    print(f"{name_:12} {x:.4f} {y:5.4f}")
    print("------------------------------")

    
if __name__ == '__main__':

    print("run interactive data analysis")

    names = [
        "./test_result_0_0_/result_obstacle.txt", 
        "./test_result_1_0_/result_obstacle.txt", 
        "./test_result_2_0_/result_obstacle.txt", 
        "./test_result_3_0_/result_obstacle.txt", 
        "./test_result_4_0_/result_obstacle.txt", 
        "./test_result_5_0_/result_obstacle.txt", 
        "./test_result_6_0_/result_obstacle.txt", 
        "./test_result_7_0_/result_obstacle.txt", 
        "./test_result_8_0_/result_obstacle.txt", 
        "./test_result_9_0_/result_obstacle.txt", 
        "./test_result_10_0_/result_obstacle.txt", 
        "./test_result_11_0_/result_obstacle.txt"
    ]
    
    for n in names:
        analysis(n)

    names_obstacle_region = [
        "./test_result_0_1_/result_obstacle.txt", 
        "./test_result_1_1_/result_obstacle.txt", 
        "./test_result_2_1_/result_obstacle.txt", 
        "./test_result_3_1_/result_obstacle.txt", 
        "./test_result_4_1_/result_obstacle.txt", 
        "./test_result_5_1_/result_obstacle.txt", 
        "./test_result_6_1_/result_obstacle.txt", 
        "./test_result_7_1_/result_obstacle.txt", 
        "./test_result_8_1_/result_obstacle.txt", 
        "./test_result_9_1_/result_obstacle.txt", 
        "./test_result_10_1_/result_obstacle.txt", 
        "./test_result_11_1_/result_obstacle.txt"
    ]
    


    for n in names_obstacle_region:
        analysis(n)

    

