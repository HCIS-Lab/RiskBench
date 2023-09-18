#!/bin/sh

echo "Input the method id you want to process"
echo "Choose from the following options:"
echo ""
echo " 0 - No mask"
echo " 1 - Ground Truth"
echo " 2 - Random"
echo " 3 - Nearest"
echo " 4 - RSS"
echo " 5 - Kalman Filter"
echo " 6 - Social-GAN"
echo " 7 - MANTRA"
echo " 8 - DSA-RNN"
echo " 9 - DSA-RNN-Supervised"
echo "10 - BC single-stage"
echo "11 - BC two-stage"
echo " "

# # whether use obstacle region
                         
read -p "Enter ID to run Planning-aware Evaluation Benchmark (obstacle): " ds_id
echo " "

read -p "Enter 1 for using obstacle region:, 0 for not using obstacle region: "  obstacle_region


rm -r ./test_result
SERVICE="CarlaUE4"

while read F  ; do
        
    array=(${F// / })  # spilt the string according to  " "
    
    while  true ; do 
        echo test ${array[0]} ${array[1]} ${array[2]}
        # ~[ grep -q ${array[0]}_${array[1]}_${array[2]} result_obstacle.txt ]
        
        if grep -q "${array[0]}#${array[1]}#${array[2]}" ./test_result/result_obstacle.txt
        then
            break
        else

            if pgrep "$SERVICE" >/dev/null
            then
                echo "$SERVICE is running"
                if [ ${obstacle_region} == 1 ]
                then
                    echo "${obstacle_region}"
                    if [ ${ds_id} == 4 ]
                    then
                        python data_generator_randomseed.py --scenario_id ${array[0]} --scenario_type obstacle --map Town10HD_opt --weather ${array[1]} --random_actors ${array[2]} --start_frame ${array[3]} --gt_x ${array[4]} --gt_y ${array[5]} --target_x ${array[6]} --target_y ${array[7]}  --replay --no_save --method ${ds_id} --save_rss --obstacle_region
                    else
                        python data_generator_randomseed.py --scenario_id ${array[0]} --scenario_type obstacle --map Town10HD_opt --weather ${array[1]} --random_actors ${array[2]} --start_frame ${array[3]} --gt_x ${array[4]} --gt_y ${array[5]} --target_x ${array[6]} --target_y ${array[7]}  --replay --no_save --method ${ds_id} --obstacle_region
                    fi
                else
                    if [ ${ds_id} == 4 ]
                    then
                        python data_generator_randomseed.py --scenario_id ${array[0]} --scenario_type obstacle --map Town10HD_opt --weather ${array[1]} --random_actors ${array[2]} --start_frame ${array[3]} --gt_x ${array[4]} --gt_y ${array[5]} --target_x ${array[6]} --target_y ${array[7]}  --replay --no_save --method ${ds_id} -save_rss
                    else
                        python data_generator_randomseed.py --scenario_id ${array[0]} --scenario_type obstacle --map Town10HD_opt --weather ${array[1]} --random_actors ${array[2]} --start_frame ${array[3]} --gt_x ${array[4]} --gt_y ${array[5]} --target_x ${array[6]} --target_y ${array[7]}  --replay --no_save --method ${ds_id}
                    fi
                fi
            else
                echo "$SERVICE is  stopped"
                ../../CarlaUE4.sh & sleep 10
            fi
        fi
    done
done <./names_obstacle
mkdir ./results
mv ./test_result ./results/test_result_${ds_id}_${obstacle_region}



