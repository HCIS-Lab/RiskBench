# kill al carla server 
killall -9 -r CarlaUE4-Linux
# rm ./result.txt

touch result.txt

SERVICE="CarlaUE4"

echo "Which scenario you want to process"
echo "Choose from the following options:"
echo ""
echo " 1 - interactive"
echo " 2 - obstacle"
echo " 3 - obstacle region"
echo ""
read -p "Enter scenario type: " scenario_id

echo ""
echo "Input the method id you want to process"
echo "Choose from the following options:"
echo ""
echo "  1 - Full Observation"
echo "  2 - Ground Truth"
echo "  3 - Random"
echo "  4 - Range"
echo "  5 - KalmanFilter"
echo "  6 - Social-GAN"
echo "  7 - MANTRA"
echo "  8 - QCNet"
echo "  9 - DSA"
echo " 10 - RRL"
echo " 11 - BP"
echo " 12 - BCP"
echo " 13 - AUTO"
echo " 14 - BCP Smoothing"
echo " 15 - RRL Smoothing"
echo " 16 - DSA Smoothing"
echo " 17 - BP Smoothing"
echo " "

read -p "Enter ID to run Planning-aware Evaluation Benchmark: " ds_id
echo " "

if [ ${scenario_id} == 1 ]
then
    scenario="interactive"
elif [ ${scenario_id} == 2 ]
then
    scenario="obstacle"
elif [ ${scenario_id} == 3 ]
then
    scenario="obstacle"
fi

if [ ${ds_id} == 1 ]
then
    mode="Full_Observation"
elif [ ${ds_id} == 2 ]
then
    mode="Ground_Truth"
elif [ ${ds_id} == 3 ]
then
    mode="Random"
elif [ ${ds_id} == 4 ]
then
    mode="Range"
elif [ ${ds_id} == 5 ]
then
    mode="Kalman_Filter"
elif [ ${ds_id} == 6 ]
then
    mode="Social-GAN"
elif [ ${ds_id} == 7 ]
then
    mode="MANTRA"
elif [ ${ds_id} == 8 ]
then
    mode="QCNet"
elif [ ${ds_id} == 9 ]
then
    mode="DSA"
elif [ ${ds_id} == 10 ]
then
    mode="RRL"
elif [ ${ds_id} == 11 ]
then
    mode="BP"
elif [ ${ds_id} == 12 ]
then
    mode="BCP"
elif [ ${ds_id} == 13 ]
then
    mode="AUTO"
elif [ ${ds_id} == 14 ]
then
    mode="BCP_smoothing"
elif [ ${ds_id} == 15 ]
then
    mode="RRL_smoothing"
elif [ ${ds_id} == 16 ]
then
    mode="DSA_smoothing"
elif [ ${ds_id} == 17 ]
then
    mode="BP_smoothing"
fi

while read F  ; do

    array=(${F// / })  

    COUNTER=0
    while  true ; do 
        echo inference ${array[0]} ${array[1]} ${array[2]} ${array[3]} ${array[4]} ${array[5]}
        if grep -q "${array[0]}#${array[1]}#${array[2]}#${array[3]}#${array[4]}#${array[5]}" ./result.txt
        then
            break
        else
            let COUNTER=COUNTER+1
            if [ ${COUNTER} == 20 ]
            then
                killall -9 -r CarlaUE4-Linux
                sleep 10
            fi

            # check if carla be alive
            if pgrep "$SERVICE" >/dev/null
            then
                echo "$SERVICE is running"
            else
                echo "$SERVICE is  stopped"
                ../../CarlaUE4.sh & sleep 10
            fi
            if [ ${scenario_id} == 3 ]
            then
                python data_generator.py --scenario_type ${array[0]} --scenario_id ${array[1]} --map ${array[2]} --weather ${array[3]} --random_actors ${array[4]} --random_seed ${array[5]} --inference --mode $mode --obstacle_region
            else
                python data_generator.py --scenario_type ${array[0]} --scenario_id ${array[1]} --map ${array[2]} --weather ${array[3]} --random_actors ${array[4]} --random_seed ${array[5]} --inference --mode $mode
            fi
        
        fi
    done
done <./${scenario}_name.txt

if [ ${scenario_id} == 3 ]
then
    mv ./result.txt ./${scenario}_region_results/$mode.txt
else
    mv ./result.txt ./${scenario}_results/$mode.txt
fi