#!/bin/bash
# This shell is going to create the video for each scenario_ID

# kill al carla server 
killall -9 -r CarlaUE4-Linux

echo "Input the scenario_type you want to process"
echo "Choose from the following options:"


echo "1 - interactive"
echo "2 - non-interactive"
echo "3 - obstacle"
echo "4 - collision"

ds_id=1
read -p "Enter scenario type ID to create a data video: " ds_id
scenario_type="interactive"
if [ ${ds_id} == 1 ]
then
    scenario_type="interactive"
elif [ ${ds_id} == 2 ]
then
    scenario_type="non-interactive"
elif [ ${ds_id} == 3 ]
then
    scenario_type="obstacle"
elif [ ${ds_id} == 4 ]
then
    scenario_type="collision"
else
    echo "Invalid ID!!!"
    echo "run default setting : interactive"
fi

len=${#scenario_type}
len=$((len + 19))
folder=`ls -d ./data_collection/${scenario_type}/*`

../../CarlaUE4.sh &
sleep 15
SERVICE="CarlaUE4"
for eachfile in $folder
do
    if pgrep "$SERVICE" >/dev/null
    then
        echo "$SERVICE is running"
    else
        echo "$SERVICE is  stopped"
        ../../CarlaUE4.sh & sleep 15	
    fi
    
    echo ${eachfile:$len} 



    if [ `echo ${eachfile:$len:1} | awk -v tem="B" '{print($1==tem)? "1":"0"}'` -eq "1" ]
    then
        # B3, B7, B8
        python data_generator.py --scenario_type ${scenario_type} --scenario_id ${eachfile:$len} --map ${eachfile:$len:2} --test  --no_save --generate_random_seed --weather ${weather[${w[${i}]}]}  --random_actors ${random_actor[j]} 
    else

        if [ `echo ${eachfile:$len:1} | awk -v tem="A" '{print($1==tem)? "1":"0"}'` -eq "1" ]
        then
            # A0, A1, A6
            python data_generator.py --scenario_type ${scenario_type} --scenario_id ${eachfile:$len} --map ${eachfile:$len:2} --test --no_save
        else
            # Carla original Town XX
            if [ `echo ${eachfile:$len:2} | awk -v tem="10" '{print($1==tem)? "1":"0"}'` -eq "1" ]
            then
                python data_generator.py --scenario_type ${scenario_type} --scenario_id ${eachfile:$len} --map Town10HD --test --no_save
            else
                python data_generator.py --scenario_type ${scenario_type} --scenario_id ${eachfile:$len} --map Town0${eachfile:$len:1} --test --no_save
            fi
        fi
    fi 

    sleep 3
done

killall -9 -r CarlaUE4-Linux
