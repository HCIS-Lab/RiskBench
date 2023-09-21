# kill al carla server 
killall -9 -r CarlaUE4-Linux

sleep 5

# rm result.txt
touch result.txt

SERVICE="CarlaUE4"

while read F  ; do

    # data format
    # interactive 10_t3-1_1_p_c_l_1_0 Town10HD ClearSunset mid 14252

    # spilt the string according to  " "
    array=(${F// / })  

    COUNTER=0
    while  true ; do 

        echo collect ${array[0]} ${array[1]} ${array[2]} ${array[3]} ${array[4]} ${array[5]}

        if grep -q "${array[0]}#${array[1]}#${array[2]}#${array[3]}#${array[4]}#${array[5]}" ./result.txt
        then
            break
        else
            let COUNTER=COUNTER+1

            if [ ${COUNTER} == 5 ]
            then
                killall -9 -r CarlaUE4-Linux
                sleep 5
            fi

            if pgrep "$SERVICE" >/dev/null
            then
                echo "$SERVICE is running"
                # python XXX
                python data_generator.py --scenario_type ${array[0]} --scenario_id ${array[1]} --map ${array[2]} --weather ${array[3]} --random_actors ${array[4]} --random_seed ${array[5]}
            else
                echo "$SERVICE is  stopped"
                ../../CarlaUE4.sh & sleep 10
                python data_generator.py --scenario_type ${array[0]} --scenario_id ${array[1]} --map ${array[2]} --weather ${array[3]} --random_actors ${array[4]} --random_seed ${array[5]}
            fi
        fi

    done
done <./name.txt