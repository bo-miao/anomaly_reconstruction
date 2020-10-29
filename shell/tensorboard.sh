#!/bin/bash

export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:${HOME}/lib
export CUDA_VISIBLE_DEVICES="0"
# run the tensorboard command periodly
echo ""
#echo "1 -- splitnet directory"
#echo -n "choose the directory: "
#read dir_choose

echo ""
#echo -n "input the port:"
#read port
port=4321
logdir='/home/miaobo/project/anomaly_demo2/runs'

# sleep time, hours
sleep_t=6
times=0

# while loop
while true
do
	tensorboard --logdir=${logdir} --host=10.26.1.24 --port=${port}&
	last_pid=$!
	sleep ${sleep_t}h
	kill -9 ${last_pid}
	times=`expr ${times} + 1`
	echo "Restart tensorboard ${times} times."
done

echo "tensorboard is stopped!"
