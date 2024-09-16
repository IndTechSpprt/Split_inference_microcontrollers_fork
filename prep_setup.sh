#!/bin/bash

#Enable errors
set -e

echo "This script will configure your teensy for inference"

read -p "Please specify the role of the device to be configured: " mcu_role

read -p "Please specify the id of the device to be configured: " mcu_id

export mcu_role
export mcu_id

read -p "Is the device connected to the PC? (Y/N)" ynconn
case $ynconn in 
    [yY]*);;
    *) echo "Check connection and try again" 
        exit;;
esac

echo "configuring teensy 4.1 as $mcu_role $mcu_id$"

cd ./MCU_code/PlatformIO_code/download
pio run --target upload
echo "Waiting for COM connection"
sleep 5 &
wait
echo "Attempting to download weights..."
cd ../../
python ./write_into_mcus.py /dev/ttyACM0
echo "now flashing worker code and setting up communication..."
cd ./PlatformIO_code/worker_code
pio run --target upload
sleep 5 &
wait
echo "----DONE!----"
#echo "checking if the configuration was successful"
#TODO dynamic
#ping -c 5 169.254.71.124
#if [$? -eq 0]; then
#    echo "----DONE!----"
#else
#    echo "FAILED!"
#    exit
#fi