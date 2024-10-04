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

echo "configuring Teensy 4.1 as $mcu_role $mcu_id"

cd ./MCU_code/PlatformIO_code/download
pio run --target upload
echo "Waiting for COM connection"
sleep 5 &
wait

echo "Download weights..."
cd ../../
python ./write_into_mcus.py /dev/ttyACM0 c
sleep 5
python ./write_into_mcus.py /dev/ttyACM0 w

echo "Flashing worker code"
cd ./PlatformIO_code/worker_code
pio run --target upload
sleep 5 &
wait

echo "checking if the configuration was successful"
ip_last=$((124+$mcu_id))
ping -c 5 169.254.71.$ip_last
res=$?
if [[ $res -eq 0 ]]; then
    echo "----DONE setting up $mcu_role $mcu_id!----"
else
    echo "FAILED!"
    exit
fi
