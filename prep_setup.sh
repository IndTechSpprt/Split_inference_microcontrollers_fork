#!/bin/bash

#Enable errors
set -e

#temp variable
temp="worker_0"
export temp

## Script to put the controller into download mode, upload weight fragments and then flash worker code
echo "flashing teensy 4.1 to put it into download mode"
cd ./MCU_code/PlatformIO_code/download
#pio run --target upload
#TODO add success check before proceeding
echo "Done, now attempting to download weights..."
sleep 5
cd ../../
#python ./write_into_mcus.py /dev/ttyACM0
echo "now flashing worker code and setting up communication..."
cd ./PlatformIO_code/worker_code
pio run --target upload
#Check if success before saying done
echo "----DONE!----"