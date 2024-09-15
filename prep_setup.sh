#!/bin/bash

#Enable errors
set -e

## Script to put the controller into download mode, upload weight fragments and then flash worker code
echo "flashing teensy 4.1 to put it into download mode"
cd ./MCU_code/PlatformIO_code/download
pio run --target upload
echo "Done, now attempting to download weights"
cd ../../../
python ./MCU_code/write_into_mcus.py /dev/ttyUSB0
echo "now flashing worker code and setting up communication..."
cd ./MCU_code/PlatformIO_code/worker_code
####TODO some form of if/define style dynamic selection of IP and ID
echo "----DONE!----"