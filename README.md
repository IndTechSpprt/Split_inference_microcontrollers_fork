 # File structure
 This repository contains the code for both the __PC side__ and the __MCU side__ for my master thesis. 
 ## The PC code
 ### Algorithms
 __images__: This folder contains some test images for the project.  
 
 __json_files__: This folder contains the recorded model files.  
 
 __python_file__: This folder contains the algorithms used to convert the original pytorch CNN models (in my case MobilenetV2 specifically) to JSON files, you can simply run `hook.py` to insert hooks in the model and then record the models in a JSON file.
 You can specify the number of layers you want to record in the line of `if layer_id == ?`. `draw.py` contains the code I used to produce some of my data image for my thesis paper.  

 __src__: This folder contains the Rust code I used to implement the project. In `lib.rs`, I implemented a simple yet effictive Rust representation of CNN models focusing on some basic functionalities
 like forward propagation and receptive field determination while making use of Rust's traits. In `calculation.rs`, I implemented some calculation
functions which are only used for testing. In `decode.rs`,
I implemented the function used to interpret the converted
 JSON files into Rust objects which are defined in `lib.rs`. In 
`operations.rs`, I implemented the algorithms used to distribute the models,
as well as the method I used to do convolution using a flattened 1-D vector and weights data.
In `util.rs`, I implemented some utility functions, the function name should
show the functionality clearly.

___main.rs___: This is the main test and debugging bed for the distribution algorithm.
The main function print the structure of the CNN model while the unit tests test the functionality of each individual function unit.

___test_references___: This folder contains the outputs produced by the original pytorch version.
 They are used as references in unit tests to judge the correctness of the algorithm used.

### Fused
This folder contains the model file after model fusion.

### Quantization
This folder contains the code used to perform quantization on the converted JSON model.

___python___: This folder contains the code used to perform verification on the quantized model.

___src___: This folder contains the source code used to perform quantization on the model files.
`merge.rs` contains the code used to perform layer fusion on the model file. `quant.rs`
contains the code used to perform quantization after layer fusion, models are converted from 32bits to 8bits following
https://arxiv.org/abs/1712.05877v1.

### Simulation
This folder contains the code used to perform simulation on a single PC.

___src___: `distribution.rs` contains the code used to generate the weights fragment data
which is then stored in each individual "MCU" as a JSON file. `node.rs` contains the 
codes to simulate each worker node and coordinator node. `simulation_setting.rs` contains the setting for
the simulation, including how each node communicate and work together. It is notable that when you are using a 
quantized model, you should use the function dubbed with __quant__ instead of the original function.
`util.rs` contains some auxiliary functions.

## The MCU code
### Arduino code
The folder contains the C++ arduino code for the actual implementation on a
Teensy4.1 MCU board. **You can ignore other folders except for `download` folder and `worker_code`
folder.**

___download___: This folder contains the code to download the weights fragment 
acquired through `distribution.rs` to MCU's FLASH memory using `LittleFS`.

___worker_code___: This folder contains the code used for performing one inference. You need 
to make sure that MCU are connected using ethernet through a network switch. `calculation.h` contains
the code used to perform convolution on a 1-D vector. `communication.h` contains the code for
collaboration of the MCUs.

### Lines
This folder contains some crucial and helpful information after the distribution 
of the model weights and workload. The number in the file's name indicates the id of the MCU.
`input_length` indicates the length of the current layer's input 1-D vector.
`result_length` indicates the length of the output 1-D vector of the current layer.
`lines` indicates for current layer's weights data,how many bytes should be read from FLASH memory.
`coor_lines` indicates for current layer's coordinator data, how many bytes should be read from FLASH memory.
All these 4 information can be acquired during the distribution of the models.

### PlatformIO code
All of the MCU code has also been ported to platform IO, it is possible to open them as individual projects and flash them through VSCode with the PlatformIO extension. This has been done to allow the use of a single IDE for flashing. However, it is still necessary to open a new instance of VSCode to flash.

# How to run the code
## How to run simulation
You can simply use `cargo run --package Simulation` to run simulation.
In case that you want to test other CNN models **(Not testd by me!)**, you need to follow the following 
steps.

*1. Run `hook.py` to get the converted JSON file.*

*2. You can test the functionality of the model using unit tests in `Algorithms\src\main.rs`.*

*3. Run `merge_batchnorm()` in `Quantization\merge.rs` to perform layer fusion.*

*4. Run quantization on the fused model. Run `quantize_layers_weights()` to perform quantization
on the weights, Run `quantize_layers_activation()` together with a **calibration set** to do quantization on the activations.
record all the scales and zero points during the process. Later you can use the 
recorded scales and zero points of the activations together with scale and zero points of the weights to perform quantization using `calculate_quantization()`.*

*5. Distribute the model and run the simulation using `cargo run --package Simulation`, remember to adjust the relevant parameters.*

## How to run on real hardware

*1. First run simulation to get the corresponding weights fragments. During the process, you can record the length of input and output of each layer in each MCU.*

*2. Connect the MCUs to a network switch, power up the MCUs.*

*3. Flash `download.ino` and `filesys.h` into the MCU, turn the MCU into download mode, run `write_into_mcus.py` on
PC to download the weights fragments acquired in step 1 into the MCU. Repeat for all the MCUs. During the process, record the 
data bytes length sent back to PC.*

*4. Flash `worker_code` into each MCU, connect MCU to the PC, start the inference and monitor the process by running `tcp.py` on PC.*

