import os
import warnings

def read_length_file(filename):
    try:
        file = open(filename,"r")
        contents = file.read()
        #TODO maybe more elegant?
        #To deal with comma at the end from rust, we remove the suffix
        contents+=";"
        return contents.removesuffix(", ;")
    except:
        return ""

if 'mcu_id' in os.environ:
    # Get ID from environment variable if set
    id = os.environ['mcu_id']
else:
    #Default ID is set to 0, remember to throw a warning
    warnings.warn("You have not specified an MCU ID, setting it to 0")
    id = "0"

#Open result lengths file as read only
result_length = read_length_file("../../../pc_code/Simulation/Simu_q/res_len_"+id+".txt")

#Open input lengths file as read only

input_length = read_length_file("../../../pc_code/Simulation/Simu_q/input_len_"+id+".txt")

if result_length == "" or input_length == "":
    warnings.warn("Using default lengths")
    input_length = "151875, 138702, 401408, 200704, 408617, 301056, 75264, 161481, 451584, 75264, 155961, 112896, 25088, 57609, 150528, 25088, 57609, 150528, 25088, 53833, 37632, 12544, 32777, 75264, 12544, 32777, 75264, 12544, 32777, 75264, 12544, 32777, 75264, 18816, 49161, 112896, 18816, 49161, 112896, 18816, 43209, 28224, 7840, 25929, 47040, 7840, 25929, 47040, 7840, 25929, 47040, 15680, 1280"
    result_length = "133804, 133804, 66903, 401409, 100353, 25089, 150529, 150529, 25089, 150529, 37633, 8364, 50177, 50177, 8364, 50177, 50177, 8364, 50177, 12545, 4183, 25089, 25089, 4183, 25089, 25089, 4183, 25089, 25089, 4183, 25089, 25089, 6273, 37633, 37633, 6273, 37633, 37633, 6273, 37633, 9409, 2615, 15681, 15681, 2615, 15681, 15681, 2615, 15681, 15681, 5228, 20908, 335"

#Open the header file to write the arrays into
lengths_h_file = open("./include/lengths.h","w")

#write input lengths
lengths_h_file.write("const int input_length[53] = {"+input_length+"};\n")
lengths_h_file.write("const int result_length[53] = {"+result_length+"};")
lengths_h_file.close()