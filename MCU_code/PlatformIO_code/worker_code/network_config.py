import os
import warnings
import json

if 'mcu_id' in os.environ:
    # Get ID from environment variable if set
    id = os.environ['mcu_id']
else:
    #Default ID is set to 0, remember to throw a warning
    warnings.warn("You have not specified an MCU ID, setting it to 0")
    id = "0"

# Convert ID to int
id_int = int(id)

# Get other parameters from ID
ip_end = str(124 - id_int)
mac_end = str(0xEB + id_int)

to_append = {
    "id": id,
    "ip_end": ip_end,
    "mac_end": mac_end
}

#Open the testbed file
if 'append_to_testbed' in os.environ:
    if os.environ['append_to_testbed'] == "Y" or os.environ['append_to_testbed'] == "y":
        if os.path.isfile('../../testbed.json'):
            with open('../../testbed.json', 'r') as file:
                try:
                    testbed = json.load(file)
                except:
                    testbed = []
        else:
            testbed = []
        with open('../../testbed.json', 'w') as file:
            testbed.append(to_append)
            json.dump(testbed,file)

print("'-DMCU_ID=%s' '-DIP_END=%s' '-DMAC_END=%s'" % (id, ip_end, mac_end))
