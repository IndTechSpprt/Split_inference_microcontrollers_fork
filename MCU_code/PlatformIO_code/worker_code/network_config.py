import os
import warnings

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

print("'-DMCU_ID=%s' '-DIP_END=%s' '-DMAC_END=%s'" % (id, ip_end, mac_end))
