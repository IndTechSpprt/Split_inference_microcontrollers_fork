import os

# Get ID from environment variable
id = os.environ['mcu_id']

# Convert ID to int
id_int = int(id)

# Get other parameters from ID
ip_end = str(124 - id_int)
mac_end = str(0xEB + id)

print("'-DMCU_ID=%s' '-DIP_END=%s' '-DMAC_END=%s'" % id % ip_end % mac_end)
