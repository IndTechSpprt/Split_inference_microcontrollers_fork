#ifndef MENU_H
#define MENU_H

/// @brief Write Types that are supported by the system
enum WriteTypes {
  Stop        = 0, // Stop writing
  Data        = 1, // Write Data
  Coordinator = 2, // Write Coordinator
  Lengths     = 3  // Write Lengths
};

/// @brief Menu handler function declaration
void menu_handler(void);

extern WriteTypes type; //Current write type

#ifdef PROFILING
extern unsigned int inference_time_layer_wise[53];
extern unsigned int wait_layer_wise[53];
extern unsigned int inference_time;
extern unsigned int wait_total;
#define MAX_RAM_USAGE_SAMPLES 1200
extern volatile unsigned int ram_usage[MAX_RAM_USAGE_SAMPLES];
extern unsigned ram_usage_by_layer[53];
#endif

#endif