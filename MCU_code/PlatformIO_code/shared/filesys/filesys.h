#ifndef FILESYS_H
#define FILESYS_H

/*
 adapted from Teensy4 arduino examples: LittleFS_Program_Simple_Datalogger:
 https://github.com/PaulStoffregen/LittleFS/tree/main/examples/Simple_DataLogger/LittleFS_Program_Simple_Datalogger

 This code is used to write the weights generated into MCUs
 */

#include <LittleFS.h>
extern LittleFS_Program myfs;

// NOTE: This option is only available on the Teensy 4.0, Teensy 4.1 and Teensy Micromod boards.
// With the additonal option for security on the T4 the maximum flash available for a
// program disk with LittleFS is 960 blocks of 1024 bytes
#define PROG_FLASH_SIZE 1024 * 1024 * 4  // Specify size to use of onboard Teensy Program Flash chip \
                                         // This creates a LittleFS drive in Teensy PCB FLash.

#define TERMINATE_CHAR '!'
#define MAX_BUFFER_LEN 300000

extern int phase;
extern File dataFile;  // Specifes that dataFile is of File type
extern int record_count;
extern uint linesize;
extern bool write_data;// Represents whether data should be written or not
extern uint32_t diskSize;
extern std::vector<uint> line_points;
extern int line_size;

extern void write_vector_byte(std::vector<byte>& weights);
extern void write_int(int& number);
extern void write_vector_int(std::vector<int>& data);
extern void write_byte(byte& number);
extern void write_float(float& data);
extern void logCoordinator();
extern void logData(int& phase);
extern void setup_filesys();
extern void reinit_line_points();

#endif