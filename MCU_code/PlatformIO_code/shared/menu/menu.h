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

#endif