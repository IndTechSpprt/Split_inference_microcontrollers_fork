/// This module holds all the menu functions

#include "menu.h"
#include "filesys.h"

/// @brief Function to print spaces on the display
/// @param num - number of spaces to print
void printSpaces(int num) {
  for (int i = 0; i < num; i++) {
    Serial.print(" ");
  }
}

/// @brief Recursive print directory function
/// @param dir directory to look through and print
/// @param numSpaces number of spaces to print
void printDirectory2(File dir, int numSpaces) {
  while (true) {
    File entry = dir.openNextFile();
    if (!entry) {
      //Serial.println("** no more files **");
      break;
    }
    printSpaces(numSpaces);
    Serial.print(entry.name());
    if (entry.isDirectory()) {
      Serial.println("/");
      printDirectory2(entry, numSpaces + 2);
    } else {
      // files have sizes, directories do not
      printSpaces(36 - numSpaces - strlen(entry.name()));
      Serial.print("  ");
      Serial.println(entry.size(), DEC);
    }
    entry.close();
  }
}

/// @brief Print directory function
/// @param fs filesystem reference
void printDirectory(FS& fs) {
  Serial.println("Directory\n---------");
  printDirectory2(fs.open("/"), 0);
  Serial.println();
}

/// @brief Prints the space, total space and all the files in the flash
void listFiles() {
  Serial.print("\n Space Used = ");
  Serial.println(myfs.usedSize() / 1024);
  Serial.print("Filesystem Size = ");
  Serial.println(myfs.totalSize() / 1024);

  printDirectory(myfs);
}

/// @brief Stop logging data, currently UNUSED
void stopLogging() {
  Serial.println("\nStopped Logging Data!!!");
  write_data = false;
  // Closes the data file.
  dataFile.close();
  Serial.printf("Records written = %d\n", record_count);
}

/// @brief Dumps all the logs
void dumpLog() {
  char serialBuffer[100];
  Serial.println("\nDumping Log!!!");
  Serial.println("\nPlease enter the name of the log you want to see:");
  while (!Serial.available()) {};
  Serial.readBytesUntil(TERMINATE_CHAR, serialBuffer, 100);
  // open the file.
  Serial.println(serialBuffer);
  dataFile = myfs.open(serialBuffer, FILE_READ);
  // if the file is available, write to it:
  if (dataFile) {
    while (dataFile.available()) {
      Serial.println(dataFile.read(), DEC);
      Serial.print(' ');
    }
    dataFile.close();
  }
  // if the file isn't open, pop up an error:
  else {
    Serial.println("error opening datalog,datalog not found!");
  }
}

/// @brief Erases all the files
void eraseFiles() {
  if(myfs.exists("Coordinator.bin")){
    myfs.remove("Coordinator.bin");
    myfs.remove("coor_lines.txt");
  }
  else{
    myfs.quickFormat();  // performs a quick format of the created di
  }
  Serial.println("\nFiles erased !");
}

/// @brief Menu display function
void display_menu() {
  Serial.println();
  Serial.println("Menu Options:");
  Serial.println("\tl - List files on disk");
  Serial.println("\te - Erase files on disk");
  Serial.println("\ts - Start Logging data (Restarting logger will append records to existing log)");
  Serial.println("\tc - Start Logging Coordinator data (Restarting logger will append records to existing log)");
  Serial.println("\td - Dump Log");
  Serial.println("\th - Menu");
  Serial.println();
}

/// @brief Handler function for the FS menu, allowing listing, erasing, dumping logs, and logging of data into logs. Only active when nothing is being written.
/// Menu options
/// `l` -> list files `e` -> erase files `s` -> log data `c` -> log coordinator `d` -> dump & `h` -> menu
/// @return New write type to switch to
void menu_handler(void) {
    char rr;
    rr = Serial.read();
    switch (rr) {
      case 'l': listFiles(); break;
      case 'e': eraseFiles(); break;
      case 's': {
          Serial.println("\nLogging Data!!!");
          write_data = true;  // sets flag to continue to write data until new command is received
          // opens a file or creates a file if not present,  FILE_WRITE will append data to
          // to the file created.
          String filename = "datalog.bin";
          type = Data;
          dataFile = myfs.open(filename.c_str(), FILE_WRITE);
          delay(1000);
          // logData(phase);
        }
      break;
      case 'c' : {
          Serial.println("\nLogging Coordinator!!!");
          write_data = true;  // sets flag to continue to write data until new command is received
          // opens a file or creates a file if not present,  FILE_WRITE will append data to
          // to the file created.
          String filename = "Coordinator.bin";
          type = Coordinator;
          dataFile = myfs.open(filename.c_str(), FILE_WRITE);
          delay(1000);
      }
      break;
      case 'd': dumpLog(); break;
      case '\r':
      case '\n':
      case 'h': display_menu(); break;
    }
    while (Serial.read() != -1);  // remove rest of characters.
}