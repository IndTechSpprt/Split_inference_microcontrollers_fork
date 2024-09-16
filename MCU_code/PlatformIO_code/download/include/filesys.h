#include <algorithm>
// #include "usb_serial.h"
#include <vector>
/*
  adapted from Teensy4 arduino examples: LittleFS  datalogger
 
 This code is used to write the weights into MCU


 */
#include <LittleFS.h>
LittleFS_Program myfs;

// NOTE: This option is only available on the Teensy 4.0, Teensy 4.1 and Teensy Micromod boards.
// With the additonal option for security on the T4 the maximum flash available for a
// program disk with LittleFS is 960 blocks of 1024 bytes
#define PROG_FLASH_SIZE 1024 * 1024 * 4  // Specify size to use of onboard Teensy Program Flash chip \
                                         // This creates a LittleFS drive in Teensy PCB FLash.
char terminateChar = '!';
const int bufferLength = 300000;
int phase = 0;
File dataFile;  // Specifes that dataFile is of File type
int record_count = 0;
uint linesize = 0;
bool write_data = false;
uint32_t diskSize;
std::vector<uint> line_points;
int line_size = 0;
void write_vector_byte(std::vector<byte>& weights) {
  if (dataFile) {
    char buffer[weights.size()];
    for (size_t i = 0; i < weights.size(); ++i) {
      buffer[i] = static_cast<char>(weights[i]);
    }
    dataFile.write(buffer, weights.size());
  }
}
void write_int(int& number) {
  if (dataFile) {
    char byteArray[sizeof(int)];
    char* bytePtr = reinterpret_cast<char*>(&number);  // Obtain a pointer to the integer's memory
    // Copy the bytes of the integer into the byte array
    for (size_t i = 0; i < sizeof(int); ++i) {
      byteArray[i] = *(bytePtr + i);
    }
    dataFile.write(byteArray, sizeof(int));
  }
}
void write_vector_int(std::vector<int>& data) {
  if (dataFile) {
    for (int d : data) {
      write_int(d);
    }
  }
}
void write_byte(byte& number) {
  if (dataFile) {
    char byteArray[1];
    byteArray[0] = static_cast<char>(number);
    dataFile.write(byteArray, sizeof(byte));
  }
}
void write_float(float& data) {
  if (dataFile) {
    char byteArray[sizeof(float)];
    char* bytePtr = reinterpret_cast<char*>(&data);  // Obtain a pointer to the integer's memory
    // Copy the bytes of the integer into the byte array
    for (size_t i = 0; i < sizeof(float); ++i) {
      byteArray[i] = *(bytePtr + i);
    }
    dataFile.write(byteArray, sizeof(float));
  }
}
void logCoordinator() {
  // Serial.println("entering coordinator");
  char serialBuffer[bufferLength];
  Serial.readBytesUntil(terminateChar, serialBuffer, bufferLength);
  int phases = std::atoi(serialBuffer);
  line_size += 4;
  // Serial.println(phases);
  write_int(phases);
  for(int i = 0;i < phases; i++){
    Serial.readBytesUntil(terminateChar, serialBuffer, bufferLength);
    int count = std::atoi(serialBuffer);
    line_size += 4;
    write_int(count);    
  } 
  int c = 0;
  int index = 0;
  for(int i =0; i < phases; i++){
    c = 0;
    index = 0;
    Serial.readBytesUntil(terminateChar, serialBuffer, bufferLength);
    std::vector<byte> map;
    for (int i = 0; i < bufferLength; i++) {
      while (serialBuffer[index] != ' ') index++;
      char substring[index - i + 1];
      strncpy(substring, &serialBuffer[i], index - i);
      substring[index - i] = '\0';
      int temp = std::atoi(substring);
      map.push_back(static_cast<byte>(temp));
      c++;
      if (c >= 16) break;
      i = index;
      index = i + 1;
    }
    line_size += 16;
    write_vector_byte(map);    
  }
  int len = 0;
  for(int i = 0; i < phases; i++){
    Serial.readBytesUntil(terminateChar, serialBuffer, bufferLength);
    index = 0;
    c = 0;
    len = 0;
    std::vector<int> padding_pos;
    for (int i = 0; i < bufferLength; i++) {
      while (serialBuffer[index] != ' ') index++;
      char substring[index - i + 1];
      strncpy(substring, &serialBuffer[i], index - i);
      substring[index - i] = '\0';
      if (i == 0) {
        len = std::atoi(substring);
        if(len == 0) break;
      } else {
        int temp = std::atoi(substring);
        padding_pos.push_back(temp);
        c++;
        if (c >= len) break;
      }
      i = index;
      index = i + 1;
    }
    line_size += 4 + padding_pos.size() * 4;
    write_int(len);
    if(len > 0){
      write_vector_int(padding_pos);    
    }
  }
  Serial.readBytesUntil(terminateChar, serialBuffer, bufferLength);
  int len_end_pos = 0;
  index = 0;
  c = 0;
  len = 0;
  std::vector<int> end_pos;
  for (int i = 0; i < bufferLength; i++) {
    while (serialBuffer[index] != ' ') index++;
    char substring[index - i + 1];
    strncpy(substring, &serialBuffer[i], index - i);
    substring[index - i] = '\0';
    if (i == 0) {
      len = std::atoi(substring);
      if (len == 0) {
        // write_byte(static_cast<byte>(len));
        break;
      }
    } else {
      int temp = std::atoi(substring);
      end_pos.push_back(temp);
      c++;
      if (c >= len * 3) break;
    }
    i = index;
    index = i + 1;
  }
  line_size += 1;
  byte temp = static_cast<byte>(len);
  write_byte(temp);
  if (len > 0) {
    line_size += end_pos.size() * 4;
    write_vector_int(end_pos);
  }
  Serial.readBytesUntil(terminateChar, serialBuffer, bufferLength);
  index = 0;
  c = 0;
  std::vector<int> zero_points;
  for (int i = 0; i < bufferLength; i++) {
    while (serialBuffer[index] != ' ') index++;
    char substring[index - i + 1];
    strncpy(substring, &serialBuffer[i], index - i);
    substring[index - i] = '\0';
    int temp = std::atoi(substring);
    zero_points.push_back(temp);
    c++;
    if (c >= 3) break;
    i = index;
    index = i + 1;
  }
  line_size += zero_points.size() * 4;
  write_vector_int(zero_points);
  Serial.readBytesUntil(terminateChar, serialBuffer, bufferLength);
  index = 0;
  c = 0;
  std::vector<float> scales;
  for (int i = 0; i < bufferLength; i++) {
    while (serialBuffer[index] != ' ') index++;
    char substring[index - i + 1];
    strncpy(substring, &serialBuffer[i], index - i);
    substring[index - i] = '\0';
    float temp = std::atof(substring);
    scales.push_back(temp);
    c++;
    if (c >= 3) break;
    i = index;
    index = i + 1;
  }
  line_size += scales.size() * 4;
  for(float f : scales){
    write_float(f);
  }
  dataFile.close();
  while (!Serial.available()) {}  //wait for the next entry
  if (Serial.peek() == '!') {
    Serial.read();
    while (!Serial.available()) {}  //wait for the next entry
    line_points.push_back(line_size);
    if (Serial.peek() == '!') {
      Serial.read();
      write_data = false;
      Serial.println("stop writing into MCU,lines:");
      for (uint i : line_points) {
        Serial.println(i);
      }
    }
    // Serial.println("\n --line written into MCU--");
    // Serial.println(linesize);
  }
  String filename = "Coordinator.bin";
  dataFile = myfs.open(filename.c_str(), FILE_WRITE);
}
void logData(int& phase) {
  if (phase == 0) {  // handle weights
    char serialBuffer[bufferLength];
    Serial.readBytesUntil(terminateChar, serialBuffer, bufferLength);
    int len = 0;
    int index = 0;
    int count = 0;
    std::vector<byte> weights;
    for (int i = 0; i < bufferLength; i++) {
      while (serialBuffer[index] != ' ') index++;
      char substring[index - i + 1];
      strncpy(substring, &serialBuffer[i], index - i);
      substring[index - i] = '\0';
      if (i == 0) {
        len = std::atoi(substring);
      } else {
        int temp = std::atoi(substring);
        weights.push_back(static_cast<byte>(temp));
        count++;
        if (count >= len) break;
      }
      i = index;
      index = i + 1;
    }
    write_int(len);
    write_vector_byte(weights);
    // Serial.print("weights:");
    // Serial.println(serialBuffer);
    linesize += weights.size() + 4;  //int and vector<byte>
    phase += 1;
  } else if (phase == 1) {  //handle bias
    char serialBuffer[50];
    Serial.readBytesUntil(terminateChar, serialBuffer, 50);
    int bias = std::atoi(serialBuffer);
    write_int(bias);
    // Serial.print("bias:");
    // Serial.println(serialBuffer);
    linesize += 4;  //int
    phase += 1;
  } else if (phase == 2) {  //handle which kernel
    char serialBuffer[50];
    Serial.readBytesUntil(terminateChar, serialBuffer, 50);
    int which = std::atoi(serialBuffer);
    write_int(which);
    // Serial.print("which kernel:");
    // Serial.println(serialBuffer);
    linesize += 4;
    phase += 1;
  } else if (phase == 3) {  //handle count
    char serialBuffer[50];
    Serial.readBytesUntil(terminateChar, serialBuffer, 50);
    int count = std::atoi(serialBuffer);
    write_int(count);
    // Serial.print("count:");
    // Serial.println(serialBuffer);
    linesize += 4;
    phase += 1;
  } else if (phase == 4) {  //handle start pos
    if (Serial.peek() == '!') {
      // Serial.println("--skipping start pos for Linear layer--");
      Serial.read();
      phase += 1;
    } else {
      char serialBuffer[50];
      Serial.readBytesUntil(terminateChar, serialBuffer, 50);
      int index = 0;
      int count = 0;
      std::vector<int> data;
      for (int i = 0; i < 50; i++) {
        while (serialBuffer[index] != ' ') index++;
        char substring[index - i + 1];
        strncpy(substring, &serialBuffer[i], index - i);
        substring[index - i] = '\0';
        int temp = std::atoi(substring);
        data.push_back(temp);
        count++;
        if (count >= 3) break;
        i = index;
        index = i + 1;
      }
      write_vector_int(data);
      // Serial.print("start_pos:");
      // Serial.println(serialBuffer);
      linesize += data.size() * 4;
      phase += 1;
    }
  } else if (phase == 5) {  //type info
    char serialBuffer[1000];
    byte type = 0;
    Serial.readBytesUntil(terminateChar, serialBuffer, 1000);
    if (serialBuffer[0] == 'C') {  //convolution
      type = 0;
    } else if (serialBuffer[0] == 'L') {  //Linear
      type = 1;
    }
    int index = 2;
    int count = 0;
    int o_pg, i_pg;
    std::vector<int> s, k, in, o;
    byte b_in;
    int c_in;
    byte b_out;
    int c_out;
    for (int i = 2; i < 1000; i++) {
      while (serialBuffer[index] != ' ') index++;
      char substring[index - i + 1];
      strncpy(substring, &serialBuffer[i], index - i);
      substring[index - i] = '\0';
      if (type == 0) {
        int temp = std::atoi(substring);
        if (count == 0) {
          o_pg = temp;
          count++;
        } else if (count == 1) {
          i_pg = temp;
          count++;
        } else if (count == 2) {
          s.push_back(temp);
          if (s.size() == 2) { count++; }
        } else if (count == 3) {
          k.push_back(temp);
          if (k.size() == 2) { count++; }
        } else if (count == 4) {
          in.push_back(temp);
          if (in.size() == 3) { count++; }
        } else if (count == 5) {
          o.push_back(temp);
          if (o.size() == 3) { break; }
        }
      } else if (type == 1) {
        int temp = std::atoi(substring);
        if (count == 0) {
          b_in = static_cast<byte>(temp);
          count++;
        } else if (count == 1) {
          c_in = temp;
          count++;
        } else if (count == 2) {
          b_out = static_cast<byte>(temp);
          count++;
        } else if (count == 3) {
          c_out = temp;
          break;
        }
      }
      i = index;
      index = i + 1;
    }
    phase += 1;
    if (type == 0) {
      byte temp = 0;
      write_byte(temp);
      write_int(o_pg);
      write_int(i_pg);
      write_vector_int(s);
      write_vector_int(k);
      write_vector_int(in);
      write_vector_int(o);
      linesize += 8 + (s.size() + k.size() + in.size() + o.size()) * 4 + 1;
    } else if (type == 1) {
      byte temp = 1;
      write_byte(temp);
      write_byte(b_in);
      write_int(c_in);
      write_byte(b_out);
      write_int(c_out);
      linesize += 1 + 4 + 1 + 4 + 1;
    }
    // Serial.print("info:");
    // Serial.println(serialBuffer);
  } else if (phase == 6) {  //zero points,m,s
    std::vector<byte> zero_points;
    float m, s_out;
    char serialBuffer[200];
    Serial.readBytesUntil(terminateChar, serialBuffer, bufferLength);
    int index = 0;
    int count = 0;
    for (int i = 0; i < 200; i++) {
      while (serialBuffer[index] != ' ') index++;
      char substring[index - i + 1];
      strncpy(substring, &serialBuffer[i], index - i);
      substring[index - i] = '\0';
      if (count == 0) {
        zero_points.push_back(std::atoi(substring));
        if (zero_points.size() == 3) { count++; }
      } else if (count == 1) {
        m = std::atof(substring);
        count++;
      } else if (count == 2) {
        s_out = std::atof(substring);
        break;
      }
      i = index;
      index = i + 1;
    }
    phase = 0;
    write_vector_byte(zero_points);
    write_float(m);
    write_float(s_out);
    linesize += 8 + zero_points.size();
    // Serial.println(s_out, 6);
    dataFile.close();
    while (!Serial.available()) {}  //wait for the next entry
    if (Serial.peek() == '!') {
      Serial.read();
      while (!Serial.available()) {}  //wait for the next entry
      line_points.push_back(linesize);
      if (Serial.peek() == '!') {
        Serial.read();
        write_data = false;
        Serial.println("stop writing into MCU,lines:");
        for (uint i : line_points) {
          Serial.println(i);
        }
      }
      // Serial.println("\n --line written into MCU--");
      // Serial.println(linesize);
    }
    String filename = "datalog.bin";
    dataFile = myfs.open(filename.c_str(), FILE_WRITE);
  }
}

void stopLogging() {
  Serial.println("\nStopped Logging Data!!!");
  write_data = false;
  // Closes the data file.
  dataFile.close();
  Serial.printf("Records written = %d\n", record_count);
}


void dumpLog() {
  char serialBuffer[100];
  Serial.println("\nDumping Log!!!");
  Serial.println("\nPlease enter the name of the log you want to see:");
  while (!Serial.available()) {};
  Serial.readBytesUntil(terminateChar, serialBuffer, 100);
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
void printSpaces(int num) {
  for (int i = 0; i < num; i++) {
    Serial.print(" ");
  }
}
void menu() {
  Serial.println();
  Serial.println("Menu Options:");
  Serial.println("\tl - List files on disk");
  Serial.println("\te - Erase files on disk");
  Serial.println("\ts - Start Logging data (Restarting logger will append records to existing log)");
  // Serial.println("\tx - Stop Logging data");
  Serial.println("\td - Dump Log");
  Serial.println("\th - Menu");
  Serial.println();
}
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
void printDirectory(FS& fs) {
  Serial.println("Directory\n---------");
  printDirectory2(fs.open("/"), 0);
  Serial.println();
}
void listFiles() {
  Serial.print("\n Space Used = ");
  Serial.println(myfs.usedSize() / 1024);
  Serial.print("Filesystem Size = ");
  Serial.println(myfs.totalSize() / 1024);

  printDirectory(myfs);
}

void eraseFiles() {
  if(myfs.exists("Coordinator.bin")){
    myfs.remove("Coordinator.bin");
  }
  else{
    myfs.quickFormat();  // performs a quick format of the created di
  }
  Serial.println("\nFiles erased !");
}


void setup_filesys() {

  // Open serial communications and wait for port to open:
  // Serial.begin(115200);
  // while (!Serial) {
  //   // wait for serial port to connect.
  // }
  // Serial.println("\n" __FILE__ " " __DATE__ " " __TIME__);

  // Serial.println("Initializing LittleFS ...");

// see if the Flash is present and can be initialized:
// lets check to see if the T4 is setup for security first
#if ARDUINO_TEENSY40
  if ((IOMUXC_GPR_GPR11 & 0x100) == 0x100) {
    //if security is active max disk size is 960x1024
    if (PROG_FLASH_SIZE > 960 * 1024) {
      diskSize = 960 * 1024;
      Serial.printf("Security Enables defaulted to %u bytes\n", diskSize);
    } else {
      diskSize = PROG_FLASH_SIZE;
      Serial.printf("Security Not Enabled using %u bytes\n", diskSize);
    }
  }
#else
  diskSize = PROG_FLASH_SIZE;
#endif

  // checks that the LittFS program has started with the disk size specified
  if (!myfs.begin(diskSize)) {
    // Serial.printf("Error starting %s\n", "PROGRAM FLASH DISK");
    while (1) {
      // Error, so don't do anything more - stay stuck here
    }
  }
  // Serial.println("LittleFS initialized.");

  // menu();
}

// void loop()
// {
//   if ( Serial.available() ) {
//     char rr;
//     rr = Serial.read();
//     switch (rr) {
//       case 'l': listFiles(); break;
//       case 'e': eraseFiles(); break;
//       case 's':
//         {
//           Serial.println("\nLogging Data!!!");
//           write_data = true;   // sets flag to continue to write data until new command is received
//           // opens a file or creates a file if not present,  FILE_WRITE will append data to
//           // to the file created.
//           dataFile = myfs.open("datalog.txt", FILE_WRITE);
//           logData();
//         }
//         break;
//       case 'x': stopLogging(); break;
//       case 'd': dumpLog(); break;
//       case '\r':
//       case '\n':
//       case 'h': menu(); break;
//     }
//     while (Serial.read() != -1) ; // remove rest of characters.
//   }

//   if(write_data) logData();
// }
