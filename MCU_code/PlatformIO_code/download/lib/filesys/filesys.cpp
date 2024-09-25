#include <algorithm>
// #include "usb_serial.h"
#include <vector>
#include "filesys.h"

LittleFS_Program myfs;

int phase = 0;
File dataFile;  // Specifes that dataFile is of File type
uint linesize = 0;
bool write_data = false;// Represents whether data should be written or not
uint32_t diskSize;
std::vector<uint> line_points;
int line_size = 0;

/// @brief A reinit function, that is called after writing is complete, so a restart between writes is not needed.
void reinit_vars() {
  int phase = 0;
  uint linesize = 0;
  bool write_data = false;// Represents whether data should be written or not  
  for (auto line_point : line_points) {
    line_point = 0;
  }
  int line_size = 0;
}

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

/// @brief Log coordinator function
void logCoordinator() {
  char serialBuffer[MAX_BUFFER_LEN];
  Serial.readBytesUntil(TERMINATE_CHAR, serialBuffer, MAX_BUFFER_LEN);
  int phases = std::atoi(serialBuffer);
  line_size += 4;
  write_int(phases);
  for(int i = 0;i < phases; i++){
    Serial.readBytesUntil(TERMINATE_CHAR, serialBuffer, MAX_BUFFER_LEN);
    int count = std::atoi(serialBuffer);
    line_size += 4;
    write_int(count);    
  } 
  int c = 0;
  int index = 0;
  for(int i =0; i < phases; i++){
    c = 0;
    index = 0;
    Serial.readBytesUntil(TERMINATE_CHAR, serialBuffer, MAX_BUFFER_LEN);
    std::vector<byte> map;
    for (int i = 0; i < MAX_BUFFER_LEN; i++) {
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
    Serial.readBytesUntil(TERMINATE_CHAR, serialBuffer, MAX_BUFFER_LEN);
    index = 0;
    c = 0;
    len = 0;
    std::vector<int> padding_pos;
    for (int i = 0; i < MAX_BUFFER_LEN; i++) {
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
  Serial.readBytesUntil(TERMINATE_CHAR, serialBuffer, MAX_BUFFER_LEN);
  int len_end_pos = 0;
  index = 0;
  c = 0;
  len = 0;
  std::vector<int> end_pos;
  for (int i = 0; i < MAX_BUFFER_LEN; i++) {
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
  Serial.readBytesUntil(TERMINATE_CHAR, serialBuffer, MAX_BUFFER_LEN);
  index = 0;
  c = 0;
  std::vector<int> zero_points;
  for (int i = 0; i < MAX_BUFFER_LEN; i++) {
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
  Serial.readBytesUntil(TERMINATE_CHAR, serialBuffer, MAX_BUFFER_LEN);
  index = 0;
  c = 0;
  std::vector<float> scales;
  for (int i = 0; i < MAX_BUFFER_LEN; i++) {
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
    char serialBuffer[MAX_BUFFER_LEN];
    Serial.readBytesUntil(TERMINATE_CHAR, serialBuffer, MAX_BUFFER_LEN);
    int len = 0;
    int index = 0;
    int count = 0;
    std::vector<byte> weights;
    for (int i = 0; i < MAX_BUFFER_LEN; i++) {
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
    Serial.readBytesUntil(TERMINATE_CHAR, serialBuffer, 50);
    int bias = std::atoi(serialBuffer);
    write_int(bias);
    // Serial.print("bias:");
    // Serial.println(serialBuffer);
    linesize += 4;  //int
    phase += 1;
  } else if (phase == 2) {  //handle which kernel
    char serialBuffer[50];
    Serial.readBytesUntil(TERMINATE_CHAR, serialBuffer, 50);
    int which = std::atoi(serialBuffer);
    write_int(which);
    // Serial.print("which kernel:");
    // Serial.println(serialBuffer);
    linesize += 4;
    phase += 1;
  } else if (phase == 3) {  //handle count
    char serialBuffer[50];
    Serial.readBytesUntil(TERMINATE_CHAR, serialBuffer, 50);
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
      Serial.readBytesUntil(TERMINATE_CHAR, serialBuffer, 50);
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
    Serial.readBytesUntil(TERMINATE_CHAR, serialBuffer, 1000);
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
    Serial.readBytesUntil(TERMINATE_CHAR, serialBuffer, MAX_BUFFER_LEN);
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

// TODO: Call this function
/// @brief Stores the coordinator lines in a file for later use
void log_coor_lines(){
  dataFile = myfs.open("coorlines.txt", FILE_WRITE);
  if (dataFile)
  {
    Serial.println("\nLogging coor_lines!!!");
    for (uint i : line_points) {
      dataFile.println(i);
      delay(100);
    }
  }
  dataFile.close();
}

/// @brief Initialize the filesystem on the flash
void setup_filesys() {
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
    Serial.printf("Error starting %s\n", "PROGRAM FLASH DISK");
    while (1) {
      // Error, so don't do anything more - stay stuck here
    }
  }
}