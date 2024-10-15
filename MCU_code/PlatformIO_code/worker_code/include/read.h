#ifndef READ_H
#define READ_H
#include <iterator>
#include <vector>
#include "worker_struct.h"
#include "coordinator_struct.h"
#include "filesys.h"
#if STATIC_LENGTHS
  const int input_length[53] = { 151875, 138702, 401408, 200704, 408617, 301056, 75264, 161481, 451584, 75264, 155961, 112896, 25088, 57609, 150528, 25088, 57609, 150528, 25088, 53833, 37632, 12544, 32777, 75264, 12544, 32777, 75264, 12544, 32777, 75264, 12544, 32777, 75264, 18816, 49161, 112896, 18816, 49161, 112896, 18816, 43209, 28224, 7840, 25929, 47040, 7840, 25929, 47040, 7840, 25929, 47040, 15680, 1280 };
  const int result_length[53] = {133804, 133804, 66903, 401409, 100353, 25089, 150529, 150529, 25089, 150529, 37633, 8364, 50177, 50177, 8364, 50177, 50177, 8364, 50177, 12545, 4183, 25089, 25089, 4183, 25089, 25089, 4183, 25089, 25089, 4183, 25089, 25089, 6273, 37633, 37633, 6273, 37633, 37633, 6273, 37633, 9409, 2615, 15681, 15681, 2615, 15681, 15681, 2615, 15681, 15681, 5228, 20908,334};
#else
  #include "lengths.h"
#endif
std::vector<int> lines;
std::vector<int> coor_lines;
//int lines[] = { 1265, 2332, 3052, 6484, 9685, 11341, 16829, 21582, 23670, 29158, 33911, 36463, 44263, 50568, 53648, 61448, 67753, 70833, 78633, 84938, 91098, 110706, 123219, 133603, 153211, 165724, 176108, 195716, 208229, 218613, 238221, 250734, 266310, 301822, 320543, 342455, 377967, 396688, 418600, 454112, 472833, 508689, 588297, 619434, 676026, 755634, 786771, 843363, 922971, 954108, 1066244, 1240460, 1681990 };
//int coor_lines[] = { 2753, 21962, 22015, 22068, 51449, 51502, 51555, 95692, 95745, 95798, 117851, 117904, 117957, 147902, 147955, 148008, 177953, 178006, 178059, 192896, 192949, 193002, 223915, 223968, 224021, 254934, 254987, 255040, 285953, 286006, 286059, 316972, 317025, 317078, 363351, 363404, 363457, 409730, 409783, 409836, 432297, 432350, 432403, 473528, 473581, 473634, 514759, 514812, 514865, 555990, 556043, 556096 };
bool reading_weight = true; 
int prev_endpos = 0;
const int LINEAR_SEGMENT  = 300;
int littleEndianToInt(const char* bytes) {
  // Interpret the reversed buffer as an integer using pointer
  int result;
  memcpy(&result, bytes, sizeof(int));
  return result;
}
float littleEndianToFloat(const char* bytes) {
  // Interpret the reversed buffer as a float using pointer
  float result;
  memcpy(&result, bytes, sizeof(float));
  return result;
}
byte read_byte(int& count) {
  byte res = 0;
  if (dataFile) {
    res = static_cast<byte>(dataFile.read());
    count += 1;
  }
  return res;
}
int read_int(int& count) {
  int res = 0;
  char buffer[4];
  if (dataFile) {
    dataFile.readBytes(buffer, 4);
    res = littleEndianToInt(buffer);
    count += 4;
  }
  return res;
}
float read_float(int& count) {
  float res = 0;
  char buffer[4];
  if (dataFile) {
    dataFile.readBytes(buffer, 4);
    res = littleEndianToFloat(buffer);
    count += 4;
  }
  return res;
}
std::vector<Weight> get_weights(int line_number,int& prev_endpos) {
  // Serial.println(prev_endpos);
  std::vector<Weight> res;
  char filename[20] = "datalog.bin";
  dataFile = myfs.open(filename, FILE_READ);
  if (dataFile) {
    reading_weight = true;
    int start = line_number == 0 ? 0 : lines[line_number - 1];
    if(prev_endpos != 0) {start = prev_endpos;}
    dataFile.seek(start);
    int count = start;
    int weight_size = 0;
    int weight_count = 0;
    while (count < lines[line_number]) {
      if (count == start) {
        Weight a;
        weight_size = read_int(count);
        for (int i = 0; i < weight_size; i++) {
          a.data.push_back(read_byte(count));
        }
        a.bias = read_int(count);
        a.which_kernel = read_int(count);
        a.count = read_int(count);
        if (line_number != 52) {  //handel linear may change later
          for (int i = 0; i < 3; i++) {
            a.start_pos_in[i] = read_int(count);
          }
        }
        if (read_byte(count) == 0) {
          a.i.type = Type::Convolution;
          a.i.c_info.o_pg = read_int(count);
          a.i.c_info.i_pg = read_int(count);
          for (int i = 0; i < 2; i++) {
            a.i.c_info.s[i] = (read_int(count));
          }
          for (int i = 0; i < 2; i++) {
            a.i.c_info.k[i] = (read_int(count));
          }
          for (int i = 0; i < 3; i++) {
            a.i.c_info.i[i] = (read_int(count));
          }
          for (int i = 0; i < 3; i++) {
            a.i.c_info.o[i] = (read_int(count));
          }
        } else {
          a.i.type = Type::Linear;
          a.i.l_info.b_in = read_byte(count);
          a.i.l_info.c_in = read_int(count);
          a.i.l_info.b_out = read_byte(count);
          a.i.l_info.c_out = read_int(count);
        }
        for (int i = 0; i < 3; i++) {
          a.zero_points[i] = (read_byte(count));
        }
        a.m = read_float(count);
        a.s_out = read_float(count);
        res.push_back(a);
        start = count;
        weight_count += 1;
        if(a.i.type == Type::Linear){
          if(count == lines[line_number]){
            dataFile.close();
            reading_weight = false;
            prev_endpos = 0;
            return res;
          }
          if(weight_count >= LINEAR_SEGMENT) {
            prev_endpos = count;
            return res;
          }
        }
      } else {
        Serial.println("data not alligned! please check the code");
        break;
      }
    }
    dataFile.close();
    reading_weight = false;
  } else (Serial.println("weights data not available!"));

  return res;
}
Mapping get_mapping(int line_number) {
  // char filename[20] = "Coordinator.bin";
  dataFile = myfs.open("Coordinator.bin", FILE_READ);
  Mapping mapping;
  if (dataFile) {  

    int start = line_number == 0 ? 0 : coor_lines[line_number - 1];
    dataFile.seek(start);
    int count = start;
    int phase = 0;
    phase = read_int(count);
    for (int i = 0; i < phase; i++) {
      mapping.count.push_back(read_int(count));
    }
    for (int i = 0; i < phase; i++) {
      std::vector<byte> temp;
      for (int j = 0; j < 16; j++) {
        temp.push_back(read_byte(count));
      }
      mapping.map.push_back(temp);
    }
    for (int i = 0; i < phase; i++) {
      int len = read_int(count);
      std::vector<int> temp(len);
      // temp.reserve(len * 4);
      for (int j = 0; j < len; j++) {
        temp[j] = (read_int(count));
      }
      mapping.padding_pos.push_back(temp);
    }
    byte len = read_byte(count);
    if (len > 0) {
      for (int i = 0; i < len; i++) {
        std::vector<int> temp;
        for (int j = 0; j < 3; j++) {
          temp.push_back(read_int(count));
        }
        mapping.end_pos.push_back(temp);
      }
    }
    for (int i = 0; i < 3; i++) {
      mapping.zero_point[i] = read_int(count);
    }
    for (int i = 0; i < 3; i++) {
      mapping.scale[i] = read_float(count);
    }
    if(count != coor_lines[line_number]){
      Serial.println("data not alligned!");
    }
    dataFile.close();
  } else {
    Serial.println("Coordinator data not available!");
  }
  return mapping;
}
std::vector<byte> decode_u128(const std::vector<byte>& input) {
    std::vector<byte> next_mcus;
    unsigned int offset = 0;

    for (const auto& t : input) {
        for (int i = 0; i < 8; ++i) {
            if ((t >> i) & 0b1) {
                next_mcus.push_back(offset + i);
            }
        }
        offset += 8;
    }
    return next_mcus;
}
void otf(byte* overflow,int size){
  char filename[20] = "overflow.bin";
  if(myfs.exists(filename)){
    myfs.remove(filename);
  }
  dataFile = myfs.open(filename,FILE_WRITE);
  for(int i = 0; i < size; i++){
    write_byte(overflow[i]);
  }
  dataFile.close();
}
void handle_residual(byte* input,int size,int layer_id,std::vector<std::vector<int>>& connections,std::vector<std::vector<byte>>& zps,std::vector<std::vector<float>>& scales){
  if(!connections.empty()){
    if(connections[0][0] - 1 == layer_id){
      char filename[20] = "residual.bin";
      if(myfs.exists(filename)){
        myfs.remove(filename);
      }
      dataFile = myfs.open(filename,FILE_WRITE);
      for(int i = 0; i < size; i++){
        write_byte(input[i]);
      }
      dataFile.close();
    }
    else if (connections[0][1] - 1 == layer_id){
      dataFile = myfs.open("residual.bin",FILE_READ);
      int count = 0;
      for(int i = 0; i < size; i++){
        int re = read_byte(count);
        int temp = int (round((float(input[i] - zps[0][1]) * scales[0][1] + float(re - zps[0][0]) * scales[0][0]) / float (scales[0][2]) + float (zps[0][2])));
        if(temp > 255) input[i] = 255;
        else if(temp < 0) input[i] = 0;
        else{
          input[i] = temp;
        }
      }
      connections.erase(connections.begin());
      zps.erase(zps.begin());
      scales.erase(scales.begin());
      dataFile.close();
      if(!connections.empty() && connections[0][0] - 1 == layer_id){
        char filename[20] = "residual.bin";
        if(myfs.exists(filename)){
          myfs.remove(filename);
        }
        dataFile = myfs.open(filename,FILE_WRITE);
        for(int i = 0; i < size; i++){
          write_byte(input[i]);
        }
        dataFile.close();
      }
    }
  }
}
#endif