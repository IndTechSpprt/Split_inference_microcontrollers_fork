#ifndef READ_H
#define READ_H
#include <iterator>
#include <vector>
#include "worker_struct.h"
#include "coordinator_struct.h"
#include "filesys.h"
int input_length[53] = {38307, 33433, 100352, 50176, 96914, 75264, 18816, 40251, 112896, 18816, 37663,28224, 6272, 15284, 37632, 6272, 15284, 37632, 6272, 13527, 9408, 3136, 9696, 18816, 3136, 9696, 18816, 3136, 9696, 18816, 3136, 9696, 18816, 4704, 14520, 28224, 4704, 14520, 28224, 4704, 14495, 9216, 2560, 10758, 15360, 2560, 10758, 15360,2560, 10758, 15360, 5120,1280};
int result_length[53] = {31159, 31159, 15580, 93448, 23383, 5860, 35047, 35047, 5860, 35047, 8776, 1972,11692, 11692, 1972, 11692, 11692, 1972, 11692, 2944, 1000, 5860, 5860, 1000, 5860, 5860, 1000, 5860, 5860, 1000, 5860, 5860, 1486, 8776, 8776, 1486, 8776, 8776, 1486, 8776, 2863, 811, 4780, 4780, 811, 4780, 4780, 811, 4780, 4780, 1594, 6373,325};
int coor_lines[] = {925, 9862, 10371, 10424, 23993, 24046, 24099, 44868, 44921, 44974, 55255, 55308, 55361, 69738, 69791, 69844, 84221, 84274, 84327, 91324, 91377, 91430, 106811,106864, 106917, 122298, 122351, 122404, 137785, 137838, 137891, 153272, 153325,153378, 176455, 176508, 176561, 199638, 199691, 199744, 222929, 222982, 223035,246996, 247049, 247102, 271063, 271116, 271169, 295130, 295183, 295236};
int lines[] = {1150, 2120, 2720, 5840, 8750, 10222, 15262, 19627, 21483, 26523, 30888, 33440,40640, 46460, 49540, 56740, 62560, 65640, 72840, 78757, 84637, 102877, 114517, 124429, 142669, 154309, 164221, 182461, 194101, 204013, 222253, 233893, 248525, 281645, 299105, 319689, 352809, 370269, 390853, 423973, 441336, 475200, 549352, 578355, 631803, 705955, 734958, 788406, 862558, 891561, 996361, 1159153, 1587503};
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