#ifndef CALCULATION_H
#define CALCULATION_H
const int STACK_SIZE = 200 * 1024;
#include <algorithm>
#include <iterator>
#include <type_traits>
#include <vector>
#include "worker_struct.h"
int get_input_count(Weight& w){
  if(w.i.type == Type::Convolution){
    auto info = w.i.c_info;
    int rows = w.count / info.o[2];
    int col = w.count - rows * info.o[2];
    int in_rows = info.k[1] + (rows - 1) * info.s[1];
    int remain = info.k[1] * info.s[1] + (col - 1) * info.s[0] * info.s[1];
    if(rows == 0){
      in_rows = 0;
      remain += (info.k[1] - info.s[1]) * info.k[0] + (col - 1) * (info.k[1] - info.s[1]) * info.s[0];
    }
    if(col == 0){
      remain = 0;
    }
    return in_rows * info.i[2] + remain;
  }
  else{
    return -1;
  }
}
void distributed_computation(std::vector<Weight>& w, std::vector<byte>& input_distribution,byte result[],byte* overflow,int len) {
  if (w.size() == 0) { return {}; }
  if (w[0].i.type == Type::Convolution) { 
    int max_index = 0;
    int start_group = 10000;
    int start_kernel = 10000;
    int total_output_count = 0;
    auto info = w[0].i.c_info;
    int start_point = 0;
    int max_visited[3] = {w[0].start_pos_in[0],w[0].start_pos_in[1],w[0].start_pos_in[2]};
    bool first_row = false;
    int out_side_rows = 0;
    int in_side_rows = 0;
    std::vector<ushort> completed_group;
    int max_pos_count = 0;
    ushort prev_group = w[0].which_kernel / info.o_pg;
    int offset = 0;
    int page_size = 0;
    int pages[w.size() / info.o_pg + 2] = {0};
    for (int i = 0; i < w.size(); i++) {
      start_group = std::min(start_group, w[i].which_kernel / info.o_pg);
      start_kernel = std::min(start_kernel,w[i].which_kernel);
      total_output_count += w[i].count;
      short padded_row = w[i].start_pos_in[1] + info.k[0] / 2;
      short padded_col = w[i].start_pos_in[2] + info.k[1] / 2;
      int cur_group = w[i].start_pos_in[0] / info.i_pg;
      if (prev_group != cur_group) {
        max_pos_count = 0;
      }
      int cur_pos_count = padded_row / info.s[1] * info.o[2] + padded_col / info.s[0];
      if (cur_pos_count <= max_pos_count) {
        max_pos_count = std::max(max_pos_count, cur_pos_count + w[i].count);
      }
      if (max_pos_count >= info.o[1] * info.o[2] && completed_group.end() == std::find(completed_group.begin(), completed_group.end(), cur_group)) {
        completed_group.push_back(cur_group);
      }
      prev_group = cur_group;
    }
    int s_p[w.size()];
    {
      int sum = 0;
      for(int i = start_kernel; i < w.size() + start_kernel; i++){
        for(int j = 0; j < w.size(); j++){
          if(w[j].which_kernel == i){
            s_p[j] = sum;
            sum += w[j].count;
          }
        }
      }
    }
    for (int j = 0; j < w.size(); j++) {
      ushort cur_group = w[j].which_kernel / info.o_pg;
      if(std::find(completed_group.begin(),completed_group.end(),cur_group) == completed_group.end() && pages[cur_group - start_group] == 0){
        pages[cur_group - start_group] = get_input_count(w[j]);
        if(j + 1 < w.size() && w[j + 1].which_kernel / info.o_pg == cur_group){
          pages[cur_group - start_group] += get_input_count(w[j + 1]);
        }
      }
    }
    prev_group = w[0].which_kernel / info.o_pg;
    // Serial.println("start calculation!");

    for(int i = 0; i < w.size(); i++){
      int cur_count = 0;
      int padded_row = w[i].start_pos_in[1] + info.k[0] / 2;
      int padded_col = w[i].start_pos_in[2] + info.k[1] / 2;
      int adjustment = 0;
      if(w[i].count == 0) continue;
      short group_nr = w[i].which_kernel / info.o_pg;
      if(prev_group != group_nr){
        offset += page_size * info.i_pg;
        prev_group = group_nr;
      }
      if(std::find(completed_group.begin(),completed_group.end(),group_nr) != completed_group.end()){
        page_size = info.i[1] * info.i[2];
      }else{
        page_size = pages[group_nr - start_group];
        if(w.size() == 1){
          page_size = len / info.i_pg;
        }
      }
      // Serial.println("handle heads");
      //handle heads
      if((std::find(completed_group.begin(),completed_group.end(),group_nr) == completed_group.end() && w.size() == 2) || i == 0){
        first_row = true;
        if(info.i[2] - padded_row <= info.k[1]){
          out_side_rows = info.k[1];
        }else{
          out_side_rows = info.s[1];
        }
        adjustment = padded_col;
        in_side_rows = info.k[1] - out_side_rows;
      }
      //switch page
      if(w[i].start_pos_in[0] > max_visited[0] ||(w[i].start_pos_in[0] == max_visited[0] && w[i].start_pos_in[1] > max_visited[1]) || (w[i].start_pos_in[0] == max_visited[0] && w[i].start_pos_in[1] == max_visited[1] && w[i].start_pos_in[2] > max_visited[2]) ){
        //switch group
        if(w[i].start_pos_in[0] / info.i_pg != max_visited[0] / info.i_pg){
          int rows_to_move_down = info.k[1] - info.s[1];
          start_point = start_point + rows_to_move_down * info.i[2] + (info.i_pg - 1) * page_size;
        }else{
          //switch page within same group
          start_point = len / info.i_pg - get_input_count(w[i]);
          first_row = true;
        }
      }else{
        //chage within same completed page
        int prev_end_pos[3] = {w[i == 0? 0 : i - 1].start_pos_in[0],w[i == 0? 0 : i - 1].start_pos_in[1],w[i == 0? 0 : i - 1].start_pos_in[2]};
        int diff[3] = {prev_end_pos[0] - w[i].start_pos_in[0],prev_end_pos[1] - w[i].start_pos_in[1],prev_end_pos[2] - w[i].start_pos_in[2]};
        start_point = start_point - diff[1] * info.i[2] - diff[2];
      }
      while(w[i].count > 0){
        padded_row = w[i].start_pos_in[1] + info.k[0] / 2;
        padded_col = w[i].start_pos_in[2] + info.k[1] / 2;
        int acc = 0;
        for(int c = 0; c < info.i_pg; c++){
          int channel = c * page_size;
          for(int j = 0; j < info.k[0]; j++){
            int col = j * info.i[2];
            for(int k = 0; k < info.k[1]; k++){
              int row = k;
              int index = channel + col + row + start_point;
              int remaining = (page_size - start_point + offset) * info.i_pg;
              //special case when 2 weight unit within the same group
              if(i == 0 && w.size() == 2 && w[i + 1].which_kernel / info.o_pg == w[i].which_kernel / info.o_pg && std::find(completed_group.begin(),completed_group.end(),w[i].which_kernel / info.o_pg) == completed_group.end()){
                remaining = (page_size - get_input_count(w[1]) - start_point) * info.i_pg;
              }
              int to_complete = (info.k[1] * info.i[2] - padded_col) * info.i_pg;
              if(remaining < to_complete && !first_row){
                if(padded_row >= info.s[1]){
                  out_side_rows = info.s[1];
                }else{
                  out_side_rows = info.k[1];
                }
                in_side_rows = info.k[1] - out_side_rows;
                int empty_pos = (to_complete - remaining) / out_side_rows / info.i_pg;
                if(j > in_side_rows){
                  index -= (j - in_side_rows) * empty_pos;
                }
              }
              else if(first_row && remaining >= to_complete){
                if(j < out_side_rows){
                  index -= j * adjustment;
                }else{
                  index -= (out_side_rows - 1) * adjustment;
                }
              }else if(first_row && remaining < to_complete){
                out_side_rows = info.k[0];
                in_side_rows = 0;
                int empty_pos = (to_complete - remaining) / out_side_rows / info.i_pg;
                if(j > in_side_rows && adjustment == 0){
                  index -= (j - in_side_rows) * empty_pos;
                }
                if(j < out_side_rows){
                  index -= j * adjustment;
                }else{
                  index -= (out_side_rows - 1) * adjustment;
                }
              }

              max_index = std::max(max_index,index);
              int c_i = static_cast<int>(w[i].data[(c * info.k[0] * info.k[1] + j * info.k[1] + k)]);
              int b_i = static_cast<int>(w[i].zero_points[0]);
              int a_i = static_cast<int>(input_distribution[index]);
              int d_i = static_cast<int>(w[i].zero_points[1]);
              acc += (a_i - b_i) * (c_i - d_i);
            }
          }
        }
        acc += w[i].bias;
        acc = static_cast<int>(round(static_cast<float>(acc) * w[i].m));
        acc += static_cast<int> (w[i].zero_points[2]);
        if(acc < 0) acc = 0;
        if(acc > 255) acc = 255; 
        if(cur_count + s_p[i] >= STACK_SIZE){
          if(overflow == nullptr){
            while(1){
              Serial.println("overflow to nullptr!");
            }
          }
          overflow[cur_count + s_p[i] - STACK_SIZE] = acc;
        }
        else{
          if(result == nullptr){
            while(1){
              Serial.println("nullptr!!");
            }
          }
          result[cur_count + s_p[i]] = acc;
        }
        // Serial.println(acc);
        w[i].start_pos_in[2] += info.s[0];
        start_point += info.s[0];
        if(w[i].start_pos_in[2] + info.k[0] / 2 + info.k[0] > info.i[2]){
          w[i].start_pos_in[2] = 0 - info.k[0] / 2;
          w[i].start_pos_in[1] += info.s[1];
          start_point = start_point - info.s[0] +info.k[0] + ((info.s[1] - 1) * info.i[1]);
          if(first_row){
            start_point -= (out_side_rows - 1) * adjustment;
            first_row = false;
          }
        }
        if(w[i].start_pos_in[0] > max_visited[0] ||(w[i].start_pos_in[0] == max_visited[0] && w[i].start_pos_in[1] > max_visited[1]) || (w[i].start_pos_in[0] == max_visited[0] && w[i].start_pos_in[1] == max_visited[1] && w[i].start_pos_in[2] > max_visited[2])){
            max_visited[0] = w[i].start_pos_in[0];
            max_visited[1] = w[i].start_pos_in[1];
            max_visited[2] = w[i].start_pos_in[2];
        }
        w[i].count -= 1;
        cur_count += 1;
      }
    }
    Serial.print("Max index:");
    Serial.println(max_index);
  } else {
    int cur_count = 0;
    for(Weight i : w){
      int p = i.which_kernel;
      int bias = i.bias;
      int r = 0;
      for(int j = 0; j < i.data.size(); j++){
        r += (int(i.data[j]) - int(i.zero_points[1])) * (int(input_distribution[j]) - int(i.zero_points[0]));
      }
      r += bias;
      r = round(float(r) * i.m) + i.zero_points[2];
      if(r < 0) r = 0;
      if(r > 255) r = 255;
      result[cur_count] = (r);
      cur_count += 1;
    }
  }
}
#endif