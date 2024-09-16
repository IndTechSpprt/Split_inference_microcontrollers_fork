
#ifndef WORKER_STRUCT_H
#define WORKER_STRUCT_H
#include<vector>
struct conv_info {
    int o_pg;
    int i_pg;
    int s[2];
    int k[2];
    int i[3];
    int o[3];
};
struct linear_info{
  byte b_in;
  int c_in;
  byte b_out;
  int c_out;
};
enum Type {
        Convolution,
        Linear
};
struct info {
    Type type;
    conv_info c_info;
    linear_info l_info;
};
struct Weight{
  std::vector<byte> data;
  int bias;
  int which_kernel;
  int count;
  int start_pos_in[3];
  info i;
  byte zero_points[3];
  float m;
  float s_out;
};
struct Worker{
  std::vector<Weight> w;
  std::vector<byte> inputs;
  bool status;
  std::vector<byte> ops;
};
#endif