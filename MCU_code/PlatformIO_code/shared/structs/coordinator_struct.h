#ifndef COORDINATOR_STRUCT
#define COORDINATOR_STRUCT
#include  <vector>
struct Mapping{
  std::vector<int> count;
  std::vector<std::vector<byte>> map;
  std::vector<std::vector<int>> padding_pos;
  std::vector<std::vector<int>> end_pos;
  int zero_point[3];
  float scale[3];
};
#endif