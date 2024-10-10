#include "read.h"
#include "calculation.h"
#include "communication.h"
#include "menu.h"

extern unsigned long _heap_start;
extern unsigned long _heap_end;
extern char* __brkval;

#define FREE_RAM_SAMPLES 1200
volatile int ram_usage[FREE_RAM_SAMPLES];
volatile int samples = 0;

/// @brief  Determine how much RAM is available, by finding the difference between the end of the heap and the current allocated memory location
void saveRAMFreeSpace() {
  if (samples < FREE_RAM_SAMPLES) {
    ram_usage[samples] = 524288 - ((char*)&_heap_end - __brkval);
    samples+=1;
  }
}

IntervalTimer ramUsageTimer; //Interval timer to keep track of RAM usage

WriteTypes type = Stop; //Current write type

byte* input_distribution;
byte* overflow = nullptr;  // Initialize overflow pointer
bool overflow_flag = false;
int rec_count = 0;
int ino_count = 0;
void setup() {
  ramUsageTimer.begin(saveRAMFreeSpace,100000);//Save RAM usage at 1 ms intervals
  setup_filesys();
  {
    setup_communication(); 
    byte* temp = new(std::nothrow) byte[450 * 1024];
    if(temp != nullptr) {Serial.println("success");}
    delete[] temp;
    // Initialize coor_lines and lines
    read_line_by_line(COOR_LINES_FILENAME, coor_lines);
    read_line_by_line(LINES_FILENAME, lines);
  }
  for (int j = 0; j < 53; j++) {
    Serial.print("Current layer: ");
    Serial.println(j);
    if(j < 52){
        if(j == 0) input_distribution = new byte[input_length[0]];
        {
            Serial.print("rec_count is: ");
            Serial.println(rec_count);
            Serial.println("not enough inputs, receiving...");
            if(input_distribution == nullptr){
              while(1){
                Serial.println("input is nullptr!");
              }
            }
            while(rec_count != input_length[j]){
                check_and_receive(rec_count,input_distribution);
            }
            Serial.println("finished...");
            rec_count = 0;
        }
        int total_output_count = result_length[j];
        int result_size = std::min(total_output_count,STACK_SIZE);
        byte result[result_size] = { 0 };  // Initialize result array
        {
          std::vector<Weight> first_line;
          first_line = get_weights(j,prev_endpos);        
          int size = 0;
          // for (int i = 0; i < input_length[j]; i++) {
          //   input_distribution[i] = 0;
          // }            
          // input_distribution[0] = 0;
          ////////////////////////////
          // Check if the total output count exceeds the threshold
          if (total_output_count > STACK_SIZE) {
            overflow_flag = true;
            overflow = new byte[total_output_count - STACK_SIZE];  // Allocate memory for overflow
            Serial.println(total_output_count - STACK_SIZE);
          } else {
            overflow_flag = false;
          }
          distributed_computation(first_line, input_distribution, result, overflow, input_length[j]);
          handle_residual(result,result_length[j],j,residual_connection,zps,scales);
          if(input_distribution != nullptr) delete[] input_distribution;
        }
        if (overflow_flag) {
          otf(overflow, total_output_count - STACK_SIZE);
          delete[] overflow;
        }
        input_distribution = new byte[input_length[j + 1]];
        Serial.println("waiting for permission...");
        wait_for_permission(rec_count,input_distribution);
        Serial.println("premission granted, sending results...");
        if (j < 51) {
          char to_send[MESSAGE_SIZE];
          to_send[0] = MCU_ID;
          int send_count = 0;
          Mapping mapping;
          // Serial.println("!!!!");
          mapping = get_mapping(j + 1);
          // Serial.println("got mapping");
          int phase = mapping.count.size();
          if (overflow_flag) {
            dataFile = myfs.open("overflow.bin", FILE_READ);
            Serial.println("opened overflow");
          }
          int core_count = 0;
          for (int i = 0; i < phase; i++) {
            std::vector<byte> mcu_mapped = decode_u128(mapping.map[i]);    
            int padding_pos_count = 0;
            for (int k = 0; k < mapping.count[i]; k++) {
              if (mapping.padding_pos[i].size() > padding_pos_count && mapping.padding_pos[i][padding_pos_count] == k) {
                //send zero point to other MCUs
                // Serial.println("sending");
                to_send[send_count + RESERVED_BYTES] = mapping.zero_point[0];
                send_count += 1;
                if(send_count == MESSAGE_SIZE - RESERVED_BYTES){
                  write_length(to_send,send_count);
                  sendtoMCUs(to_send,mcu_mapped,MCU_ID,input_distribution,rec_count,send_count);
                  send_count = 0;
                }
                // Serial.println("send complete");
                padding_pos_count += 1;
              } else {
                if (core_count >= STACK_SIZE && overflow_flag) {
                  int count = 0;
                  to_send[send_count + RESERVED_BYTES] = read_byte(count);
                  send_count += 1;
                  if(send_count == MESSAGE_SIZE - RESERVED_BYTES){
                    write_length(to_send,send_count);
                    sendtoMCUs(to_send,mcu_mapped,MCU_ID,input_distribution,rec_count,send_count);
                    send_count = 0;
                  }
                } else {
                  to_send[send_count + RESERVED_BYTES] = result[core_count];
                  send_count += 1;
                  if(send_count == MESSAGE_SIZE - RESERVED_BYTES){
                    write_length(to_send,send_count);
                    sendtoMCUs(to_send,mcu_mapped,MCU_ID,input_distribution,rec_count,send_count);
                    send_count = 0;
                  }
                }
                core_count += 1;
              }
              //check regularly to avoid clogging
              if(rec_count < input_length[j + 1]) {
                  check_and_receive( rec_count, input_distribution);
              }
            }
            //send the rest of the data
            if(send_count != 0 ){
              write_length(to_send,send_count);
              sendtoMCUs(to_send,mcu_mapped,MCU_ID,input_distribution,rec_count,send_count);
              send_count = 0;
            }
          }
          if (overflow_flag) dataFile.close();
          to_send[1] = Complete; //signal the end
          send_message_to_coordinator(to_send);
        }
        else if(j == 51){
          char to_send[MESSAGE_SIZE];
          to_send[0] = MCU_ID;
          to_send[1] = Adaptive_Pooling;
          int send_count = 0;
          for(int i = 0; i < result_length[j];i++){
            to_send[RESERVED_BYTES + send_count] = result[i];
            send_count += 1;
            if(send_count == MESSAGE_SIZE - RESERVED_BYTES){
              write_length(to_send,send_count);
              send_message_to_coordinator(to_send);
              send_count = 0;
            }
          }
          if(send_count != 0){
            write_length(to_send,send_count);
            send_message_to_coordinator(to_send);
            send_count = 0; 
          }
          to_send[1] = Complete;
          send_message_to_coordinator(to_send);
        }
        ///////////////////////////
      }
      else if(j >= 52 ){
        byte result[result_length[j]] = {0};
        int count = 0;
        reading_weight = true;
        {
          Serial.print("rec_count is: ");
          Serial.println(rec_count);
          Serial.println("not enough inputs, receiving...");
          while(rec_count != input_length[j]){
              check_and_receive(rec_count,input_distribution);
          }
          Serial.println("finished...");
          rec_count = 0;
        }
        while(reading_weight){
          byte segment[LINEAR_SEGMENT] = {0};
          std::vector<Weight> weight = get_weights(j,prev_endpos);
          distributed_computation(weight, input_distribution, segment, overflow, input_length[j]);
          for(int i = 0; i < LINEAR_SEGMENT; i++){
            result[i + count] = segment[i];
            if(i + count >= result_length[j]) break;
          }
          count += LINEAR_SEGMENT;
        }
        for(int k = 0; k < result_length[j]; k++){
          Serial.print(k);
          Serial.print(" ");
          Serial.println(result[k]);
        }
      }
    }
  ramUsageTimer.end();
}
void loop() {
  if (Serial.available()) {
    menu_handler();
  }
  // sendUDPMessage("1 to 2", ip2, localPort);
  // sendUDPMessage("1 to 3", ip3, localPort);
  // delay(100);
}