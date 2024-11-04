//200: permission granted 199: ask for permission 198:send complete 197:ack, 196:adaptive pooling
#ifndef COMMUNICATION_H
#define COMMUNICATION_H
#include <SPI.h>
#include <NativeEthernet.h>

// Protocol Defs

#define MESSAGE_SIZE 1400
#define RESERVED_BYTES 6 //|from which|to which/message type|length * 4|

// Packet content
#define SOURCE_ID 0
#define MESSAGE_HEADER 1
#define MESSAGE_LENGTH 2

/// @brief All supported message headers
enum MessageHeaders{
  Inference_Results     = (byte)195,
  Adaptive_Pooling      = (byte)196,
  ACK                   = (byte)197,
  Complete              = (byte)198,
  Request_Permission    = (byte)199,
  Permission_Granted    = (byte)200
};

//Server defines
IPAddress server(169,254,71,125);
#define SERVER_PORT 8080

// Define ip address and ports for each MCU
IPAddress ip(169,254,71,IP_END);
byte mac[] = {
  0xDE, 0xAD, 0xBE, 0xEF, 0xFE, MAC_END
};

std::vector<std::vector<int>> residual_connection = {{6,9},{12,15},{15,18},{21,24},{24,27},{27,30},{33,36},{36,39},{42,45},{45,48}};
std::vector<std::vector<byte>> zps = {{117,123,121},{136,133,132},{132,131,135},{132,135,130},{130,136,131},{131,142,132},{126,128,126},{126,125,124},{133,121,126},{126,128,126}};
std::vector<std::vector<float>> scales = {{0.026925236,0.030070057,0.040337086},{0.021326661,0.018447906,0.026975844},{0.026975844,0.01711597,0.03300309},{0.017282467,0.013020085,0.019367196},{0.019367196,0.010685049,0.020728296},{0.020728296,0.013996841,0.022537425},{0.016347256,0.011939427,0.019542953},{0.019542953,0.017455006,0.026837224},{0.013501433,0.010944449,0.019419255},{0.019419255,0.022365179,0.0346372}};

EthernetClient client;
bool permission_flag = false;
/// @brief Print device information on serial to help identify the MCU
void print_device_information() {
  Serial.println("Device Information:");
  Serial.print("MCU ID: ");
  Serial.print(MCU_ID);
  Serial.print(" IP: ");
  Serial.println(ip);
}
void setup_communication() {
  Ethernet.setStackHeap(10 * 1024);
  Ethernet.setSocketSize(8 * 1024);
  Ethernet.setSocketNum(1);
  Ethernet.begin(mac,ip);
  Serial.println("connecting...");
  while (!client.connect(server, SERVER_PORT)) {} //connect to server
  while (!client.available()) {} //read a byte from server to indicate communication established
  char c = client.read();
  Serial.println("connection established!");
  print_device_information();
}
bool wait_for_ack(){
  while(!client.available()){};
  // delay(5);
  char message[3];
  client.readBytes(message,3);
  if(message[MESSAGE_HEADER] != ACK){
    return true;
  }
  return false;
}
bool send_message_to_coordinator(const char* message){
  delay(10);
  if(client.write(message,MESSAGE_SIZE) == 0) return false;
  if(wait_for_ack()){
    Serial.println("ack message wrong, stop executing...");
    while(1){};
  };
  return true;  
}

void send_ack(){
  // delay(5);
  char message[3];
  message[SOURCE_ID] = MCU_ID;
  message[MESSAGE_HEADER] = ACK;
  client.write(message,MESSAGE_SIZE);
}

void sendtoMCUs(char* message, std::vector<byte>& MCUs,const byte cur_mcu,std::vector<byte>& cur_input,int& rec_count,int& send_count){
  message[1] = 0;
  for(byte m : MCUs){
    if(m == cur_mcu){
      for(int i = RESERVED_BYTES; i <send_count + RESERVED_BYTES; i++ ){
        cur_input[rec_count] = message[i];
        rec_count += 1;
      }
      if(MCUs.size() == 1){
        delay(20);
      }
    }
    else{
      message[1] |= 1 << m;
    }     
  }
  if(message[1] != 0){
    send_message_to_coordinator(message);    
  }
}
void check_and_receive(int& rec_count,std::vector<byte>& input_distribution){
  int count = 0;
  if(client.available()){  
    delay(10);
    byte buffer[MESSAGE_SIZE];
    client.readBytes(buffer,MESSAGE_SIZE);
    if(buffer[1] == Permission_Granted){ 
      permission_flag = true;
    }else{
      byte from_which = buffer[0];
      byte to_which = buffer[1]; 
      int length = 0;
      memcpy(&length, buffer + 2, sizeof(int));
      if(to_which != MCU_ID){
        Serial.println("FATAL ERROR: Received wrong message!!! Stop execution..");
        Serial.println((byte) to_which);
        while(1){};
      }
      for(int i = 0; i < length; i++){
        input_distribution[rec_count] = buffer[i + RESERVED_BYTES];
        rec_count += 1;
      } 
    }
    send_ack();      
  }
}
void wait_for_permission(int& rec_count,std::vector<byte>& input_distribution){
  check_and_receive(rec_count,input_distribution);
  char message[MESSAGE_SIZE];
  message[SOURCE_ID] = MCU_ID;
  message[MESSAGE_HEADER] = Request_Permission; // 199 for ask for permission
  Serial.println("send request...");
  client.write(message,MESSAGE_SIZE);
  while(!permission_flag){
    check_and_receive(rec_count,input_distribution);
  }
  permission_flag = false; //reset the flag
}
void write_length(byte* message,int length){
    char intBytes[sizeof(int)];
    memcpy(intBytes, &length, sizeof(int));
    memcpy(message + 2, intBytes, sizeof(int));
}
#endif
