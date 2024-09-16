#include "worker_struct.h"
#include "filesys.h"
byte type = 0;
void setup() {
  // 启动串口通讯
  Serial.begin(9600);
  setup_filesys();
  Serial.println("---ready to download---");
}
void loop() {
  if (Serial.available() && !write_data) {
    char rr;
    rr = Serial.read();
    switch (rr) {
      case 'l': listFiles(); break;
      case 'e': eraseFiles(); break;
      case 's':
        {
          Serial.println("\nLogging Data!!!");
          write_data = true;  // sets flag to continue to write data until new command is received
          // opens a file or creates a file if not present,  FILE_WRITE will append data to
          // to the file created.
          String filename = "datalog.bin";
          type = 1;
          dataFile = myfs.open(filename.c_str(), FILE_WRITE);
          delay(1000);
          // logData(phase);
        }
        break;
      // case 'x': stopLogging(); break;
      case 'c' :{
          Serial.println("\nLogging Coordinator!!!");
          write_data = true;  // sets flag to continue to write data until new command is received
          // opens a file or creates a file if not present,  FILE_WRITE will append data to
          // to the file created.
          String filename = "Coordinator.bin";
          type = 2;
          dataFile = myfs.open(filename.c_str(), FILE_WRITE);
          delay(1000);
      }
      break;
      case 'd': dumpLog(); break;
      case '\r':
      case '\n':
      case 'h': menu(); break;
    }
    while (Serial.read() != -1);  // remove rest of characters.
  }
  if (Serial.available() && write_data){
    if(type == 1){
     logData(phase); 
    }
    else if(type == 2){
      logCoordinator();
    }
    else{
      write_data = false;
    }
  } 
}