// Download weights application, sets up the file system and provides the user a menu that allows interaction with the filesystem and MCU, to download,\
// erase and view the stored files.

#include "worker_struct.h"
#include "filesys.h"
#include "menu.h"

WriteTypes type = Stop; //Current write type

void setup() {
  Serial.begin(115200);
  setup_filesys();
  Serial.println("---Ready to download, press 'h' to display the menu---");
}

void loop() {
  // Write data can be updated in the menu handler, making two checks necessary
  // When nothing is being written, call the menu handler, which updates the type
  if (Serial.available() && !write_data) {
    menu_handler();
  }

  // Write only when allowed, disable the menu handler
  if (Serial.available() && write_data) {
    switch (type) {
      case Data: logData(phase); break;
      case Coordinator: logCoordinator(); break;
      default: write_data = false; break;
    }
  }
}