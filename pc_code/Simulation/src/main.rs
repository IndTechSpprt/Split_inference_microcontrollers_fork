extern crate core;

mod distribution;
mod nodes;
mod simulation_settings;
mod util;

use crate::distribution::{distribute_mapping_weight, distribute_mapping_weight_quant};
use crate::nodes::{Message, Result, Work};
use crate::simulation_settings::{c_1_simulation, c_1_simulation_quant};
use algo::decode;
use std::fs::File;
use std::sync::{mpsc, Arc, Mutex};
use std::thread;
//1 : 9.50529s  2:6.2564s 3: 5.9846s 4:5.8289s 5:8.40441s  6:8.6238s
fn main() {
    let file = File::open("./pc_code/Fused/fused_layers_141_small.json").expect("Failed to open file");
    let layers = decode::decode_json(file);
    let num_workers = 4;
    let protions = vec![1;num_workers as usize];
    distribute_mapping_weight_quant(layers,num_workers,(3,128,128),"pc_code/Simulation/Simu_q".to_string(),protions);
    c_1_simulation_quant(num_workers, 70);
}
