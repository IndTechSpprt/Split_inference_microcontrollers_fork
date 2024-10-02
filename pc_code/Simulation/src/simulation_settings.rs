use crate::nodes::{Coordinator, Message, Worker};
use crate::util::{
    decode_coordinator, decode_worker, flatten_3d_array, generate_test_input, test_equal,
};
use algo::util::{pre_processing, read_and_store_image};
use algo::{QuantizedMapping, QuantizedWeightUnit};
use chrono::prelude::*;
use core::num;
use std::fs::File;
use std::io::{BufRead, BufReader};
use std::sync::{mpsc, Arc, Mutex};
use std::thread;
use std::time::Instant;
use chrono::{Duration, TimeDelta};
use algo::operations::find_which_cpu;

pub fn preparation_phase() {
    todo!()
} //distribute weight, analyse mapping,distribute coordinators,distribute workers write into files.
pub fn c_1_simulation(num_workers: u8, end: usize) {
    // 创建一个消息发送者和多个消息接收者
    let residual_connections = vec![
        vec![6, 9],   //10,15
        vec![12, 15], //20,25
        vec![15, 18], //25,30,
        vec![21, 24], //35,40
        vec![24, 27], //40,45
        vec![27, 30], //45,50
        vec![33, 36], //55,60
        vec![36, 39], //60,65
        vec![42, 45], //70,75
        vec![45, 48], //75,80
    ];
    let (coordinator_sender, coordinator_receiver) = mpsc::channel::<Message<f32>>();
    let start_time = Instant::now();
    let mut handles = vec![];
    let mut worker_send_channel = vec![];
    for worker_id in 0..num_workers {
        let (worker_sender, worker_receiver) = mpsc::channel::<Message<f32>>();
        let coordinator_sender_clone = coordinator_sender.clone();
        let file_name = format!("pc_code/Simulation/Simu/worker_{:?}.json", worker_id);
        let handle = thread::spawn(move || {
            let mut phase = 0;
            let mut buffer = Vec::new();
            // Worker线程的接收端
            loop {
                if phase >= 53 {
                    phase = 0
                };
                let mut worker = decode_worker(&file_name, phase, buffer).unwrap();
                println!(
                    "worker{:?} start receiving,time:{:?}",
                    worker_id,
                    start_time.elapsed()
                );
                worker.receive(&worker_receiver, worker_id);
                println!(
                    "worker{:?} finished receiving,time:{:?}",
                    worker_id,
                    start_time.elapsed()
                );
                if worker.status == true {
                    break;
                }
                if phase == 52 {
                    worker.adaptive_pooling();
                }
                buffer = worker.work(&coordinator_sender_clone, &worker_receiver, worker_id); //buffer is the data received while working
                phase += 1;
            }
            println!("worker{:?}, exited", worker_id);
        });

        // 主线程将Worker线程的发送端和句柄保存在Vec中
        handles.push(handle);
        worker_send_channel.push(worker_sender);
    }
    let file_name = "pc_code/Simulation/Simu/Coordinator.json";
    let coordinator_handle = thread::spawn(move || {
        let mut phase = 0;
        let mut res = Vec::new();
        loop {
            if phase >= end {
                let coodinator = Coordinator {
                    mapping: vec![],
                    // operations: vec![],
                };
                let result_vec = coodinator.receive_and_terminate(
                    &coordinator_receiver,
                    &worker_send_channel,
                    num_workers,
                );
                println!("{:?}", result_vec);
                // test_equal(result_vec);
                break;
            }
            match decode_coordinator(file_name, phase) {
                Ok(mut coordinator) => {
                    coordinator.receive_and_send(
                        &coordinator_receiver,
                        &worker_send_channel,
                        num_workers,
                        &mut res,
                        &residual_connections,
                        phase,
                    );
                    println!("phase{:?} finished", phase);
                    phase += 1;
                }
                Err(me) => {
                    let coodinator = Coordinator {
                        mapping: vec![],
                        // operations: vec![],
                    };
                    let result_vec = coodinator.receive_and_terminate(
                        &coordinator_receiver,
                        &worker_send_channel,
                        num_workers,
                    );
                    if let Some((index, val)) = result_vec
                        .iter()
                        .enumerate()
                        .max_by(|(_, &a), (_, &b)| a.partial_cmp(&b).unwrap())
                    {
                        println!("Index of the biggest element: {} {}", index, val);
                    } else {
                        println!("Vector is empty.");
                    }
                    // test_equal(result_vec);
                    break;
                }
            }
        }
    });
    handles.push(coordinator_handle);
    //intput
    let image = pre_processing(read_and_store_image("pc_code/Algorithms/images/img.png").unwrap());
    let input = flatten_3d_array(image);
    for i in 0..input.len(){
        coordinator_sender.send(Message::Result(Some(input[i]))).expect("start failed");
    }
    // for i in 0..num_workers {
    //     let coordinator_sender_clone = coordinator_sender.clone();
    //     for j in 0..num_per_cpu {
    //         if count < input.len() {
    //             coordinator_sender_clone
    //                 .send(Message::Result(Some(input[count])))
    //                 .expect("Start failed");
    //         }
    //         count += 1;
    //     }
    //     coordinator_sender_clone
    //         .send(Message::Result(None))
    //         .expect("start failed");
    // }
    // 等待所有Worker线程完成
    for handle in handles {
        handle.join().unwrap();
    }
} //start the simulation
pub fn c_1_simulation_quant(num_workers: u8, end: usize) {
    // 创建一个消息发送者和多个消息接收者
    let residual_connections = vec![
        vec![6, 9],   //10,15
        vec![12, 15], //20,25
        vec![15, 18], //25,30,
        vec![21, 24], //35,40
        vec![24, 27], //40,45
        vec![27, 30], //45,50
        vec![33, 36], //55,60
        vec![36, 39], //60,65
        vec![42, 45], //70,75
        vec![45, 48], //75,80
    ];
    let (coordinator_sender, coordinator_receiver) = mpsc::channel::<Message<u8>>();
    let start_time = Instant::now();
    let mut handles = vec![];
    let mut worker_send_channel = vec![];
    for worker_id in 0..num_workers {
        let (worker_sender, worker_receiver) = mpsc::channel::<Message<u8>>();
        let coordinator_sender_clone = coordinator_sender.clone();
        let file_name = format!("pc_code/Simulation/Simu_q/worker_{:?}.json", worker_id);
        let mut res_file = File::create(format!("pc_code/Simulation/Simu_q/res_len_{}.txt",worker_id)).unwrap();
        let mut in_file = File::create(format!("pc_code/Simulation/Simu_q/input_len_{}.txt",worker_id)).unwrap();
        let handle = thread::spawn(move || {
            let mut phase = 0;
            let mut buffer = Vec::new();
            let mut calc_duration: TimeDelta = TimeDelta::zero();
            // Worker线程的接收端
            loop {
                if phase >= 53 {
                    phase = 0
                };
                let mut worker: Worker<QuantizedWeightUnit, u8> =
                    decode_worker(&file_name, phase, buffer).unwrap();
                println!(
                    "worker{:?} start receiving,time:{:?}",
                    worker_id,
                    start_time.elapsed()
                );
                worker.receive_q(&worker_receiver, worker_id);
                println!(
                    "worker{:?} finished receiving,time:{:?}",
                    worker_id,
                    start_time.elapsed()
                );
                if worker.status == true {
                    break;
                }
                if phase == 52 {
                    worker.adaptive_pooling_q();
                }
                buffer = worker.work_q(&coordinator_sender_clone, &worker_receiver, worker_id,&mut calc_duration, &mut in_file, &mut res_file); //buffer is the data received while working
                phase += 1;
            }
            println!("worker{:?}, exited", worker_id);
        });

        // 主线程将Worker线程的发送端和句柄保存在Vec中
        handles.push(handle);
        worker_send_channel.push(worker_sender);
    }
    let file_name = "pc_code/Simulation/Simu_q/Coordinator.json";
    let coordinator_handle = thread::spawn(move || {
        let mut residual: Vec<u8> = Vec::new();
        let mut parameters_res: ((u8, u8, u8), (f32, f32, f32)) = ((0, 0, 0), (0.0, 0., 0.));
        let mut phase = 0;
        let mut scales = Vec::new();
        let mut zero_points = Vec::new();
        loop {
            if phase >= end {
                let coodinator = Coordinator {
                    mapping: vec![],
                    // operations: vec![],
                };
                let result_vec = coodinator.receive_and_terminate_q(
                    &coordinator_receiver,
                    &worker_send_channel,
                    num_workers,
                );
                for i in 100000..130000{
                    print!("{}",result_vec[i]);
                    print!(" ");
                }
                // test_equal(result_vec);
                break;
            }
            match decode_coordinator::<QuantizedMapping>(file_name, phase) {
                Ok(mut coordinator) => {
                    if !coordinator.mapping.is_empty() {
                        scales.push(coordinator.mapping[0].scale.0);
                        zero_points.push(coordinator.mapping[0].zero_point.0);
                    }
                    coordinator.receive_and_send_q(
                        &coordinator_receiver,
                        &worker_send_channel,
                        num_workers,
                        &mut residual,
                        &residual_connections,
                        phase,
                        &mut parameters_res,
                    );
                    println!("phase{:?} finished", phase);
                    phase += 1;
                }
                Err(me) => {
                    println!("{:?}", scales);
                    println!("{:?}", zero_points);
                    let coodinator = Coordinator {
                        mapping: vec![],
                        // operations: vec![],
                    };
                    let result_vec = coodinator.receive_and_terminate_q(
                        &coordinator_receiver,
                        &worker_send_channel,
                        num_workers,
                    );
                    if let Some((index, val)) = result_vec
                        .iter()
                        .enumerate()
                        .max_by(|(_, &a), (_, &b)| a.partial_cmp(&b).unwrap())
                    {
                        println!("Index of the biggest element: {} {}", index, val);
                    } else {
                        println!("Vector is empty.");
                    } // test_equal(result_vec);
                    break;
                }
            }
        }
    });
    handles.push(coordinator_handle);
    //intput
    let image = pre_processing(read_and_store_image("pc_code/Algorithms/images/img.png").unwrap());
    let raw_input = flatten_3d_array(image);
    let input = raw_input
        .into_iter()
        .map(|x| (x / 0.017818455 + 114.38545).round().clamp(0., 255.) as u8)
        .collect::<Vec<u8>>(); //input quantization
    for i in 0..input.len(){
        coordinator_sender.send(Message::Result(Some(input[i]))).expect("start failed");
    }
    // 等待所有Worker线程完成
    for handle in handles {
        handle.join().unwrap();
    }
} //start the simulation
