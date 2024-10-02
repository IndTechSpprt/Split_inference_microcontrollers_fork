use crate::util::{coordinator_send, decode_u128, send_to_all_workers, wait_for_signal};
use algo::calculations::batchnorm;
use algo::{Mapping, QuantizedMapping, QuantizedWeightUnit, WeightUnit};
use serde::{Deserialize, Serialize};
use std::fs::{File, OpenOptions};
use std::io::Write;
use std::ops::{Add, AddAssign};
use std::result;
use std::sync::mpsc;
use std::sync::mpsc::RecvError;
use std::time::Instant;
use chrono::{Duration, TimeDelta};

pub type Work<T> = Option<T>;
pub type Result<T> = Option<T>;
#[derive(Debug, Serialize, Deserialize, Clone)]
pub enum Message<T> {
    Work(Work<T>),
    Result(Result<T>),
    Quit,
    StartTransmission,
}
#[derive(Serialize, Deserialize)]
pub struct Coordinator<T> {
    pub(crate) mapping: Vec<T>,
    // pub(crate) batch_norm: Vec<f32>,
    // pub(crate) operations: Vec<u8>,
}
#[derive(Debug, Serialize, Deserialize)]
pub struct Worker<T, U> {
    pub(crate) weights: Vec<T>,
    pub(crate) inputs: Vec<U>,
    pub status: bool,
    pub operations: Vec<u8>,
}

impl Coordinator<Mapping> {
    pub fn receive_and_send(
        &mut self,
        rec: &mpsc::Receiver<Message<f32>>,
        send: &Vec<mpsc::Sender<Message<f32>>>,
        worker_swarm_size: u8,
        res: &mut Vec<f32>,
        con: &Vec<Vec<i32>>,
        phase: usize,
    ) {
        // let mut intermediate = Vec::new();
        let mut flag = 0;
        let mut total_count = 0;
        for c in con {
            if c[0] == phase as i32 {
                if flag == 0 {
                    res.clear();
                    flag = 1;
                } else if flag == 2 {
                    flag = 3;
                }
            } else if c[1] == phase as i32 {
                flag = 2;
            }
        }
        for i in 0..worker_swarm_size as usize {
            send[i]
                .send(Message::StartTransmission)
                .expect("start transmission failed.");
            println!("coordinator start receiving from {:?}", i);
            let mut cur_phase = 0;
            let mut count = 0;
            loop {
                if !self.mapping.is_empty()
                    && cur_phase < self.mapping[i].count.len()
                    && !self.mapping[i].padding_pos[cur_phase].is_empty()
                    && count == self.mapping[i].padding_pos[cur_phase][0]
                {
                    let mut next_mcus = decode_u128(&self.mapping[i].map[cur_phase]);
                    coordinator_send(
                        next_mcus,
                        send,
                        0.,
                        &self.mapping[i].end_pos,
                        cur_phase,
                        count,
                    );
                    self.mapping[i].padding_pos[cur_phase].remove(0);
                    count += 1;
                    // total_count += 1;
                    if count == self.mapping[i].count[cur_phase] {
                        // println!("coordinator receiving from {:?} switch phase,count{:?},phase:{:?},total_count:{:?}",i,count,cur_phase,total_count);
                        cur_phase += 1;
                        count = 0;
                        if cur_phase >= self.mapping[i].count.len() {
                            //todo! send to the next coordinator
                            continue;
                        }
                    }
                } else if let Ok(data) = rec.recv() {
                    // println!("received data from {:?},data{:?} ",i,data);
                    match data {
                        Message::Result(Some(mut d)) => {
                            // intermediate.push(d);
                            if flag == 1 {
                                res.push(d);
                            } else if flag == 2 {
                                d += res[total_count];
                            } else if flag == 3 {
                                d += res[total_count];
                                res[total_count] = d;
                            }
                            if self.mapping.is_empty() {
                                send_to_all_workers(Message::Work(Some(d)), send);
                                continue;
                            }
                            let mut next_mcus = decode_u128(&self.mapping[i].map[cur_phase]);
                            coordinator_send(
                                next_mcus,
                                send,
                                d,
                                &self.mapping[i].end_pos,
                                cur_phase,
                                count,
                            );
                            count += 1;
                            total_count += 1;
                            if count == self.mapping[i].count[cur_phase] {
                                // println!("coordinator receiving from {:?} switch phase,count{:?},phase:{:?},total_count:{:?}",i,count,cur_phase,total_count);
                                cur_phase += 1;
                                count = 0;
                                if cur_phase >= self.mapping[i].count.len() {
                                    //todo! send to the next coordinator
                                    if phase == 0 {
                                        break;
                                    }
                                    continue;
                                }
                            }
                        }
                        Message::Result(None) => {
                            if self.mapping.is_empty() && i as u8 == worker_swarm_size - 1 {
                                send_to_all_workers(Message::Work(None), send);
                            }
                            // assert_eq!(count, 0);
                            // assert_eq!(cur_phase, self.mapping[i].count.len());
                            // println!("finished receiving from {:?}",i);
                            break;
                        }
                        _ => {}
                    }
                }
            }
        }
        // let serialized_inter = serde_json::to_string(&intermediate).unwrap();
        // let file_name = "intermediate_o.json".to_string();
        // let mut file = OpenOptions::new()
        //     .create(true)
        //     .append(true)
        //     .open("./".to_string() + "/" + &file_name)
        //     .unwrap();
        // writeln!(file, "{}", serialized_inter).unwrap();
    }
    // fn normalize(&mut self, input: f32, channel: u8) -> f32 {
    //     let mut result : f32 = 0.;
    //     for op in &self.operations {
    //         match op {
    //             1 => {
    //                 // result = batchnorm(input, &self.batch_norm, channel);
    //             } //batchnorm
    //             2 => {
    //                 result = result.clamp(0., 6.0);
    //             } //relu6
    //             _ => {}
    //         }
    //     }
    //     result
    // }
    pub fn receive_and_terminate(
        &self,
        rec: &mpsc::Receiver<Message<f32>>,
        send: &Vec<mpsc::Sender<Message<f32>>>,
        worker_swarm_size: u8,
    ) -> Vec<f32> {
        let mut result_vec = Vec::new();
        println!("coordinator receiving result");
        for i in 0..worker_swarm_size as usize {
            send[i]
                .send(Message::StartTransmission)
                .expect("start transmission failed.");
            println!("coordinator start receiving from {:?}", i);
            loop {
                if let Ok(data) = rec.recv() {
                    match data {
                        Message::Result(Some(d)) => {
                            result_vec.push(d);
                        }
                        Message::Result(None) => {
                            break;
                        }
                        _ => {}
                    }
                }
            }
            println!("coordinator send quit to {}", i);
            send[i].send(Message::Quit).unwrap();
        }
        // let serialized_inter = serde_json::to_string(&result_vec).unwrap();
        // let file_name = "intermediate_o.json".to_string();
        // let mut file = OpenOptions::new()
        //     .create(true)
        //     .append(true)
        //     .open("./".to_string()  + "/" + &file_name)
        //     .unwrap();
        // writeln!(file, "{}", serialized_inter).unwrap();
        result_vec
    }
}
impl Worker<WeightUnit, f32> {
    pub fn receive(&mut self, rec: &mpsc::Receiver<Message<f32>>, id: u8) {
        loop {
            if let Ok(data) = rec.recv() {
                match data {
                    Message::Work(Some(d)) => {
                        self.inputs.push(d);
                    }
                    Message::Work(None) => {
                        println!("worker{:?} breaking", id);
                        break;
                    }
                    Message::Quit => {
                        self.status = true;
                        break;
                    }
                    _ => {}
                }
            }
        }
    }
    pub fn work(
        self,
        sender: &mpsc::Sender<Message<f32>>,
        rec: &mpsc::Receiver<Message<f32>>,
        id: u8,
    ) -> Vec<f32> {
        let mut result = algo::operations::distributed_computation(self.inputs, self.weights);
        if self.operations.contains(&1) {
            for i in 0..result.len() {
                result[i] = result[i].clamp(0., 6.);
            }
        }
        let mut buffer = Vec::new();
        // println!("worker{:?},result size:{:?}",id,result.len());
        wait_for_signal(rec, &mut buffer);
        let start_time = Instant::now();
        for i in result {
            sender.send(Message::Result(Some(i))).unwrap();
        }
        sender
            .send(Message::Result(None))
            .expect("Send None is not allowed");
        println!(
            "worker{:?} send None,time consumed:{:?}",
            id,
            start_time.elapsed()
        );
        buffer
    }
    pub fn adaptive_pooling(&mut self) {
        // can be done in worker or coordinator, here I choose to do it in worker, may adjust this according to the ram size of the worker
        let window_size = self.inputs.len() / 1280;
        let len = self.inputs.len();
        let mut result_index = 0;
        while result_index + window_size <= self.inputs.len() {
            let avg = self.inputs[result_index..result_index + window_size]
                .iter()
                .sum::<f32>()
                / window_size as f32;
            self.inputs[result_index] = avg;
            self.inputs
                .drain(result_index + 1..result_index + window_size);
            result_index += 1;
        }
        assert_eq!(self.inputs.len(), 1280);
    }
}
impl Coordinator<QuantizedMapping> {
    pub fn receive_and_send_q(
        &mut self,
        rec: &mpsc::Receiver<Message<u8>>,
        send: &Vec<mpsc::Sender<Message<u8>>>,
        worker_swarm_size: u8,
        res: &mut Vec<u8>,
        con: &Vec<Vec<i32>>,
        phase: usize,
        parameters: &mut ((u8, u8, u8), (f32, f32, f32)),
    ) {
        // let mut intermediate = Vec::new();
        let mut flag = 0;
        let mut total_count = 0;
        for c in con {
            if phase as i32 + 1 == c[1] {
                parameters.0 .1 = self.mapping[0].zero_point.2;
                parameters.1 .1 = self.mapping[0].scale.2;
            }
            if c[0] == phase as i32 {
                if flag == 0 {
                    res.clear();
                    flag = 1;
                    parameters.0 .0 = self.mapping[0].zero_point.0;
                    parameters.1 .0 = self.mapping[0].scale.0;
                } else if flag == 2 {
                    flag = 3;
                }
            } else if c[1] == phase as i32 {
                flag = 2;
                parameters.0 .2 = self.mapping[0].zero_point.0;
                parameters.1 .2 = self.mapping[0].scale.0;
            }
        }
        for i in 0..worker_swarm_size as usize {
            send[i]
                .send(Message::StartTransmission)
                .expect("start transmission failed.");
            println!("coordinator start receiving from {:?}", i);
            let mut cur_phase = 0;
            let mut count = 0;
            loop {
                if !self.mapping.is_empty()
                    && cur_phase < self.mapping[i].count.len()
                    && !self.mapping[i].padding_pos[cur_phase].is_empty()
                    && count == self.mapping[i].padding_pos[cur_phase][0]
                {
                    let mut next_mcus = decode_u128(&self.mapping[i].map[cur_phase]);
                    coordinator_send(
                        next_mcus,
                        send,
                        self.mapping[i].zero_point.0,
                        &self.mapping[i].end_pos,
                        cur_phase,
                        count,
                    );
                    self.mapping[i].padding_pos[cur_phase].remove(0);
                    count += 1;
                    if count == self.mapping[i].count[cur_phase] {
                        // println!("coordinator receiving from {:?} switch phase,count{:?},phase:{:?},total_count:{:?}",i,count,cur_phase,total_count);
                        cur_phase += 1;
                        count = 0;
                        if cur_phase >= self.mapping[i].count.len() {
                            //todo! send to the next coordinator
                            continue;
                        }
                    }
                } else if let Ok(data) = rec.recv() {
                    // println!("received data from {:?},data{:?} ",i,data);
                    match data {
                        Message::Result(Some(mut d)) => {
                            // intermediate.push(d);
                            if flag == 1 {
                                res.push(d);
                            } else if flag == 2 {
                                // println!("phase:{:?},parameter:{:?}",phase,parameters);
                                // panic!("!!");
                                d = (((d as f32 - parameters.0 .1 as f32) * parameters.1 .1
                                    + (res[total_count] as f32 - parameters.0 .0 as f32)
                                        * parameters.1 .0)
                                    / parameters.1 .2
                                    + parameters.0 .2 as f32)
                                    .round()
                                    .clamp(0., 255.) as u8;
                            } else if flag == 3 {
                                d = (((d as f32 - parameters.0 .1 as f32) * parameters.1 .1
                                    + (res[total_count] as f32 - parameters.0 .0 as f32)
                                        * parameters.1 .0)
                                    / parameters.1 .2
                                    + parameters.0 .2 as f32)
                                    .round()
                                    .clamp(0., 255.) as u8;
                                res[total_count] = d;
                            }
                            if self.mapping.is_empty() {
                                send_to_all_workers(Message::Work(Some(d)), send);
                                continue;
                            }
                            let mut next_mcus = decode_u128(&self.mapping[i].map[cur_phase]);
                            coordinator_send(
                                next_mcus,
                                send,
                                d,
                                &self.mapping[i].end_pos,
                                cur_phase,
                                count,
                            );
                            count += 1;
                            total_count += 1;
                            if count == self.mapping[i].count[cur_phase] {
                                // println!("coordinator receiving from {:?} switch phase,count{:?},phase:{:?},total_count:{:?}",i,count,cur_phase,total_count);
                                cur_phase += 1;
                                count = 0;
                                if cur_phase >= self.mapping[i].count.len() {
                                    //todo! send to the next coordinator
                                    if phase == 0 {break; }
                                    continue;
                                }
                            }
                        }
                        Message::Result(None) => {
                            if self.mapping.is_empty() && i as u8 == worker_swarm_size - 1 {
                                send_to_all_workers(Message::Work(None), send);
                            }
                            // assert_eq!(count, 0);
                            // assert_eq!(cur_phase, self.mapping[i].count.len());
                            // println!("finished receiving from {:?}",i);
                            break;
                        }
                        _ => {}
                    }
                }
            }
        }
        if flag == 3 {
            parameters.0 .0 = self.mapping[0].zero_point.0;
            parameters.1 .0 = self.mapping[0].scale.0;
        }
        // let serialized_inter = serde_json::to_string(&intermediate).unwrap();
        // let file_name = "intermediate_q.json".to_string();
        // let mut file = OpenOptions::new()
        //     .create(true)
        //     .append(true)
        //     .open("./".to_string()  + "/" + &file_name)
        //     .unwrap();
        // writeln!(file, "{}", serialized_inter).unwrap();
    }
    pub fn receive_and_terminate_q(
        &self,
        rec: &mpsc::Receiver<Message<u8>>,
        send: &Vec<mpsc::Sender<Message<u8>>>,
        worker_swarm_size: u8,
    ) -> Vec<u8> {
        let mut result_vec = Vec::new();
        println!("coordinator receiving result");
        for i in 0..worker_swarm_size as usize {
            send[i]
                .send(Message::StartTransmission)
                .expect("start transmission failed.");
            println!("coordinator start receiving from {:?}", i);
            loop {
                if let Ok(data) = rec.recv() {
                    match data {
                        Message::Result(Some(d)) => {
                            result_vec.push(d);
                        }
                        Message::Result(None) => {
                            break;
                        }
                        _ => {}
                    }
                }
            }
            println!("coordinator send quit to {}", i);
            send[i].send(Message::Quit).unwrap();
        }
        // let serialized_inter = serde_json::to_string(&result_vec).unwrap();
        // let file_name = "intermediate_q.json".to_string();
        // let mut file = OpenOptions::new()
        //     .create(true)
        //     .append(true)
        //     .open("./".to_string()  + "/" + &file_name)
        //     .unwrap();
        // writeln!(file, "{}", serialized_inter).unwrap();
        result_vec
    }
}
impl Worker<QuantizedWeightUnit, u8> {
    pub fn receive_q(&mut self, rec: &mpsc::Receiver<Message<u8>>, id: u8) {
        loop {
            if let Ok(data) = rec.recv() {
                match data {
                    Message::Work(Some(d)) => {
                        self.inputs.push(d);
                    }
                    Message::Work(None) => {
                        println!("worker{:?} breaking", id);
                        break;
                    }
                    Message::Quit => {
                        self.status = true;
                        break;
                    }
                    _ => {}
                }
            }
        }
    }
    pub fn work_q(
        self,
        sender: &mpsc::Sender<Message<u8>>,
        rec: &mpsc::Receiver<Message<u8>>,
        id: u8,
        mut calculation_duration: &mut TimeDelta,
        input_len_file: &mut File,
        res_len_file: &mut File
    ) -> Vec<u8> {
        let max = (6. / self.weights[0].s_out).round().clamp(0., 255.) as u8;
        input_len_file.write_fmt(format_args!("{}, ",self.inputs.len())).unwrap();
        let temp = Instant::now();
        let mut result = algo::operations::distributed_computation_quant(self.inputs, self.weights);
        calculation_duration.add_assign(TimeDelta::new(temp.elapsed().as_secs() as i64,temp.elapsed().subsec_nanos()).unwrap());
        // println!("time spent on calculation:{:?}",calculation_duration.to_std().unwrap());
        println!("id: {:?}, len: {:?}",id,result.len());
        res_len_file.write_fmt(format_args!("{}, ",result.len())).unwrap();
        if self.operations.contains(&1) {
            for i in 0..result.len() {
                result[i] = result[i].clamp(0, max); //todo change rulu int relu6
            }
        }
        let mut buffer = Vec::new();
        // println!("worker{:?},result size:{:?}",id,result.len());
        wait_for_signal(rec, &mut buffer);
        let start_time = Instant::now();
        for i in result {
            sender.send(Message::Result(Some(i))).unwrap();
        }
        sender
            .send(Message::Result(None))
            .expect("Send None is not allowed");
        println!(
            "worker{:?} send None,time consumed:{:?}",
            id,
            start_time.elapsed()
        );
        buffer
    }
    pub fn adaptive_pooling_q(&mut self) {
        // can be done in worker or coordinator, here I choose to do it in worker, may adjust this according to the ram size of the worker
        let window_size = self.inputs.len() / 1280;
        let len = self.inputs.len();
        let mut result_index = 0;
        while result_index + window_size <= self.inputs.len() {
            let avg = self.inputs[result_index..result_index + window_size]
                .iter()
                .map(|&x| x as u16)
                .sum::<u16>()
                / window_size as u16;
            self.inputs[result_index] = avg as u8;
            self.inputs
                .drain(result_index + 1..result_index + window_size);
            result_index += 1;
        }
        assert_eq!(self.inputs.len(), 1280);
    }
}
