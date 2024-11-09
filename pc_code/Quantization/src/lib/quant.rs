use algo::util::{pre_processing, read_and_store_image};
use algo::{
    calculations, util, InfoWrapper, Layer, LayerWrapper, Mapping, QuantizedMapping,
    QuantizedWeightUnit, WeightUnit,
};
use std::cmp::{max, min};
use std::collections::HashMap;
use std::fs;
use std::fs::File;
use std::io::{BufRead, BufReader};

//r = (q-z) * s; https://arxiv.org/abs/1712.05877v1
pub fn quantize_layers_weights(
    layers: &HashMap<i32, Box<dyn Layer>>,
) -> (Vec<Vec<u8>>, Vec<f32>, Vec<f32>) {
    let mut res = Vec::new();
    let mut scales = vec![0.; 100];
    let mut zero_points = vec![0.; 100];
    //determine the float point range
    for i in 1..=layers.len() {
        let l = layers.get(&(i as i32));
        match l {
            None => {
                continue;
            }
            _ => {}
        }
        let layer = l.unwrap();
        let weights = layer.get_weights();
        if weights.is_empty() {
            continue;
        }
        let weights_max = weights
            .iter()
            .max_by(|a, b| a.partial_cmp(b).unwrap())
            .unwrap();
        let weights_min = weights
            .iter()
            .min_by(|a, b| a.partial_cmp(b).unwrap())
            .unwrap();
        let range = weights_max - weights_min;
        let scale = range / 255.;
        let zero_point = -(weights_min / scale); // z = -r / s + q
        let mut weights_quantized = layer
            .get_weights()
            .into_iter()
            .map(|x| ((x / scale) + (zero_point)).round() as u8)
            .collect::<Vec<u8>>();

        res.push(weights_quantized);
        scales[i] = scale;
        zero_points[i] = zero_point;
        // print some property of the weights
        // let mean = weights.iter().map(|&x| x as f64).sum::<f64>() / weights.len() as f64;
        // let squared_diff_sum: f64 = weights
        //     .iter()
        //     .map(|&x| (x as f64 - mean).powi(2))
        //     .sum();
        // let mut variance = squared_diff_sum / weights.len() as f64;
        // variance = variance.sqrt();
        // println!("mean:{},std:{},max{},min{},range{}",mean,variance,weights_max,weights_min,range);
    }
    println!("scales:{:?},zero:{:?}", scales, zero_points);
    (res, scales, zero_points)
}
pub fn quantize_layers_activation(
    layers: HashMap<i32, Box<dyn Layer>>,
    calibration_set: String,
) -> (Vec<u32>, Vec<u8>) {
    // M = S1 * S2 / S3;
    let mut m_scale: Vec<u32> = vec![0; 100];
    let mut scales: Vec<f32> = vec![0.; 100];
    let mut zero_points: Vec<f32> = vec![0.; 100];
    let mut residual_scale: Vec<f32> = vec![0.; 100];
    let mut residual_zero_points: Vec<f32> = vec![0.; 100];
    let mut test_result = Vec::new();
    let residual_connections = vec![
        vec![10, 15], //10,15
        vec![20, 25], //20,25
        vec![25, 30], //25,30,
        vec![35, 40], //35,40
        vec![40, 45], //40,45
        vec![45, 50], //45,50
        vec![55, 60], //55,60
        vec![60, 65], //60,65
        vec![70, 75], //70,75
        vec![75, 80], //75,80
    ];
    // Read the directory entries
    if let Ok(entries) = fs::read_dir(calibration_set) {
        for entry in entries {
            if let Ok(entry) = entry {
                let file_path = entry.path();
                // Check if it's a file (not a directory, symlink, etc.)
                if file_path.is_file() {
                    // Do something with the file, e.g., print its path
                    println!("File: {:?}", file_path);
                    println!("scales:{:?}", scales);
                    println!("zero_points:{:?}", zero_points);
                    println!("resi scales:{:?}", residual_scale);
                    println!("resi zero:{:?}", residual_zero_points);
                    let image = read_and_store_image(file_path.to_str().unwrap()).unwrap();
                    let mut input = pre_processing(image);
                    let mut intermediate_output: Vec<Vec<Vec<f32>>> = Vec::new();
                    for i in 1..=layers.len() + 1 {
                        //find the maximum and minimum element in the input
                        let (mi, ma) = input
                            .iter()
                            .flat_map(|row| row.iter().flat_map(|col| col.iter()))
                            .fold((f32::INFINITY, f32::NEG_INFINITY), |(mi, ma), &value| {
                                (mi.min(value), ma.max(value))
                            });
                        //calculate the scale the zero point
                        let range = ma - mi;
                        let scale = range / 255.;
                        let zero_point = -(mi / scale).round(); // z = -r / s + q
                                                                //use EWMA to get the scale and zero point
                        scales[i] = scales[i] * 0.99 + 0.01 * (scale);
                        zero_points[i] = zero_points[i] * 0.99 + 0.01 * (zero_point);
                        //perform forward operation
                        if i == 88 {
                            for i in 0..input.len() {
                                let temp = &input[i];
                                let mut acc = 0.;
                                temp.into_iter()
                                    .for_each(|x| acc += x.into_iter().sum::<f32>());
                                let mean = acc / input[i].len() as f32 / input[i][0].len() as f32;
                                input[i] = vec![vec![mean]];
                            } //adaptive pooling
                              // continue
                        }
                        if i > layers.len() {
                            // print!("!!!!!!!");
                            test_result = input;
                            break;
                        }
                        let layer = layers.get(&(i as i32)).unwrap();
                        let output_shape = layer.get_output_shape();
                        match layer.identify() {
                            "Convolution" => {
                                let mut output = vec![
                                    vec![
                                        vec![0.; output_shape[2] as usize];
                                        output_shape[1] as usize
                                    ];
                                    output_shape[0] as usize
                                ];
                                let mut flag = true;
                                for j in 0..output_shape[0] as usize {
                                    flag = true;
                                    let mut weights: Vec<f32> = Vec::new();
                                    let mut bias = layer.get_bias(j as i32);
                                    for k in 0..output_shape[1] as usize {
                                        for m in 0..output_shape[2] as usize {
                                            let pos = vec![j as i32, k as i32, m as i32];
                                            let inputs_p = layer.get_input(pos);
                                            //each output channel only need to sample weight once
                                            if flag {
                                                weights = layer.get_weights_from_input(
                                                    inputs_p.clone(),
                                                    j as i32,
                                                );
                                                flag = false;
                                            }
                                            let inputs = util::sample_input_from_p_zero_padding(
                                                inputs_p, &input,
                                            );
                                            let result = calculations::vector_mul_b(
                                                inputs,
                                                weights.clone(),
                                                bias,
                                            );
                                            output[j][k][m] = result;
                                        }
                                    }
                                }
                                //next layer's input = this layer's output
                                input = output;
                            }
                            "Batchnorm2d" => {
                                let Ok(_a) = layer.functional_forward(&mut input) else {
                                    panic!("wrong layer")
                                };
                            }
                            "Relu6" => {
                                let Ok(_a) = layer.functional_forward(&mut input) else {
                                    panic!("wrong layer")
                                };
                            }
                            "Linear" => {
                                assert_eq!(input.len(), 1280);
                                assert!(input[0].len() == 1 && input[0][0].len() == 1);
                                let mut output = vec![vec![vec![0.0]]; 1000];
                                let weights = layer.get_weights();
                                if let InfoWrapper::Linear(info) = layer.get_info() {
                                    let weights_shape = [info.c_out, info.c_in]; //1000,1280
                                    for i in 0..weights_shape[0] as usize {
                                        let mut acc = 0.;
                                        for j in 0..weights_shape[1] as usize {
                                            acc += weights[i * weights_shape[1] as usize + j]
                                                * input[j][0][0];
                                        }
                                        output[i][0][0] = acc + layer.get_bias(i as i32);
                                    }
                                } else {
                                    panic!("not a linear layer")
                                }
                                input = output;
                            }
                            _ => {}
                        }
                        //handle residual connection
                        for r in 0..residual_connections.len() {
                            if residual_connections[r][1] == i {
                                let (mi, ma) = input
                                    .iter()
                                    .flat_map(|row| row.iter().flat_map(|col| col.iter()))
                                    .fold(
                                        (f32::INFINITY, f32::NEG_INFINITY),
                                        |(mi, ma), &value| (mi.min(value), ma.max(value)),
                                    );
                                //calculate the scale the zero point
                                let range = ma - mi;
                                let scale = range / 255.;
                                let zero_point = -(mi / scale).round(); // z = -r / s + q
                                                                        //use EWMA to get the scale and zero point
                                residual_scale[i] = residual_scale[i] * 0.99 + 0.01 * (scale);
                                residual_zero_points[i] =
                                    residual_zero_points[i] * 0.99 + 0.01 * (zero_point);
                                for j in 0..output_shape[0] as usize {
                                    for k in 0..output_shape[1] as usize {
                                        for m in 0..output_shape[2] as usize {
                                            input[j][k][m] += intermediate_output[j][k][m];
                                        }
                                    }
                                }
                            }
                            if residual_connections[r][0] == i {
                                intermediate_output = input.clone();
                            }
                        }
                    }
                }
            }
        }
    } else {
        println!("Error reading directory");
    }
    (
        m_scale,
        zero_points.into_iter().map(|x| x.round() as u8).collect(),
    )
}
pub fn calculate_quantization(
    original_weights: Vec<Vec<WeightUnit>>,
    original_mapping: Vec<Mapping>,
    weight_scales: Vec<f32>,
    weight_zero_points: Vec<f32>,
    layer_id: usize,
) -> (Vec<Vec<QuantizedWeightUnit>>, Vec<QuantizedMapping>) {
    //pre calculated values using quantize_layers_activation function for pytorch version of mobilenet v2
    let scales: Vec<f32> = vec![
        0.0, 0.017050628, 0.059874874, 0.023510717, 0.077991895, 0.023528527, 0.043043755, 0.041247565, 0.019669991, 0.04375145, 0.015492544, 0.030381273, 0.02179407, 0.009130553, 0.028741388, 0.014508855, 0.046173487, 0.021573933, 0.010297046, 0.02477803, 0.010213688, 0.024543108, 0.008275525, 0.0043514133, 0.015088752, 0.0066927955, 0.028775794, 0.008486401, 0.0041753906, 0.014366644, 0.0063380706, 0.031978507, 0.013272538, 0.006569647, 0.014843575, 0.0076167486, 0.021849547, 0.0061078398, 0.003288166, 0.010992496, 0.0044823377, 0.02312003, 0.00598988, 0.0029214076, 0.010844032, 0.005074731, 0.023845209, 0.006212397, 0.0028218033, 0.011636673, 0.0053011286, 0.024384094, 0.008112852, 0.004354821, 0.01604635, 0.009336259, 0.020551097, 0.007557358, 0.003906706, 0.015182976, 0.00761451, 0.022480486, 0.008598712, 0.0043971506, 0.018224234, 0.010677889, 0.029062292, 0.0105717825, 0.005258989, 0.014457449, 0.009047309, 0.015943345, 0.007378558, 0.0044045816, 0.012302028, 0.006023713, 0.018774118, 0.007028581, 0.003520944, 0.014610644, 0.0071549965, 0.02899628, 0.008812615, 0.0039076963, 0.013363283, 0.006123409, 0.0111313835, 0.06071932, 0.02352783, 0.10530223, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0
    ];
    let zero_points: Vec<f32> = vec![
        0.0, 117.85401, 92.58973, 0.0, 129.85863, 0.0, 125.33349, 133.0275, 0.0, 164.5871, 0.0, 122.431366, 148.30037, 0.0, 125.57285, 0.0, 131.13191, 133.78134, 0.0, 149.49226, 0.0, 129.0628, 121.11929, 0.0, 141.55995, 0.0, 125.722786, 129.80463, 0.0, 142.56842, 0.0, 124.34266, 129.16527, 0.0, 123.58052, 0.0, 129.39217, 118.154686, 0.0, 150.07324, 0.0, 126.26587, 130.5784, 0.0, 136.15375, 0.0, 123.68658, 138.8246, 0.0, 138.796, 0.0, 127.22833, 118.132904, 0.0, 107.2402, 0.0, 128.97179, 123.29459, 0.0, 130.02814, 0.0, 128.79446, 125.460526, 0.0, 106.95209, 0.0, 127.27601, 128.95506, 0.0, 98.22474, 0.0, 125.49445, 103.21538, 0.0, 131.31839, 0.0, 127.703354, 126.812325, 0.0, 131.31557, 0.0, 134.61806, 140.18808, 0.0, 136.28755, 0.0, 126.31436, 117.82339, 0.0, 83.416145, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0
    ];
    let res_scales: Vec<f32> = vec![
        0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.04237047, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.017695753, 0.0, 0.0, 0.0, 0.0, 0.018468888, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0162777, 0.0, 0.0, 0.0, 0.0, 0.009754247, 0.0, 0.0, 0.0, 0.0, 0.012484188, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.013122553, 0.0, 0.0, 0.0, 0.0, 0.02074431, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.011427514, 0.0, 0.0, 0.0, 0.0, 0.019429747, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0
    ];
    let res_zeros: Vec<f32> = vec![
        0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 147.16971, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 124.37584, 0.0, 0.0, 0.0, 0.0, 129.45807, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 119.91027, 0.0, 0.0, 0.0, 0.0, 129.03853, 0.0, 0.0, 0.0, 0.0, 129.12782, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 126.28977, 0.0, 0.0, 0.0, 0.0, 132.51447, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 124.45109, 0.0, 0.0, 0.0, 0.0, 142.59338, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0
    ];
    let mut m = 0.;
    let mut s1 = scales[layer_id];
    let s2 = weight_scales[layer_id];
    let mut s3 = scales[layer_id + 1];
    if zero_points[layer_id + 2] == 0. && scales[layer_id + 2] != 0. {
        s3 = scales[layer_id + 2];
    }
    // skip the relu6
    else if res_scales[layer_id] != 0.0 {
        //residual connection M  = S1 * S2 / S3
        s3 = res_scales[layer_id];
    }
    m = s1 * s2 / s3;
    let mut zero1 = zero_points[layer_id].round() as u8;
    let zero2 = weight_zero_points[layer_id];
    if zero2 == 0. {
        panic!("weights not get")
    }
    let mut zero3 = zero_points[layer_id + 1].round() as u8;
    if zero_points[layer_id + 2] == 0. && scales[layer_id + 2] != 0. {
        zero3 = zero_points[layer_id + 2].round() as u8;
    } else if res_scales[layer_id] != 0.0 {
        //residual connection M  = S1 * S2 / S3
        zero3 = res_zeros[layer_id].round() as u8;
    }
    let quant_weights = original_weights
        .into_iter()
        .map(|x| {
            x.into_iter()
                .map(|y| QuantizedWeightUnit {
                    data: y
                        .data
                        .into_iter()
                        .map(|i| (i / s2 + zero2.round()).round().clamp(0., 255.) as u8)
                        .collect(),
                    bias: (y.bias / (s1 * s2)).round() as i32,
                    which_kernel: y.which_kernel,
                    count: y.count,
                    start_pos_in: y.start_pos_in,
                    info: y.info,
                    zero_points: (zero1, zero2.round() as u8, zero3),
                    m: m,
                    s_out: s3,
                })
                .collect::<Vec<QuantizedWeightUnit>>()
        })
        .collect::<Vec<Vec<QuantizedWeightUnit>>>();
    let quant_mapping = original_mapping
        .into_iter()
        .map(|x| QuantizedMapping {
            count: x.count,
            map: x.map,
            padding_pos: x.padding_pos,
            end_pos: x.end_pos,
            zero_point: (zero1, zero2.round() as u8, zero3),
            scale: (s1, s2, s3),
        })
        .collect::<Vec<QuantizedMapping>>();
    (quant_weights, quant_mapping)
}
