use maxpool2d::MaxPool2d;
use serde::{Deserialize, Serialize};

use std::fmt::Debug;
pub mod calculations;
pub mod decode;
pub mod operations;
pub mod util;
pub mod maxpool2d;

#[derive(Debug, Serialize, Deserialize)]
pub enum LayerWrapper {
    Convolution(Conv),
    Linear(Linear),
    BatchNorm2d(Batchnorm2d),
    ReLU6(Relu6),
    MaxPool2d(MaxPool2d)
}
#[derive(Debug, Serialize, Deserialize, Clone)]
pub enum InfoWrapper {
    Convolution(ConvMapping),
    Linear(LinearMapping),
    BatchNorm2d(Vec<i32>),
    ReLU6(Vec<i32>),
    MaxPool2d((i32, i32), i32, i32) //TODO better data format
}
#[derive(Debug, Serialize, Deserialize, Clone)]
pub struct WeightUnit {
    pub data: Vec<f32>,
    pub bias: f32,
    pub which_kernel: u16,
    pub count: i32,
    pub start_pos_in: Vec<i32>,
    pub info: InfoWrapper,
}
#[derive(Clone, Serialize, Deserialize)]
pub struct Mapping {
    pub count: Vec<u32>,
    pub map: Vec<Vec<u8>>, // from which node,to which node
    // pub channel: Vec<u16>,            //used for batch norm,deleted after fusion with convolution,24/2/29
    pub padding_pos: Vec<Vec<u32>>, //padding counts, when reached, should give 0
    pub end_pos: Vec<(u16, u8, u32)>, //phase,next_mcu,count
}
#[derive(Debug, Serialize, Deserialize, Clone)]
pub struct QuantizedWeightUnit {
    pub data: Vec<u8>,
    pub bias: i32,
    pub which_kernel: u16,
    pub count: i32,
    pub start_pos_in: Vec<i32>,
    pub info: InfoWrapper,
    pub zero_points: (u8, u8, u8),
    pub m: f32,
    pub s_out: f32,
}
#[derive(Debug, Serialize, Deserialize, Clone)]
pub struct QuantizedMapping {
    pub count: Vec<u32>,
    pub map: Vec<Vec<u8>>, // from which node,to which node
    // pub channel: Vec<u16>,            //used for batch norm,deleted after fusion with convolution,24/2/29
    pub padding_pos: Vec<Vec<u32>>, //padding counts, when reached, should give 0
    pub end_pos: Vec<(u16, u8, u32)>, //phase,next_mcu,count
    pub zero_point: (u8, u8, u8),
    pub scale: (f32, f32, f32), //todo! convert this to u32
}

///Layer trait, that each object is associated with
pub trait Layer {
    /// Returns the layer type
    fn identify(&self) -> &str;
    /// Find the positions of the input neurons by tracing back the receptive field
    fn get_input(&self, position: Vec<i32>) -> Vec<Vec<i32>>;
    // fn get_weight(&self,position:Vec<i32>) -> f32;
    /// Returns the output dimension of the layer
    fn get_output_shape(&self) -> Vec<i32>;
    /// Layer specific information, wrapped inside [`InfoWrapper`], with the exact variant based
    /// on the layer type
    fn get_info(&self) -> InfoWrapper;
    /// Returns the bias used to calculate the result
    fn get_bias(&self, p: i32) -> f32;
    /// Print all the layer information
    fn get_all(&self) -> &dyn Debug;
    /// Prints shape of the weights of the layer
    fn print_weights_shape(&self);
    /// Returns sample weight data using the input positions and output channel
    fn get_weights_from_input(&self, input: Vec<Vec<i32>>, c: i32) -> Vec<f32>;
    /// Only applicable to the functional layer - perform forward propagation
    fn functional_forward(
        &self,
        input: &mut Vec<Vec<Vec<f32>>>,
    ) -> Result<&'static str, &'static str>;
    /// Return the layer's weights
    fn get_weights(&self) -> Vec<f32>;
    /// Returns the layer specific information wrapped inside [`InfoWrapper`] without padding
    fn get_info_no_padding(&self) -> InfoWrapper;
}
#[derive(Debug, Serialize, Deserialize)]
pub struct Conv {
    pub w: Vec<Vec<Vec<Vec<f32>>>>,
    pub info: ConvMapping,
    pub bias: Vec<f32>,
}

#[derive(Debug, Serialize, Deserialize, Clone)]
pub struct ConvMapping {
    pub o_pg: i32,
    pub i_pg: i32,
    pub s: (i32, i32),
    pub k: (i32, i32),
    pub i: (i32, i32, i32),
    pub o: (i32, i32, i32),
}

#[derive(Debug, Serialize, Deserialize)]
pub struct Linear {
    pub w: Vec<Vec<f32>>,
    pub info: LinearMapping,
    pub bias: Vec<f32>,
}

#[derive(Debug, Serialize, Deserialize, Clone)]
pub struct LinearMapping {
    pub b_in: i32,
    pub c_in: i32,
    pub b_out: i32,
    pub c_out: i32,
}
#[derive(Debug, Serialize, Deserialize)]
pub struct Batchnorm2d {
    w: Vec<f32>,
    bias: Vec<f32>,
    r_m: Vec<f32>,
    r_v: Vec<f32>,
    input_shape: Vec<i32>,
}
#[derive(Debug, Serialize, Deserialize)]
pub struct Relu6 {
    pub input_shape: Vec<i32>,
}
pub trait IOMapping {
    fn map_to_input(&self, o_position: Vec<i32>) -> Vec<Vec<i32>>;
}

impl IOMapping for ConvMapping {
    fn map_to_input(&self, o_position: Vec<i32>) -> Vec<Vec<i32>> {
        assert_eq!(o_position.len(), 3);
        let h_offset = &o_position[1] * &self.s.0;
        let w_offset = &o_position[2] * &self.s.1;
        let which_group = (&o_position[0] / &self.o_pg) * &self.i_pg;
        let mut result: Vec<Vec<i32>> = Vec::new();
        for q in 0..self.i_pg {
            for h in -self.k.0 / 2..=self.k.0 / 2 {
                for w in -self.k.1 / 2..=self.k.1 / 2 {
                    result.push(vec![&which_group + &q, &h_offset + &h, &w_offset + w]);
                }
            }
        }
        result
    }
}

impl IOMapping for LinearMapping {
    fn map_to_input(&self, o_position: Vec<i32>) -> Vec<Vec<i32>> {
        assert_eq!(o_position.len(), 2);
        let mut result: Vec<Vec<i32>> = Vec::new();
        for i in 0..self.c_in {
            result.push(vec![o_position[0], i]);
        }
        result
    }
}
///Convolution layer implementation
impl Layer for Conv {
    fn identify(&self) -> &str {
        "Convolution"
    }

    // fn get_weight(&self, position: Vec<i32>) -> f32 {
    //     // Implement your logic to get the weight based on position
    //     // For example, you might want to access self.w with the given position
    //     assert_eq!(position.len(), 4);
    //
    //     let r = (position[0], position[1], position[2], position[3]);
    //
    //     // Directly index into the vector without cloning
    //     self.w[r.0 as usize][r.1 as usize][r.2 as usize][r.3 as usize]
    // }

    fn get_input(&self, position: Vec<i32>) -> Vec<Vec<i32>> {
        self.info.map_to_input(position)
    }

    fn get_output_shape(&self) -> Vec<i32> {
        let mut reuslt = Vec::new();
        reuslt.push(self.info.o.0);
        reuslt.push(self.info.o.1);
        reuslt.push(self.info.o.2);
        reuslt
    }

    /// Returns the information specific to the convolution layer as InfoWrapper::Convolution
    /// The data returned is of the type [`ConvMapping`]
    fn get_info(&self) -> InfoWrapper {
        let mut padded_input = self.info.i;
        if self.info.k.0 > 1 && self.info.k.1 > 1 {
            if (self.info.i.2 - 1) % self.info.s.0 == (self.info.k.0 / 2) {
                padded_input.2 += self.info.k.1 / 2;
                padded_input.1 += self.info.k.0 / 2;
            } else {
                padded_input.2 += self.info.k.1 / 2 * 2;
                padded_input.1 += self.info.k.0 / 2 * 2;
            }
        }
        InfoWrapper::Convolution(ConvMapping {
            o_pg: self.info.o_pg,
            i_pg: self.info.i_pg,
            s: self.info.s,
            k: self.info.k,
            i: padded_input,
            o: self.info.o,
        })
    }

    fn get_bias(&self, i: i32) -> f32 {
        if self.bias.is_empty() {
            return 0.;
        }
        self.bias[i as usize]
    }

    fn get_all(&self) -> &dyn Debug {
        self
    }

    fn print_weights_shape(&self) {
        println!(
            "weight shape:{:?},{:?},{:?},{:?}",
            self.w.len(),
            self.w[0].len(),
            self.w[0][0].len(),
            self.w[0][0][0].len()
        );
    }

    fn get_weights_from_input(&self, input: Vec<Vec<i32>>, output_channel: i32) -> Vec<f32> {
        let mut result = Vec::new();
        for i in 0..input.len() {
            let col = i % self.info.k.1 as usize;
            let row = (i / self.info.k.1 as usize) % self.info.k.0 as usize;
            let c = input[i][0] % self.info.i_pg;
            result.push(self.w[output_channel as usize][c as usize][row][col]);
        }
        result
    }

    fn functional_forward(
        &self,
        _input: &mut Vec<Vec<Vec<f32>>>,
    ) -> Result<&'static str, &'static str> {
        Err("This is a convolutional layer, not a functional layer")
    }

    fn get_weights(&self) -> Vec<f32> {
        self.w
            .clone()
            .into_iter()
            .flat_map(|level1| {
                level1
                    .into_iter()
                    .flat_map(|level2| level2.into_iter().flatten())
            })
            .collect::<Vec<f32>>()
    }

    fn get_info_no_padding(&self) -> InfoWrapper {
        InfoWrapper::Convolution(ConvMapping {
            o_pg: self.info.o_pg,
            i_pg: self.info.i_pg,
            s: self.info.s,
            k: self.info.k,
            i: self.info.i,
            o: self.info.o,
        })
    }
}

impl Layer for Linear {
    fn identify(&self) -> &str {
        "Linear"
    }

    // fn get_weight(&self, position: Vec<i32>) -> f32 {
    //     // Implement your logic to get the weight based on position
    //     // For example, you might want to access self.w with the given position
    //     assert_eq!(position.len(), 2);
    //     let r = (position[0] as usize, position[1] as usize);
    //     self.w[r.0][r.1]
    // }

    fn get_input(&self, position: Vec<i32>) -> Vec<Vec<i32>> {
        self.info.map_to_input(position)
    }

    fn get_output_shape(&self) -> Vec<i32> {
        let mut reuslt = Vec::new();
        reuslt.push(self.info.b_out);
        reuslt.push(self.info.c_out);
        reuslt
    }

    fn get_info(&self) -> InfoWrapper {
        InfoWrapper::Linear(LinearMapping {
            b_in: self.info.b_in,
            c_in: self.info.c_in,
            b_out: self.info.b_out,
            c_out: self.info.c_out,
        })
    }

    fn get_bias(&self, p: i32) -> f32 {
        if self.bias.is_empty() {
            return 0.;
        }
        self.bias[p as usize]
    }

    fn get_all(&self) -> &dyn Debug {
        self
    }

    fn print_weights_shape(&self) {
        println!("Weight shape:{:?},{:?}", self.w.len(), self.w[0].len());
    }

    fn get_weights_from_input(&self, input: Vec<Vec<i32>>, p: i32) -> Vec<f32> {
        let mut result: Vec<f32> = Vec::new();
        for i in 0..input.len() {
            result.push(self.w[p as usize][input[i][1] as usize]);
        }
        result
    }

    fn functional_forward(
        &self,
        _input: &mut Vec<Vec<Vec<f32>>>,
    ) -> Result<&'static str, &'static str> {
        Err("This is a Linear layer, not a functional layer")
    }

    fn get_weights(&self) -> Vec<f32> {
        self.w.clone().into_iter().flatten().collect()
    }

    fn get_info_no_padding(&self) -> InfoWrapper {
        InfoWrapper::Linear(LinearMapping {
            b_in: self.info.b_in,
            c_in: self.info.c_in,
            b_out: self.info.b_out,
            c_out: self.info.c_out,
        })
    }
}

impl Layer for Batchnorm2d {
    fn identify(&self) -> &str {
        "Batchnorm2d"
    }

    fn get_input(&self, position: Vec<i32>) -> Vec<Vec<i32>> {
        vec![position]
    }

    fn get_output_shape(&self) -> Vec<i32> {
        let mut s = self.input_shape.clone();
        //remove the batch dimension
        s.remove(0);
        s
    }

    fn get_info(&self) -> InfoWrapper {
        InfoWrapper::BatchNorm2d(self.input_shape.clone())
    }

    fn get_bias(&self, p: i32) -> f32 {
        self.bias[p as usize]
    }

    fn get_all(&self) -> &dyn Debug {
        self
    }

    fn print_weights_shape(&self) {
        println!("Input shape : {:?}", self.input_shape)
    }
    //assuming the input starts with channel, ie (c,h,w)
    fn get_weights_from_input(&self, input: Vec<Vec<i32>>, c: i32) -> Vec<f32> {
        let mut result = Vec::new();
        for _i in 0..input.len() {
            result.push(self.r_m[c as usize]);
            result.push(self.r_v[c as usize]);
            result.push(self.w[c as usize]);
            result.push(self.bias[c as usize]);
        }
        result
    }

    fn functional_forward(
        &self,
        input: &mut Vec<Vec<Vec<f32>>>,
    ) -> Result<&'static str, &'static str> {
        let c = input.len();
        let h = input[0].len();
        let w = input[0][0].len();
        for i in 0..c {
            // Update elements using batch normalization
            for j in 0..h {
                for k in 0..w {
                    input[i][j][k] = (input[i][j][k] - self.r_m[i]) / (self.r_v[i] + 1e-5).sqrt()
                        * self.w[i]
                        + self.bias[i];
                }
            }
        }

        Ok("finished")
    }

    fn get_weights(&self) -> Vec<f32> {
        [
            self.r_m.clone(),
            self.r_v.clone(),
            self.w.clone(),
            self.bias.clone(),
        ]
        .concat()
    }

    fn get_info_no_padding(&self) -> InfoWrapper {
        InfoWrapper::BatchNorm2d(self.input_shape.clone())
    }
}

impl Layer for Relu6 {
    fn identify(&self) -> &str {
        "Relu6"
    }

    fn get_input(&self, position: Vec<i32>) -> Vec<Vec<i32>> {
        vec![position]
    }

    fn get_output_shape(&self) -> Vec<i32> {
        let mut s = self.input_shape.clone();
        //remove the batch dimension
        s.remove(0);
        s
    }

    fn get_info(&self) -> InfoWrapper {
        InfoWrapper::ReLU6(self.input_shape.clone())
    }

    fn get_bias(&self, _p: i32) -> f32 {
        0.0
    }

    fn get_all(&self) -> &dyn Debug {
        self
    }

    fn print_weights_shape(&self) {
        println!("Input shape: {:?}", self.input_shape)
    }

    fn get_weights_from_input(&self, _input: Vec<Vec<i32>>, _c: i32) -> Vec<f32> {
        vec![0.0]
    }

    fn functional_forward(
        &self,
        input: &mut Vec<Vec<Vec<f32>>>,
    ) -> Result<&'static str, &'static str> {
        for i in 0..input.len() {
            for j in 0..input[0].len() {
                for k in 0..input[0][0].len() {
                    if input[i][j][k] < 0.0 {
                        input[i][j][k] = 0.;
                    } else if input[i][j][k] >= 6.0 {
                        input[i][j][k] = 6.0;
                    }
                }
            }
        }
        Ok("finished")
    }

    fn get_weights(&self) -> Vec<f32> {
        vec![]
    }

    fn get_info_no_padding(&self) -> InfoWrapper {
        InfoWrapper::ReLU6(self.input_shape.clone())
    }
}
