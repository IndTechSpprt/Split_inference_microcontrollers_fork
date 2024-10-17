use std::vec;

use serde::{Deserialize, Serialize};

use crate::Layer;

#[derive(Debug, Serialize, Deserialize)]
//TODO do we need any other fields? (Input and output dimensions mabe be of help for debugging)
pub struct MaxPool2d {
    pub s: i32, //stride
    pub k: i32, //kernel
    pub d: i32  //dilation
}

impl Layer for MaxPool2d {
    fn identify(&self) -> &str {
        "MaxPool2d"
    }

    fn get_input(&self, position: Vec<i32>) -> Vec<Vec<i32>> {
        vec![position]
    }

    fn get_output_shape(&self) -> Vec<i32> {
        vec![]
    }

    fn get_info(&self) -> crate::InfoWrapper {
        crate::InfoWrapper::MaxPool2d((self.k, 1), self.s, self.d)
    }

    fn get_bias(&self, _p: i32) -> f32 {
        0.0
    }

    fn get_all(&self) -> &dyn std::fmt::Debug {
        self
    }

    fn print_weights_shape(&self) {
        println!("No weights!")
    }

    fn get_weights_from_input(&self, _input: Vec<Vec<i32>>, _c: i32) -> Vec<f32> {
        vec![0.0]
    }

    fn functional_forward(
        &self,
        _input: &mut Vec<Vec<Vec<f32>>>,
    ) -> Result<&'static str, &'static str> {
        todo!()
    }

    fn get_weights(&self) -> Vec<f32> {
        vec![]
    }

    fn get_info_no_padding(&self) -> crate::InfoWrapper {
        crate::InfoWrapper::MaxPool2d((self.k, 1), self.s, self.d)
    }
}