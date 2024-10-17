use std::collections::HashMap;

use crate::{Layer, LayerWrapper};
use std::fs::File;
use std::io::Read;

pub fn decode_json(mut file: File) -> HashMap<i32, Box<dyn Layer>> {
    // Read the JSON file into a string
    let mut json_string = String::new();
    file.read_to_string(&mut json_string)
        .expect("Failed to read file");

    // Deserialize the JSON string into a HashMap<String, LayerWrapper>
    let mapping: HashMap<i32, LayerWrapper> =
        serde_json::from_str(&json_string).expect("Failed to deserialize JSON");

    // Convert LayerWrapper to Box<dyn Layer>
    let converted_mapping: HashMap<i32, Box<dyn Layer>> = mapping
        .into_iter()
        .map(|(key, value)| match value {
            LayerWrapper::Convolution(conv) => (key, Box::new(conv) as Box<dyn Layer>),
            LayerWrapper::Linear(linear) => (key, Box::new(linear) as Box<dyn Layer>),
            LayerWrapper::BatchNorm2d(norm) => (key, Box::new(norm) as Box<dyn Layer>),
            LayerWrapper::ReLU6(relu) => (key, Box::new(relu) as Box<dyn Layer>),
            LayerWrapper::MaxPool2d(max_pool2d) => (key, Box::new(max_pool2d) as Box<dyn Layer>),
        })
        .collect();
    converted_mapping
}
