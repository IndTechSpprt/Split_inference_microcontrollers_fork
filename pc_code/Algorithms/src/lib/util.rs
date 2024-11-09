pub fn sample_input_from_p_zero_padding(p: Vec<Vec<i32>>, input: &Vec<Vec<Vec<f32>>>) -> Vec<f32> {
    let mut result = Vec::new();
    for i in 0..p.len() {
        let a = &p[i];
        if a[0] < 0
            || a[1] < 0
            || a[2] < 0
            || a[0] >= input.len() as i32
            || a[1] >= input[0].len() as i32
            || a[2] >= input[0][0].len() as i32
        {
            result.push(0.);
        } else {
            result.push(input[a[0] as usize][a[1] as usize][a[2] as usize]);
        }
    }
    result
}
pub fn sample_input_linear(p: Vec<Vec<i32>>, input: &Vec<Vec<f32>>) -> Vec<f32> {
    let mut result = Vec::new();
    for i in 0..p.len() {
        let a = &p[i];
        result.push(input[a[0] as usize][a[1] as usize]);
    }
    result
}
pub fn split_u128_to_u8(number: u128) -> Vec<u8> {
    let mut result = vec![0; 16];

    // Iterate over each 8-bit chunk
    for i in 0..16 {
        let shift = i * 8;
        let mut chunk = ((number >> shift) & 0xFF) as u8;
        if i == 15 && chunk >> 7 & 0b1 == 1 {
            chunk &= 0x7F;
        } // padding pos
        result[i] = chunk;
    }

    result
}
extern crate image;

use image::{DynamicImage, GenericImageView, Rgba};

pub fn read_and_store_image(file_path: &str) -> Option<Vec<Vec<Vec<u8>>>> {
    // Attempt to open the image file
    if let Ok(img) = image::open(file_path) {
        // Resize and center crop the image to 224x224
        let mut resized_img = img.resize_exact(147, 147, image::imageops::FilterType::Triangle);
        let cropped_img = resized_img.crop(19, 19, 128, 128);

        // Convert the image to RGB format
        let rgb_img = cropped_img.to_rgb8();

        // Create a nested vector [3, 224, 224]
        let mut result: Vec<Vec<Vec<u8>>> = vec![vec![vec![0; 128]; 128]; 3];

        // Iterate over the pixels and store them in the nested vector
        for (y, row) in rgb_img.enumerate_rows() {
            for (x, pixel) in row.enumerate() {
                result[0][y as usize][x] = pixel.2[0];
                result[1][y as usize][x] = pixel.2[1];
                result[2][y as usize][x] = pixel.2[2];
            }
        }

        Some(result)
    } else {
        None
    }
}
pub fn pre_processing(image: Vec<Vec<Vec<u8>>>) -> Vec<Vec<Vec<f32>>> {
    let mean = [0.485, 0.456, 0.406];
    let std = [0.229, 0.224, 0.225];
    assert!(image.len() == 3 && image[0].len() == 128 && image[0][0].len() == 128);
    // Convert the RGB values to f32, normalize, subtract mean, and scale by std
    let mut normalized_image: Vec<Vec<Vec<f32>>> = vec![vec![vec![0.; 128]; 128]; 3];
    for i in 0..3 {
        for j in 0..128 {
            for k in 0..128 {
                normalized_image[i][j][k] = (image[i][j][k] as f32 / 255. - mean[i]) / std[i];
            }
        }
    }
    normalized_image
}
