use std::path::PathBuf;

use candle::{Device, Result, Shape, Tensor};

pub fn load_image<P: AsRef<std::path::Path>>(
    p: P,
    resize_longest: Option<usize>,
) -> Result<(Tensor, usize, usize)> {
    let img = image::io::Reader::open(p)?
        .decode()
        .map_err(candle::Error::wrap)?;
    let (initial_h, initial_w) = (img.height() as usize, img.width() as usize);
    let img = match resize_longest {
        None => img,
        Some(resize_longest) => {
            let (height, width) = (img.height(), img.width());
            let resize_longest = resize_longest as u32;
            let (height, width) = if height < width {
                let h = (resize_longest * height) / width;
                (h, resize_longest)
            } else {
                let w = (resize_longest * width) / height;
                (resize_longest, w)
            };
            img.resize_exact(width, height, image::imageops::FilterType::CatmullRom)
        }
    };
    let (height, width) = (img.height() as usize, img.width() as usize);
    println!("Image shape {:?}", (height, width));
    let img = img.to_rgb8();
    let data = img.into_raw();
    let data = Tensor::from_vec(data, (height, width, 3), &Device::Cpu)?.permute((2, 0, 1))?;
    Ok((data, initial_h, initial_w))
}

pub fn load_image_and_resize<P: AsRef<std::path::Path>>(
    p: P,
    width: usize,
    height: usize,
) -> Result<Tensor> {
    let img = image::io::Reader::open(p)?
        .decode()
        .map_err(candle::Error::wrap)?
        .resize_to_fill(
            width as u32,
            height as u32,
            image::imageops::FilterType::Triangle,
        );
    let img = img.to_rgb8();
    let data = img.into_raw();
    Tensor::from_vec(data, (width, height, 3), &Device::Cpu)?.permute((2, 0, 1))
}

pub struct CLIPVisionModel;
pub struct CLIPImageProcessor;
pub struct CLIPVisionConfig;

pub trait ImageProcessor {
    fn preprocess<P: AsRef<std::path::Path>>(path: P) -> Result<Tensor>;
}

impl ImageProcessor for CLIPImageProcessor {
    fn preprocess<P: AsRef<std::path::Path>>(path: P) -> Result<Tensor> {
        let (data, _, _) = load_image(path, None)?;
        Ok(data)
    }
}

pub struct CLIPVisionTower {
    vision_tower_name: String,
    select_layer: String,
    select_feature: String,
    image_processor: CLIPImageProcessor,
    vision_tower: CLIPVisionModel,
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn test_load_image() {
        let path = std::path::PathBuf::from("tests/fixtures/bike.jpg");
        let (data, _, _) = load_image(path, None).unwrap();
        println!("Tensor shape {:?}", data.shape());
        //assert_eq!(data.shape(), Shape::from(&[3, 640, 640]));
    }
}
