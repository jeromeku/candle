use std::path::PathBuf;

use candle::{Device, Result, Shape, Tensor};
use image::{DynamicImage, RgbImage};

pub enum ChannelDim {
    First,
    Last,
}
fn load_image<P: AsRef<std::path::Path>>(
    p: P,
    resize_longest: Option<usize>,
) -> Result<(DynamicImage, usize, usize)> {
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
    Ok((img, initial_h, initial_w))
}

pub fn convert_image_to_tensor(img: DynamicImage, channel_dim: ChannelDim) -> Result<Tensor> {
    let (height, width) = (img.height() as usize, img.width() as usize);
    let data = img.to_rgb8().into_raw();
    let data = match channel_dim {
        ChannelDim::First => Tensor::from_vec(data, (3, height, width), &Device::Cpu)?,
        ChannelDim::Last => Tensor::from_vec(data, (height, width, 3), &Device::Cpu)?,
    };

    Ok(data)
}

pub fn process_image<P: AsRef<std::path::Path>>(
    p: P,
    resize_longest: Option<usize>,
    channel_dim: Option<ChannelDim>,
) -> Result<(Tensor, usize, usize)> {
    let (img, initial_h, initial_w) = load_image(p, resize_longest)?;
    let channel_dim = channel_dim.unwrap_or(ChannelDim::First);
    let data = convert_image_to_tensor(img, channel_dim)?;
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
pub struct CLIPVisionConfig;

pub trait ImageProcessor {
    fn preprocess<P: AsRef<std::path::Path>>(
        path: P,
        channel_dim: Option<ChannelDim>,
    ) -> Result<Tensor>;
}

fn image_to_channel_dim(data: Tensor, channel_dim: ChannelDim) -> Result<Tensor> {
    match channel_dim {
        ChannelDim::First => Ok(data.permute((2, 0, 1))?),
        ChannelDim::Last => Ok(data),
    }
}
pub struct CLIPImageProcessor;
impl ImageProcessor for CLIPImageProcessor {
    fn preprocess<P: AsRef<std::path::Path>>(
        path: P,
        channel_dim: Option<ChannelDim>,
    ) -> Result<Tensor> {
        let (data, _, _) = process_image(path, None, channel_dim)?;

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
        let (_, h, w) = load_image(&path, None).unwrap();
        let channel_dim = ChannelDim::First;
        let tensor = CLIPImageProcessor::preprocess(&path, Some(channel_dim)).unwrap();
        println!("Tensor shape {:?}", tensor.shape());

        let expected_shape_channel_first = Shape::from(&[3, h, w]);
        assert_eq!(*tensor.shape(), expected_shape_channel_first);

        let channel_dim = ChannelDim::Last;
        let tensor = CLIPImageProcessor::preprocess(path, Some(channel_dim)).unwrap();
        println!("Tensor shape {:?}", tensor.shape());
        let expected_shape_channel_last = Shape::from(&[h, w, 3]);

        assert_eq!(*tensor.shape(), expected_shape_channel_last);
    }
}
