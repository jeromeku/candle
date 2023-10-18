use anyhow::Result;
use candle_transformers::models::mistral::{Config as MistralConfig, Model as MistralModel};
use candle_transformers::models::quantized_mistral::{
    Config as QMistralConfig, Model as QMistralModel,
};
use candle_transformers::quantized_var_builder::VarBuilder as QVarBuilder;
use std::path::PathBuf;

use candle::{DType, Device, Result as CandleResult, Shape, Tensor};
use candle_nn::VarBuilder;

use hf_hub::api::sync::{Api, ApiRepo};
use hf_hub::{Repo, RepoType};
pub enum PretrainedFormat {
    GGuf,
    SafeTensors,
}

impl PretrainedFormat {
    fn as_str(&self) -> &'static str {
        match self {
            PretrainedFormat::GGuf => "gguf",
            PretrainedFormat::SafeTensors => "safetensors",
        }
    }
}

pub trait PretrainedModel {
    type ConfigType;
    const IS_QUANTIZED: bool;
    const PRETRAINED_FORMAT: PretrainedFormat;

    fn from_pretrained(
        cfg: &Self::ConfigType,
        filenames: Option<Vec<PathBuf>>,
        dtype: DType,
        device: &Device,
        trace: bool,
    ) -> CandleResult<Self>
    where
        Self: Sized;
}

pub trait ModelLoader
where
    Self::ModelType: PretrainedModel,
{
    type ModelType;

    fn load_from_hub(model_id: &str, revision: Option<&str>) -> Result<ApiRepo> {
        let api = Api::new()?;

        let repo = api.repo(Repo::with_revision(
            model_id.to_string(),
            RepoType::Model,
            revision.map_or("main".to_string(), |s| s.to_string()),
        ));
        Ok(repo)
    }
    fn load_model_files(model_id: &str, revision: Option<&str>) -> Result<Vec<PathBuf>> {
        let repo = Self::load_from_hub(model_id, revision)?;
        let repo_info = repo.info()?;
        let mut filenames = Vec::new();
        let model_format = Self::ModelType::PRETRAINED_FORMAT.as_str();
        for f in repo_info.siblings {
            if f.rfilename.ends_with(model_format) {
                filenames.push(repo.get(&f.rfilename)?);
            }
        }
        Ok(filenames)
    }

    fn load_model(
        model_id: &str,
        revision: Option<&str>,
        cfg: <<Self as ModelLoader>::ModelType as PretrainedModel>::ConfigType,
        // quantized: bool,
        device: &Device,
        trace: Option<bool>,
    ) -> Result<Self::ModelType> {
        let trace = trace.unwrap_or(false);
        let filenames = match trace {
            true => None,
            false => Some(Self::load_model_files(model_id, revision)?),
        };

        let dtype = if device.is_cuda() {
            DType::BF16
        } else {
            DType::F32
        };
        let model = Self::ModelType::from_pretrained(&cfg, filenames, dtype, device, trace)?;
        Ok(model)
    }
}

impl PretrainedModel for MistralModel {
    type ConfigType = MistralConfig;
    const IS_QUANTIZED: bool = false;
    const PRETRAINED_FORMAT: PretrainedFormat = PretrainedFormat::SafeTensors;
    fn from_pretrained(
        cfg: &Self::ConfigType,
        filenames: Option<Vec<PathBuf>>,
        dtype: DType,
        device: &Device,
        trace: bool,
    ) -> CandleResult<Self>
    where
        Self: Sized,
    {
        if trace {
            println!("Using Debugging VarBuilder");
        }
        let vb = match trace {
            true => VarBuilder::debug(dtype, device),
            false => unsafe {
                VarBuilder::from_mmaped_safetensors(&filenames.unwrap(), dtype, device)?
            },
        };

        Self::new(cfg, vb)
    }
}
impl PretrainedModel for QMistralModel {
    type ConfigType = QMistralConfig;
    const IS_QUANTIZED: bool = true;
    const PRETRAINED_FORMAT: PretrainedFormat = PretrainedFormat::GGuf;

    fn from_pretrained(
        cfg: &Self::ConfigType,
        filenames: Option<Vec<PathBuf>>,
        dtype: DType,
        device: &Device,
        trace: bool,
    ) -> CandleResult<Self>
    where
        Self: Sized,
    {
        // if trace {
        //     Self::new(cfg, VarBuilder::debug(dtype, device))
        // } else {
        let filenames = filenames.unwrap();
        assert!(
            filenames.len() == 1,
            "Only single file GGML quantized models are supported, found {:?}",
            filenames
                .iter()
                .map(|p| p.file_name().unwrap())
                .collect::<Vec<_>>()
        );
        let gguf_file = filenames[0].as_path();
        println!(
            "Downloading quantized model: {:?}",
            gguf_file.file_name().unwrap()
        );
        // let vb = match trace {
        //     true => VarBuilder::debug(dtype, device),
        //     false => QVarBuilder::from_gguf(gguf_file)?,
        // };
        let vb = QVarBuilder::from_gguf(gguf_file)?;
        Self::new(cfg, vb)
        // }
    }
}
pub struct MistralModelLoader;

impl ModelLoader for MistralModelLoader {
    type ModelType = MistralModel;
}

pub struct QMistralModelLoader;
impl ModelLoader for QMistralModelLoader {
    type ModelType = QMistralModel;
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn test_load_model_files() -> Result<()> {
        use std::collections::HashSet;
        let model_id = "lmz/candle-mistral";
        let expected_files: HashSet<&str> = HashSet::from_iter(
            [
                "pytorch_model-00001-of-00002.safetensors",
                "pytorch_model-00002-of-00002.safetensors",
            ]
            .iter()
            .cloned(),
        );

        let filenames = MistralModelLoader::load_model_files(model_id, None)?;
        let actual_files = filenames
            .iter()
            .map(|p| p.file_name().unwrap())
            .map(|s| s.to_str().unwrap())
            .collect::<HashSet<_>>();
        assert_eq!(actual_files, expected_files);
        println!("Files: {:?}", filenames);
        Ok(())
    }

    #[test]
    fn test_model_loading() -> Result<()> {
        Ok(())
    }
}
