use anyhow::Result;
use candle::{DType, Device, Result as CandleResult, Shape, Tensor};
use candle_nn::{Module, VarBuilder};
use tracing_appender::non_blocking::WorkerGuard;
use tracing_appender::rolling::{RollingFileAppender, Rotation};
use tracing_subscriber::layer::SubscriberExt;
use tracing_subscriber::{filter, prelude::*};

pub fn init_tracing(out_dir: Option<&str>, filename: &str) -> WorkerGuard {
    let file_appender = RollingFileAppender::new(
        Rotation::DAILY,
        out_dir.unwrap_or("tracing"),
        format!("{}.json", filename.split('/').last().unwrap_or(filename)),
    );
    let (non_blocking, _guard) = tracing_appender::non_blocking(file_appender);

    tracing_subscriber::registry()
        // first layer for console output, use pretty formatter and level filter
        .with(
            tracing_subscriber::fmt::layer()
                .pretty()
                .with_filter(filter::LevelFilter::from(tracing::Level::DEBUG)),
        )
        // second layer for log file appender, use json formatter, no filter
        .with(
            tracing_subscriber::fmt::layer()
                .json()
                .with_writer(non_blocking)
                .with_filter(filter::LevelFilter::from(tracing::Level::DEBUG)),
        )
        .init();
    // tracing::subscriber::set_global_default(subscriber).expect("Unable to set a global subscriber");
    _guard
}

#[cfg(test)]
mod tests {
    use std::path::Path;

    use super::*;
    use test_context::{test_context, TestContext};
    struct TraceTestContext {
        out_dir: &'static str,
    }

    struct MistralTestContext {
        model_id: &'static str,
        filename: &'static str,
        trace_ctx: TraceTestContext,
    }

    impl TestContext for TraceTestContext {
        fn setup() -> TraceTestContext {
            let out_dir = "trace_test";
            if Path::new(out_dir).exists() {
                std::fs::remove_dir_all(out_dir).unwrap();
            }

            TraceTestContext { out_dir }
        }
        fn teardown(self) {
            std::fs::remove_dir_all(self.out_dir).unwrap();
        }
    }

    impl TestContext for MistralTestContext {
        fn setup() -> MistralTestContext {
            // let out_dir = "mistral_trace";
            let model_id = "lmz/candle-mistral";
            let filename = model_id.split('/').last().unwrap_or(model_id);
            let trace_ctx = TraceTestContext::setup();

            MistralTestContext {
                model_id,
                filename,
                trace_ctx,
            }
        }
        fn teardown(self) {
            self.trace_ctx.teardown();
        }
    }

    struct DinoTestContext {
        image_path: &'static str,
        model_id: &'static str,
        filename: &'static str,
        trace_ctx: TraceTestContext,
    }
    impl TestContext for DinoTestContext {
        fn setup() -> DinoTestContext {
            let image_path = "tests/fixtures/bike.jpg";
            let model_id = "lmz/candle-dino-v2";
            let filename = model_id.split('/').last().unwrap_or(model_id);
            let trace_ctx = TraceTestContext::setup();

            DinoTestContext {
                image_path,
                model_id,
                filename,
                trace_ctx,
            }
        }

        fn teardown(self) {
            self.trace_ctx.teardown();
        }
    }

    pub fn load_image224<P: AsRef<std::path::Path>>(p: P) -> CandleResult<Tensor> {
        let img = image::io::Reader::open(p)?
            .decode()
            .map_err(candle::Error::wrap)?
            .resize_to_fill(224, 224, image::imageops::FilterType::Triangle);
        let img = img.to_rgb8();
        let data = img.into_raw();
        let data = Tensor::from_vec(data, (224, 224, 3), &Device::Cpu)?.permute((2, 0, 1))?;
        let mean = Tensor::new(&[0.485f32, 0.456, 0.406], &Device::Cpu)?.reshape((3, 1, 1))?;
        let std = Tensor::new(&[0.229f32, 0.224, 0.225], &Device::Cpu)?.reshape((3, 1, 1))?;
        (data.to_dtype(DType::F32)? / 255.)?
            .broadcast_sub(&mean)?
            .broadcast_div(&std)
    }

    #[test_context(DinoTestContext)]
    #[test]
    fn test_tracing_small_model(ctx: &mut DinoTestContext) -> Result<()> {
        use candle_transformers::models::dinov2;

        let out_dir = ctx.trace_ctx.out_dir;
        let filename = ctx.filename;
        let _guard = init_tracing(Some(out_dir), filename);

        let img_path = ctx.image_path;
        let image = load_image224(img_path)?;
        let device = Device::Cpu;
        let dtype = DType::F32;

        let vb_debug = VarBuilder::debug(dtype, &device);
        let model = dinov2::vit_small(vb_debug)?;
        _ = model.forward(&image.unsqueeze(0)?)?;

        assert!(Path::new(out_dir).exists());
        let mut files: Vec<std::path::PathBuf> = std::fs::read_dir(out_dir)
            .unwrap()
            .map(|res| res.unwrap().path())
            .collect();
        assert!(!files.is_empty());

        //Find most recent file
        files.sort_by_key(|file| file.metadata().unwrap().modified().unwrap());
        let trace_file = files.last().unwrap();
        assert!(trace_file.to_str().unwrap().contains(filename));

        Ok(())
    }

    #[test_context(MistralTestContext)]
    #[test]
    fn test_tracing_mistral(ctx: &mut MistralTestContext) -> Result<()> {
        use crate::utilities::model_loading::{MistralModelLoader, ModelLoader};
        use candle_transformers::models::mistral::Config as MistralConfig;

        let model_id = ctx.model_id;
        let out_dir = ctx.trace_ctx.out_dir;
        let filename = ctx.filename;
        //   std::fs::remove_dir_all(out_dir)?;

        //Initialized tracing -- **must** call before loading model
        let _guard = init_tracing(Some(out_dir), filename);

        const BATCH_SIZE: usize = 1;
        const SEQ_LEN: usize = 5;

        let cfg = MistralConfig::config_7b_v0_1(false);
        let device = Device::cuda_if_available(0).unwrap_or(Device::Cpu);

        let mut model = MistralModelLoader::load_model(model_id, None, cfg, &device, Some(true))?;

        let shape = [BATCH_SIZE, SEQ_LEN];
        let input_shape = Shape::from(&shape);
        let input_ids = (1..(SEQ_LEN + 1) as u32)
            .cycle()
            .take(BATCH_SIZE * SEQ_LEN)
            .collect::<Vec<_>>();

        let input_tensors = Tensor::from_vec(input_ids, input_shape, &device)?;
        let seqlen_offset = 0;
        _ = model.forward(&input_tensors, seqlen_offset)?;

        assert!(Path::new(out_dir).exists());
        let mut files: Vec<std::path::PathBuf> = std::fs::read_dir(out_dir)
            .unwrap()
            .map(|res| res.unwrap().path())
            .collect();
        assert!(!files.is_empty());

        //Find most recent file
        files.sort_by_key(|file| file.metadata().unwrap().modified().unwrap());
        let trace_file = files.last().unwrap();
        assert!(trace_file.to_str().unwrap().contains(filename));

        Ok(())
    }
}
