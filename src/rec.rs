#[cfg(feature = "fast_image_resize")]
use fast_image_resize::Resizer;
use image::{DynamicImage, GenericImageView};
use mnn::{BackendConfig, ForwardType, Interpreter, PowerMode, PrecisionMode, ScheduleConfig};
use ndarray::{Array, ArrayBase, Dim, OwnedRepr};
use std::{borrow::Cow, path::Path};
use tracing::{debug, error, info, trace, warn};

use crate::error::OcrResult;

/// 文本识别模型
///
/// Text recognition model that identifies characters in text images
pub struct Rec {
    interpreter: Interpreter,
    session: Option<mnn::Session>,
    keys: Vec<char>,
    min_score: f32,
    punct_min_score: f32,
    #[cfg(feature = "fast_resize")]
    resizer: fast_image_resize::Resizer,
}

impl Rec {
    const MIN_SCORE_DEFAULT: f32 = 0.6;
    const PUNCT_MIN_SCORE_DEFAULT: f32 = 0.1;

    const PUNCTUATIONS: [char; 49] = [
        ',', '.', '!', '?', ';', ':', '"', '\'', '(', ')', '[', ']', '{', '}', '-', '_', '/', '\\',
        '|', '@', '#', '$', '%', '&', '*', '+', '=', '~', '，', '。', '！', '？', '；', '：', '、',
        '「', '」', '『', '』', '（', '）', '【', '】', '《', '》', '—', '…', '·', '～',
    ];

    /// 创建新的文本识别器实例
    ///
    /// Create a new text recognizer instance
    pub fn new(interpreter: Interpreter, keys: Vec<char>) -> Self {
        Self {
            interpreter,
            session: None,
            keys,
            min_score: Self::MIN_SCORE_DEFAULT,
            punct_min_score: Self::PUNCT_MIN_SCORE_DEFAULT,
            #[cfg(feature = "fast_resize")]
            resizer: fast_image_resize::Resizer::new(),
        }
    }

    /// 从模型文件和字符集文件创建文本识别器
    ///
    /// Create a text recognizer from model file and character set file
    pub fn from_file(model_path: impl AsRef<Path>, keys_path: impl AsRef<Path>) -> OcrResult<Self> {
        let model_path_str = model_path.as_ref().to_string_lossy().to_string();
        let keys_path_str = keys_path.as_ref().to_string_lossy().to_string();
        trace!("Rec::from_file called with model: {}, keys: {}", model_path_str, keys_path_str);

        let interpreter = Interpreter::from_file(model_path)
            .map_err(|e| {
                error!("Interpreter::from_file failed for model {}: {:?}", model_path_str, e);
                e
            })?;
        debug!("Interpreter created from file: {}.", model_path_str);
        
        let keys_content = std::fs::read_to_string(keys_path)
            .map_err(|e| {
                error!("Failed to read keys file {}: {:?}", keys_path_str, e);
                e
            })?;
        debug!("Keys content loaded from file: {}. Length: {}", keys_path_str, keys_content.len());

        let keys = " "
            .chars()
            .chain(keys_content.chars().filter(|x| *x != '\n' && *x != '\r'))
            .chain(" ".chars())
            .collect();
        trace!("Rec::from_file finished.");
        Ok(Self {
            interpreter,
            session: None,
            keys,
            min_score: Self::MIN_SCORE_DEFAULT,
            punct_min_score: Self::PUNCT_MIN_SCORE_DEFAULT,
            #[cfg(feature = "fast_resize")]
            resizer: fast_image_resize::Resizer::new(),
        })
    }

    /// 从模型字节创建文本识别器，需要提供字符集文件路径
    ///
    /// Create a text recognizer from model bytes and character set file
    pub fn from_bytes(
        model_bytes: impl AsRef<[u8]>,
        keys_path: impl AsRef<Path>,
    ) -> OcrResult<Self> {
        let bytes_len = model_bytes.as_ref().len();
        let keys_path_str = keys_path.as_ref().to_string_lossy().to_string();
        trace!("Rec::from_bytes called with {} model bytes, keys: {}", bytes_len, keys_path_str);

        let interpreter = Interpreter::from_bytes(model_bytes)
            .map_err(|e| {
                error!("Interpreter::from_bytes failed for {} bytes: {:?}", bytes_len, e);
                e
            })?;
        debug!("Interpreter created from {} bytes.", bytes_len);
        
        let keys_content = std::fs::read_to_string(keys_path)
            .map_err(|e| {
                error!("Failed to read keys file {}: {:?}", keys_path_str, e);
                e
            })?;
        debug!("Keys content loaded from file: {}. Length: {}", keys_path_str, keys_content.len());

        let keys = " "
            .chars()
            .chain(keys_content.chars().filter(|x| *x != '\n' && *x != '\r'))
            .chain(" ".chars())
            .collect();
        trace!("Rec::from_bytes finished.");
        Ok(Self {
            interpreter,
            session: None,
            keys,
            min_score: Self::MIN_SCORE_DEFAULT,
            punct_min_score: Self::PUNCT_MIN_SCORE_DEFAULT,
            #[cfg(feature = "fast_resize")]
            resizer: fast_image_resize::Resizer::new(),
        })
    }

    /// 从模型字节和字符集字节创建文本识别器
    ///
    /// Create a text recognizer from model bytes and character set bytes
    pub fn from_bytes_with_keys(
        model_bytes: impl AsRef<[u8]>,
        keys_bytes: impl AsRef<[u8]>,
    ) -> OcrResult<Self> {
        let model_bytes_len = model_bytes.as_ref().len();
        let keys_bytes_len = keys_bytes.as_ref().len();
        trace!("Rec::from_bytes_with_keys called with {} model bytes, {} keys bytes.", model_bytes_len, keys_bytes_len);

        let interpreter = Interpreter::from_bytes(model_bytes)
            .map_err(|e| {
                error!("Interpreter::from_bytes failed for {} model bytes: {:?}", model_bytes_len, e);
                e
            })?;
        debug!("Interpreter created from {} model bytes.", model_bytes_len);
        
        let keys_content = std::str::from_utf8(keys_bytes.as_ref()).map_err(|e| {
            error!("Failed to convert keys bytes to UTF-8: {:?}", e);
            crate::error::OcrError::IOError(std::io::Error::new(std::io::ErrorKind::InvalidData, e))
        })?;
        debug!("Keys content loaded from bytes. Length: {}", keys_content.len());

        let keys = " "
            .chars()
            .chain(keys_content.chars().filter(|x| *x != '\n' && *x != '\r'))
            .chain(" ".chars())
            .collect();
        trace!("Rec::from_bytes_with_keys finished.");
        Ok(Self {
            interpreter,
            session: None,
            keys,
            min_score: Self::MIN_SCORE_DEFAULT,
            punct_min_score: Self::PUNCT_MIN_SCORE_DEFAULT,
            #[cfg(feature = "fast_resize")]
            resizer: fast_image_resize::Resizer::new(),
        })
    }

    /// 设置常规字符的最小识别置信度阈值
    ///
    /// Set the minimum confidence threshold for regular characters
    pub fn with_min_score(mut self, min_score: f32) -> Self {
        self.min_score = min_score;
        self
    }

    /// 设置标点符号的最小识别置信度阈值
    ///
    /// Set the minimum confidence threshold for punctuation characters
    pub fn with_punct_min_score(mut self, punct_min_score: f32) -> Self {
        self.punct_min_score = punct_min_score;
        self
    }

    #[inline]
    fn is_punctuation(&self, ch: char) -> bool {
        Self::PUNCTUATIONS.contains(&ch)
    }

    /// 识别图像中的文本，返回字符及其置信度
    ///
    /// Recognize text in the image, returning characters and their confidence scores
    pub fn predict_char_score(&mut self, img: &DynamicImage) -> OcrResult<Vec<(char, f32)>> {
        trace!("Rec::predict_char_score called for image {}x{}.", img.width(), img.height());
        #[cfg(not(feature = "fast_resize"))]
        {
            debug!("Rec::predict_char_score: Preprocessing without fast_resize...");
            let input = Self::preprocess(img)?;
            debug!("Rec::predict_char_score: Preprocessing finished. Input shape: {:?}", input.shape());
            let output = self.run_model(&input)?;
            debug!("Rec::predict_char_score: Model run finished. Found {} char scores.", output.len());
            Ok(output)
        }
        #[cfg(feature = "fast_resize")]
        {
            debug!("Rec::predict_char_score: Preprocessing with fast_resize...");
            let input = Self::preprocess(img, &mut self.resizer)?;
            debug!("Rec::predict_char_score: Preprocessing finished. Input shape: {:?}", input.shape());
            let output = self.run_model(&input)?;
            debug!("Rec::predict_char_score: Model run finished. Found {} char scores.", output.len());
            Ok(output)
        }
    }

    /// 识别图像中的文本，返回字符串
    ///
    /// Recognize text in the image, returning a string
    pub fn predict_str(&mut self, img: &DynamicImage) -> OcrResult<String> {
        let (width, height) = (img.width(), img.height());
        trace!("Rec::predict_str called for image {}x{}.", width, height);
        trace!("Rec::predict_str: Image format: {:?}", img.color());
        trace!("Rec::predict_str: Image pixel count: {}", width * height);

        debug!("Rec::predict_str: Calling predict_char_score...");
        let ret = self.predict_char_score(img)?;
        debug!("Rec::predict_str: predict_char_score returned {} characters.", ret.len());

        let result_str: String = ret.into_iter().map(|x| x.0).collect();
        trace!("Rec::predict_str finished. Result: '{}'", result_str);
        trace!("Rec::predict_str: Result length: {}", result_str.len());
        info!("Rec::predict_str: Text recognition completed successfully. Result: '{}', length: {}", result_str, result_str.len());
        Ok(result_str)
    }

    /// 识别图像中的文本，返回字符串和置信度
    ///
    /// Recognize text in the image, returning a string and confidence score
    pub fn predict_with_confidence(&mut self, img: &DynamicImage) -> OcrResult<(String, f32)> {
        let char_scores = self.predict_char_score(img)?;

        if char_scores.is_empty() {
            return Ok((String::new(), 0.0));
        }

        // 计算平均置信度
        let total_score: f32 = char_scores.iter().map(|(_, score)| *score).sum();
        let avg_score = total_score / char_scores.len() as f32;

        // 提取文本
        let text: String = char_scores.into_iter().map(|(ch, _)| ch).collect();

        Ok((text, avg_score))
    }

    #[cfg(feature = "fast_resize")]
    fn preprocess(
        img: &DynamicImage,
        resizer: &mut Resizer,
    ) -> OcrResult<ArrayBase<OwnedRepr<f32>, Dim<[usize; 4]>>> {
        trace!("Rec::preprocess (fast_resize) called for image {}x{}.", img.width(), img.height());
        use fast_image_resize::{FilterType, ResizeAlg, ResizeOptions};
        let (w, h) = img.dimensions();
        let img = if h <= 48 {
            debug!("Rec::preprocess: Image height {} <= 48, no resize needed.", h);
            Cow::Borrowed(img)
        } else {
            debug!("Rec::preprocess: Image height {} > 48, resizing to 48px height.", h);
            let resize_option =
                ResizeOptions::new().resize_alg(ResizeAlg::Convolution(FilterType::CatmullRom));
            let mut dst_img = DynamicImage::new(w * 48 / h, 48, img.color());
            resizer.resize(img, &mut dst_img, &resize_option)
                .map_err(|e| {
                    error!("Rec::preprocess: fast_image_resize failed: {:?}", e);
                    crate::OcrError::EngineError(format!("Image resize failed: {}", e))
                })?;
            debug!("Rec::preprocess: Image resized to {}x{}.", dst_img.width(), dst_img.height());
            Cow::Owned(dst_img)
        };

        let (w, h) = img.dimensions();
        let mut input = Array::zeros((1, 3, h as usize, w as usize));
        trace!("Rec::preprocess: Created input array with shape {:?}", input.shape());

        const MEAN: f32 = 0.5;
        const STD: f32 = 0.5;

        for pixel in img.pixels() {
            let x = pixel.0 as usize;
            let y = pixel.1 as usize;
            let [r, g, b, _] = pixel.2 .0;

            input[[0, 0, y, x]] = (r as f32 / 255.0 - MEAN) / STD;
            input[[0, 1, y, x]] = (g as f32 / 255.0 - MEAN) / STD;
            input[[0, 2, y, x]] = (b as f32 / 255.0 - MEAN) / STD;
        }
        trace!("Rec::preprocess (fast_resize) finished.");
        Ok(input)
    }

    #[cfg(not(feature = "fast_resize"))]
    fn preprocess(img: &DynamicImage) -> OcrResult<ArrayBase<OwnedRepr<f32>, Dim<[usize; 4]>>> {
        trace!("Rec::preprocess (no fast_resize) called for image {}x{}.", img.width(), img.height());
        let (w, h) = img.dimensions();
        let img = if h <= 48 {
            debug!("Rec::preprocess: Image height {} <= 48, no resize needed.", h);
            Cow::Borrowed(img)
        } else {
            debug!("Rec::preprocess: Image height {} > 48, resizing to 48px height using image::imageops::FilterType::CatmullRom.", h);
            Cow::Owned(img.resize_exact(w * 48 / h, 48, image::imageops::FilterType::CatmullRom))
        };

        let (w, h) = img.dimensions();
        let mut input = Array::zeros((1, 3, h as usize, w as usize));
        trace!("Rec::preprocess: Created input array with shape {:?}", input.shape());

        const MEAN: f32 = 0.5;
        const STD: f32 = 0.5;

        for pixel in img.pixels() {
            let x = pixel.0 as usize;
            let y = pixel.1 as usize;
            let [r, g, b, _] = pixel.2 .0;

            input[[0, 0, y, x]] = (r as f32 / 255.0 - MEAN) / STD;
            input[[0, 1, y, x]] = (g as f32 / 255.0 - MEAN) / STD;
            input[[0, 2, y, x]] = (b as f32 / 255.0 - MEAN) / STD;
        }
        trace!("Rec::preprocess (no fast_resize) finished.");
        Ok(input)
    }

    fn run_model(
        &mut self,
        input: &ArrayBase<OwnedRepr<f32>, Dim<[usize; 4]>>,
    ) -> OcrResult<Vec<(char, f32)>> {
        trace!("Rec::run_model called. Input array shape: {:?}", input.shape());
        if self.session.is_none() {
            debug!("Rec::run_model: Session is none, creating new session.");
            let mut config = ScheduleConfig::new();
            config.set_type(ForwardType::CPU);

            let mut backend_config = BackendConfig::new();
            // 使用更低精度以提升性能
            backend_config.set_precision_mode(PrecisionMode::Low);
            backend_config.set_power_mode(PowerMode::High);

            config.set_backend_config(backend_config);

            trace!("Rec::run_model: Calling interpreter.create_session()...");
            let session = self.interpreter.create_session(config)
                .map_err(|e| {
                    error!("Rec::run_model: interpreter.create_session failed: {:?}", e);
                    e
                })?;
            self.session = Some(session);
            debug!("Rec::run_model: Session created successfully.");
        }

        // 获取输入输出张量列表，然后取第一个
        let (input_tensor_name, output_tensor_name) = {
            let session = self.session.as_ref().unwrap();
            let inputs = self.interpreter.inputs(session);
            let outputs = self.interpreter.outputs(session);

            // 获取第一个输入和输出张量的信息
            let input_info = inputs.iter().next().ok_or(
                crate::error::OcrError::EngineError("No input tensor found for session".to_string())
            )?;
            let output_info = outputs.iter().next().ok_or(
                crate::error::OcrError::EngineError("No output tensor found for session".to_string())
            )?;

            (
                input_info.name().to_string(),
                output_info.name().to_string(),
            )
        };
        debug!("Rec::run_model: Input tensor: {}, output tensor: {}", input_tensor_name, output_tensor_name);

        let input_shape = input.shape();
        {
            debug!("Rec::run_model: Resizing tensor and session to input shape: {:?}", input_shape);
            let session = self.session.as_mut().unwrap();
            trace!("Rec::run_model: Getting input_unresized tensor...");
            let mut input_tensor = unsafe {
                self.interpreter
                    .input_unresized::<f32>(session, &input_tensor_name)?
            };

            self.interpreter.resize_tensor(
                &mut input_tensor,
                [
                    input_shape[0] as i32,
                    input_shape[1] as i32,
                    input_shape[2] as i32,
                    input_shape[3] as i32,
                ],
            );

            drop(input_tensor); // input_tensor must be dropped before resize_session
            trace!("Rec::run_model: Calling interpreter.resize_session()...");
            self.interpreter.resize_session(session);
            debug!("Rec::run_model: Tensor and session resized.");
        }

        let (output_data, output_shape) = {
            let session = self.session.as_mut().unwrap();
            trace!("Rec::run_model: Getting input tensor for data filling...");
            let mut input_tensor = self.interpreter.input::<f32>(session, &input_tensor_name)?;

            if let Some(flat_data) = input.as_slice() {
                trace!("Rec::run_model: Input data is contiguous. Copying with copy_from_slice.");
                let mut host_tensor = input_tensor.create_host_tensor_from_device(false);
                let host_data_mut = host_tensor.host_mut();
                
                // CRITICAL SECTION: Data Copy
                trace!("Rec::run_model: Calling host_data_mut.copy_from_slice()...");
                host_data_mut.copy_from_slice(flat_data);
                trace!("Rec::run_model: host_data_mut.copy_from_slice() completed.");
                
                trace!("Rec::run_model: Calling input_tensor.copy_from_host_tensor()...");
                input_tensor.copy_from_host_tensor(&host_tensor)?;
                trace!("Rec::run_model: input_tensor.copy_from_host_tensor() completed.");
            } else {
                trace!("Rec::run_model: Input data is not contiguous. Copying element by element.");
                let mut host_tensor = input_tensor.create_host_tensor_from_device(false);
                let host_data_mut = host_tensor.host_mut();
                for (i, val) in input.iter().enumerate() {
                    host_data_mut[i] = *val;
                }
                input_tensor.copy_from_host_tensor(&host_tensor)?;
                trace!("Rec::run_model: Element-by-element copy completed.");
            }

            debug!("Rec::run_model: Calling interpreter.run_session()...");
            self.interpreter.run_session(session)
                .map_err(|e| {
                    error!("Rec::run_model: interpreter.run_session failed: {:?}", e);
                    e
                })?;
            debug!("Rec::run_model: interpreter.run_session completed.");

            trace!("Rec::run_model: Getting output tensor...");
            let output = self
                .interpreter
                .output::<f32>(session, &output_tensor_name)?;
            trace!("Rec::run_model: Calling output.wait()...");
            output.wait(mnn::ffi::MapType::MAP_TENSOR_READ, true);
            trace!("Rec::run_model: output.wait() completed.");

            trace!("Rec::run_model: Creating host tensor from device and copying data...");
            let shape = output.shape();
            let output_host_tensor = output.create_host_tensor_from_device(true);
            (output_host_tensor.host().to_vec(), shape)
        };
        debug!("Rec::run_model: Output data extracted from MNN session.");

        let sequence_length = output_shape[1] as usize;
        let vocab_size = output_shape[2] as usize;
        debug!("Rec::run_model: Output sequence_length: {}, vocab_size: {}.", sequence_length, vocab_size);

        let mut results = Vec::with_capacity(sequence_length);
        let mut last_char: Option<char> = None;

        for i in 0..sequence_length {
            let mut max_idx = 0;
            let mut max_score = 0.0f32;

            for j in 0..vocab_size {
                let offset = i * vocab_size + j;
                if offset < output_data.len() && output_data[offset] > max_score {
                    max_score = output_data[offset];
                    max_idx = j;
                }
            }

            if max_idx > 0 && max_idx < self.keys.len() {
                if let Some(&ch) = self.keys.get(max_idx) {
                    let threshold = if self.is_punctuation(ch) {
                        self.punct_min_score
                    } else {
                        self.min_score
                    };

                    if max_score > threshold {
                        if last_char != Some(ch) || self.is_punctuation(ch) {
                            results.push((ch, max_score));
                        }
                        last_char = Some(ch);
                    } else {
                        if self.is_punctuation(ch) && max_score > self.punct_min_score * 0.8 {
                            results.push((ch, max_score));
                        } else {
                            last_char = None;
                        }
                    }
                }
            } else {
                last_char = None;
            }
        }
        trace!("Rec::run_model: Character decoding finished. Found {} results.", results.len());
        Ok(results)
    }
}

impl Drop for Rec {
    fn drop(&mut self) {
        trace!("Rec::drop called.");
        if let Some(session) = self.session.take() {
            drop(session);
            trace!("Rec::drop: MNN session dropped.");
        } else {
            trace!("Rec::drop: No MNN session to drop.");
        }
    }
}
