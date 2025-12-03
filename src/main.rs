use clap::{Parser, ValueEnum};
use rust_paddle_ocr::{OcrEngineManager, OcrError, OcrResult};
use serde::{Deserialize, Serialize};
use std::path::PathBuf;
use tracing::{error, info};
use tracing_subscriber::{layer::SubscriberExt, util::SubscriberInitExt};

// 根据feature flag选择不同版本的模型
#[cfg(feature = "v5")]
mod models {
    pub static DET_MODEL: &[u8] = include_bytes!("../models/PP-OCRv5_mobile_det_fp16.mnn");
    pub static REC_MODEL: &[u8] = include_bytes!("../models/PP-OCRv5_mobile_rec_fp16.mnn");
    pub static KEYS_DATA: &[u8] = include_bytes!("../models/ppocr_keys_v5.txt");
    pub const VERSION: &str = "v5";
}

#[cfg(not(feature = "v5"))]
mod models {
    pub static DET_MODEL: &[u8] = include_bytes!("../models/ch_PP-OCRv4_det_infer.mnn");
    pub static REC_MODEL: &[u8] = include_bytes!("../models/ch_PP-OCRv4_rec_infer.mnn");
    pub static KEYS_DATA: &[u8] = include_bytes!("../models/ppocr_keys_v4.txt");
    pub const VERSION: &str = "v4";
}

use models::{DET_MODEL, KEYS_DATA, REC_MODEL};

// 定义输出模式
#[derive(ValueEnum, Clone, Debug)]
enum OutputMode {
    /// 详细输出模式，使用JSON包含文本内容和位置信息
    Json,
    /// 简单输出模式，仅输出识别的文本内容
    Text,
}

// 命令行参数
#[derive(Parser, Debug)]
#[command(author, version, about = "PaddleOCR command line tool")]
struct Args {
    /// 要识别的图像路径
    #[arg(short, long, value_name = "IMAGE_PATH")]
    path: Option<PathBuf>,

    /// 输出模式：json(详细) 或 text(简单)
    #[arg(short, long, value_enum, default_value_t = OutputMode::Text)]
    mode: OutputMode,

    /// 是否显示详细日志
    #[arg(short, long)]
    verbose: bool,

    /// 显示模型版本信息
    #[arg(long)]
    version_info: bool,
}

// 文本识别结果的JSON表示
#[derive(Serialize, Deserialize)]
struct TextBox {
    text: String,
    confidence: f32,
    position: TextBoxPosition,
}

#[derive(Serialize, Deserialize)]
struct TextBoxPosition {
    left: i32,
    top: i32,
    width: u32,
    height: u32,
}

fn main() -> OcrResult<()> {
    // 解析命令行参数
    let args = Args::parse();

    // 如果请求版本信息，则显示并退出
    if args.version_info {
        println!("PaddleOCR CLI - Model Version: PP-OCR{}", models::VERSION);
        return Ok(());
    }

    // 检查是否提供了图片路径
    let image_path = match &args.path {
        Some(path) => path,
        None => {
            eprintln!("Error: Image path is required for OCR processing. Use --path to specify the image file.");
            std::process::exit(1);
        }
    };

    // 配置tracing日志
    let log_level = if args.verbose {
        "info"
    } else {
        "error"
    };

    tracing_subscriber::fmt()
        .with_env_filter(log_level)
        .with_target(false)
        .with_thread_ids(false)
        .with_file(false)
        .with_line_number(false)
        .init();

    info!("Starting PaddleOCR command line tool");

    // 检查输入文件是否存在
    if !image_path.exists() {
        error!("Input image file does not exist: {:?}", image_path);
        return Err(OcrError::InputError(format!(
            "Input file not found: {:?}",
            image_path
        )));
    }

    let result = process_ocr(&args, image_path);

    info!("OCR process completed");
    result
}

fn process_ocr(args: &Args, image_path: &PathBuf) -> OcrResult<()> {
    // 直接使用字节数据初始化OCR引擎
    info!(
        "Initializing OCR engine from embedded PP-OCR{} models...",
        models::VERSION
    );
    OcrEngineManager::initialize_with_config_and_bytes(
        DET_MODEL, REC_MODEL, KEYS_DATA, 12,    // rect_border_size
        false, // merge_boxes
        1,     // merge_threshold
    )?;

    // 加载图像
    info!("Loading image from {:?}...", image_path);
    let img = match image::open(image_path) {
        Ok(img) => {
            info!("Image loaded, size: {}x{}", img.width(), img.height());
            img
        }
        Err(e) => {
            error!("Failed to load image: {}", e);
            return Err(e.into());
        }
    };

    // 根据输出模式处理结果
    match args.mode {
        OutputMode::Json => {
            info!("Processing in JSON mode...");

            // 获取文本区域矩形框
            let text_rects = OcrEngineManager::get_text_rects(&img)?;
            info!("Found {} text regions", text_rects.len());

            if text_rects.is_empty() {
                info!("No text regions detected in the image.");
                println!("[]");
                return Ok(());
            }

            // 获取文本区域图像
            let text_images = OcrEngineManager::get_text_images(&img)?;
            info!("Successfully extracted {} text images", text_images.len());

            // 确保文本区域和图像数量一致
            if text_rects.len() != text_images.len() {
                error!(
                    "Mismatch between text rectangles ({}) and text images ({})",
                    text_rects.len(),
                    text_images.len()
                );
                return Err(OcrError::EngineError(
                    "Inconsistent detection results".to_string(),
                ));
            }

            let mut results = Vec::new();

            for (i, (rect, text_img)) in text_rects.iter().zip(text_images.iter()).enumerate() {
                info!("Processing text region {} of {}", i + 1, text_rects.len());

                // 检查子图像是否有效
                if text_img.width() == 0 || text_img.height() == 0 {
                    error!("Invalid subimage with zero dimensions at index {}", i);
                    continue;
                }

                match OcrEngineManager::recognize_text(text_img.clone()) {
                    Ok(text) => {
                        results.push(TextBox {
                            text,
                            confidence: 1.0, // 使用引擎管理器无法获取置信度，设为默认值
                            position: TextBoxPosition {
                                left: rect.left(),
                                top: rect.top(),
                                width: rect.width(),
                                height: rect.height(),
                            },
                        });
                    }
                    Err(e) => {
                        error!("Failed to recognize text in region {}: {}", i, e);
                    }
                }
            }

            // 输出JSON结果
            let json = serde_json::to_string_pretty(&results)
                .map_err(|e| OcrError::OutputError(e.to_string()))?;
            println!("{}", json);
        }

        OutputMode::Text => {
            info!("Processing in text mode...");

            // 直接使用完整的OCR处理
            let texts = OcrEngineManager::process_ocr(img)?;

            for text in texts {
                println!("{}", text);
            }
        }
    }

    Ok(())
}
