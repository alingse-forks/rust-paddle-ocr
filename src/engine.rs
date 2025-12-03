use crate::{Det, OcrError, OcrResult, Rec};

use crossbeam_channel::{unbounded, Receiver, Sender};
use image::DynamicImage;
use imageproc::rect::Rect;
use std::{
    path::Path,
    sync::{Arc, Mutex},
    thread,
};
use tracing::{debug, error, info, trace, warn};

/// OCR请求类型
///
/// Types of OCR requests
#[derive(Debug)]
pub enum OcrRequest {
    /// 文本检测请求
    /// Text detection request
    DetectText {
        /// 输入图像
        /// Input image
        image: DynamicImage,
        /// 结果发送通道
        /// Result sender channel
        result_sender: Sender<OcrResult<Vec<DynamicImage>>>,
    },
    /// 文本识别请求
    /// Text recognition request
    RecognizeText {
        /// 输入图像
        /// Input image
        image: DynamicImage,
        /// 结果发送通道
        /// Result sender channel
        result_sender: Sender<OcrResult<String>>,
    },
    /// 完整OCR处理请求
    /// Full OCR processing request
    ProcessOcr {
        /// 输入图像
        /// Input image
        image: DynamicImage,
        /// 结果发送通道
        /// Result sender channel
        result_sender: Sender<OcrResult<Vec<String>>>,
    },
    /// 获取文本区域矩形框请求
    /// Get text region rectangles request
    GetTextRects {
        /// 输入图像
        /// Input image
        image: DynamicImage,
        /// 结果发送通道
        /// Result sender channel
        result_sender: Sender<OcrResult<Vec<Rect>>>,
    },
    /// 获取文本区域图像请求
    /// Get text region images request
    GetTextImages {
        /// 输入图像
        /// Input image
        image: DynamicImage,
        /// 结果发送通道
        /// Result sender channel
        result_sender: Sender<OcrResult<Vec<DynamicImage>>>,
    },
    /// 使用高效裁剪获取文本区域图像请求
    /// Get text region images using efficient cropping
    GetTextImagesEfficient {
        /// 输入图像
        /// Input image
        image: DynamicImage,
        /// 结果发送通道
        /// Result sender channel
        result_sender: Sender<OcrResult<Vec<DynamicImage>>>,
    },
    /// 使用高效裁剪的完整OCR处理请求
    /// Full OCR processing request with efficient cropping
    ProcessOcrEfficient {
        /// 输入图像
        /// Input image
        image: DynamicImage,
        /// 结果发送通道
        /// Result sender channel
        result_sender: Sender<OcrResult<Vec<String>>>,
    },
    /// 关闭引擎请求
    /// Shutdown engine request
    Shutdown,
}

/// 线程安全的OCR引擎管理器
///
/// Thread-safe OCR engine manager
pub struct OcrEngine {
    request_sender: Sender<OcrRequest>,
    worker_handle: Option<thread::JoinHandle<()>>,
}

impl OcrEngine {
    /// 创建并启动一个新的OCR引擎实例
    ///
    /// Create and start a new OCR engine instance
    pub fn new(
        det_model_path: impl AsRef<Path>,
        rec_model_path: impl AsRef<Path>,
        keys_path: impl AsRef<Path>,
    ) -> OcrResult<Self> {
        Self::new_with_config(
            det_model_path,
            rec_model_path,
            keys_path,
            Det::RECT_BORDER_SIZE,
            false,
            Det::DEFAULT_MERGE_THRESHOLD,
        )
    }

    /// 创建并启动一个带有自定义配置的OCR引擎实例
    ///
    /// Create and start a new OCR engine instance with custom configuration
    pub fn new_with_config(
        det_model_path: impl AsRef<Path>,
        rec_model_path: impl AsRef<Path>,
        keys_path: impl AsRef<Path>,
        rect_border_size: u32,
        merge_boxes: bool,
        merge_threshold: i32,
    ) -> OcrResult<Self> {
        // 创建通信通道
        let (tx, rx) = unbounded();

        // 创建工作线程，该线程将持有OCR模型
        let worker_handle = thread::spawn({
            let det_path = det_model_path.as_ref().to_path_buf();
            let rec_path = rec_model_path.as_ref().to_path_buf();
            let keys = keys_path.as_ref().to_path_buf();

            move || match Self::run_worker(
                det_path,
                rec_path,
                keys,
                rx,
                rect_border_size,
                merge_boxes,
                merge_threshold,
            ) {
                Ok(_) => {}
                Err(e) => eprintln!("OCR worker error: {}", e),
            }
        });

        Ok(Self {
            request_sender: tx,
            worker_handle: Some(worker_handle),
        })
    }

    /// 创建并启动一个带有自定义配置和字节数据的OCR引擎实例
    ///
    /// Create and start a new OCR engine instance with custom configuration and byte data
    pub fn new_with_config_and_bytes(
        det_model_data: &[u8],
        rec_model_data: &[u8],
        keys_data: &[u8],
        rect_border_size: u32,
        merge_boxes: bool,
        merge_threshold: i32,
    ) -> OcrResult<Self> {
        // 创建通信通道
        let (tx, rx) = unbounded();

        // 克隆字节数据，准备传递给工作线程
        let det_data = det_model_data.to_vec();
        let rec_data = rec_model_data.to_vec();
        let keys = keys_data.to_vec();

        // 创建工作线程，该线程将持有OCR模型
        let worker_handle = thread::spawn(move || {
            match Self::run_worker_with_bytes(
                det_data,
                rec_data,
                keys,
                rx,
                rect_border_size,
                merge_boxes,
                merge_threshold,
            ) {
                Ok(_) => {}
                Err(e) => eprintln!("OCR worker error: {}", e),
            }
        });

        Ok(Self {
            request_sender: tx,
            worker_handle: Some(worker_handle),
        })
    }

    /// 在图像中检测文本区域
    ///
    /// Detect text regions in the image
    pub fn detect_text(&self, image: DynamicImage) -> OcrResult<Vec<DynamicImage>> {
        // 创建结果通道
        let (result_tx, result_rx) = unbounded();

        // 发送请求
        self.request_sender
            .send(OcrRequest::DetectText {
                image,
                result_sender: result_tx,
            })
            .map_err(|_| {
                OcrError::EngineError("OCR engine worker thread has terminated".to_string())
            })?;

        // 等待结果
        result_rx.recv().map_err(|_| {
            OcrError::EngineError("Failed to receive result from worker thread".to_string())
        })?
    }

    /// 获取文本区域的矩形框
    ///
    /// Get text region rectangles
    pub fn get_text_rects(&self, image: &DynamicImage) -> OcrResult<Vec<Rect>> {
        // 创建结果通道
        let (result_tx, result_rx) = unbounded();

        // 发送请求
        self.request_sender
            .send(OcrRequest::GetTextRects {
                image: image.clone(),
                result_sender: result_tx,
            })
            .map_err(|_| {
                OcrError::EngineError("OCR engine worker thread has terminated".to_string())
            })?;

        // 等待结果
        result_rx.recv().map_err(|_| {
            OcrError::EngineError("Failed to receive result from worker thread".to_string())
        })?
    }

    /// 获取文本区域图像
    ///
    /// Get text region images
    pub fn get_text_images(&self, image: &DynamicImage) -> OcrResult<Vec<DynamicImage>> {
        // 创建结果通道
        let (result_tx, result_rx) = unbounded();

        // 发送请求
        self.request_sender
            .send(OcrRequest::GetTextImages {
                image: image.clone(),
                result_sender: result_tx,
            })
            .map_err(|_| {
                OcrError::EngineError("OCR engine worker thread has terminated".to_string())
            })?;

        // 等待结果
        result_rx.recv().map_err(|_| {
            OcrError::EngineError("Failed to receive result from worker thread".to_string())
        })?
    }

    /// 识别图像中的文本
    ///
    /// Recognize text in the image
    pub fn recognize_text(&self, image: DynamicImage) -> OcrResult<String> {
        // 创建结果通道
        let (result_tx, result_rx) = unbounded();

        // 发送请求
        self.request_sender
            .send(OcrRequest::RecognizeText {
                image,
                result_sender: result_tx,
            })
            .map_err(|_| {
                OcrError::EngineError("OCR engine worker thread has terminated".to_string())
            })?;

        // 等待结果
        result_rx.recv().map_err(|_| {
            OcrError::EngineError("Failed to receive result from worker thread".to_string())
        })?
    }

    /// 完整的OCR处理，检测并识别图像中的所有文本
    ///
    /// Complete OCR processing, detecting and recognizing all text in the image
    pub fn process_ocr(&self, image: DynamicImage) -> OcrResult<Vec<String>> {
        trace!("OcrEngine::process_ocr called, sending request to worker thread.");
        // 创建结果通道
        let (result_tx, result_rx) = unbounded();

        // 发送请求
        self.request_sender
            .send(OcrRequest::ProcessOcr {
                image,
                result_sender: result_tx,
            })
            .map_err(|e| {
                error!("Failed to send OcrRequest::ProcessOcr to worker thread: {:?}", e);
                OcrError::EngineError("OCR engine worker thread has terminated".to_string())
            })?;

        // 等待结果
        let res = result_rx.recv().map_err(|e| {
            error!("Failed to receive result from worker thread for ProcessOcr: {:?}", e);
            OcrError::EngineError("Failed to receive result from worker thread".to_string())
        })?;
        trace!("OcrEngine::process_ocr received result from worker thread.");
        res
    }

    /// 使用高效裁剪获取文本区域图像
    ///
    /// Get text region images using efficient cropping
    pub fn get_text_images_efficient(&self, image: &DynamicImage) -> OcrResult<Vec<DynamicImage>> {
        // 创建结果通道
        let (result_tx, result_rx) = unbounded();

        // 发送请求
        self.request_sender
            .send(OcrRequest::GetTextImagesEfficient {
                image: image.clone(),
                result_sender: result_tx,
            })
            .map_err(|_| {
                OcrError::EngineError("OCR engine worker thread has terminated".to_string())
            })?;

        // 等待结果
        result_rx.recv().map_err(|_| {
            OcrError::EngineError("Failed to receive result from worker thread".to_string())
        })?
    }

    /// 使用高效裁剪的完整OCR处理
    ///
    /// Complete OCR processing using efficient cropping
    pub fn process_ocr_efficient(&self, image: DynamicImage) -> OcrResult<Vec<String>> {
        // 创建结果通道
        let (result_tx, result_rx) = unbounded();

        // 发送请求
        self.request_sender
            .send(OcrRequest::ProcessOcrEfficient {
                image,
                result_sender: result_tx,
            })
            .map_err(|_| {
                OcrError::EngineError("OCR engine worker thread has terminated".to_string())
            })?;

        // 等待结果
        result_rx.recv().map_err(|_| {
            OcrError::EngineError("Failed to receive result from worker thread".to_string())
        })?
    }

    /// 工作线程的主处理函数
    ///
    /// Main processing function for the worker thread
    fn run_worker(
        det_model_path: impl AsRef<Path>,
        rec_model_path: impl AsRef<Path>,
        keys_path: impl AsRef<Path>,
        receiver: Receiver<OcrRequest>,
        rect_border_size: u32,
        merge_boxes: bool,
        merge_threshold: i32,
    ) -> OcrResult<()> {
        trace!("OCR worker thread started.");
        debug!("Worker: Initializing Det model from path: {:?}", det_model_path.as_ref());
        // 初始化模型，应用自定义配置
        let mut det = Det::from_file(det_model_path)
            .map_err(|e| {
                error!("Worker: Det model initialization failed: {:?}", e);
                e
            })?
            .with_rect_border_size(rect_border_size)
            .with_merge_boxes(merge_boxes)
            .with_merge_threshold(merge_threshold);
        debug!("Worker: Det model initialized.");

        debug!("Worker: Initializing Rec model from path: {:?}", rec_model_path.as_ref());
        let mut rec = Rec::from_file(rec_model_path, keys_path)
            .map_err(|e| {
                error!("Worker: Rec model initialization failed: {:?}", e);
                e
            })?;
        debug!("Worker: Rec model initialized.");

        trace!("Worker: Entering request processing loop.");
        // 处理请求循环
        for request in receiver {
            trace!("Worker: Received request: {:?}", request);
            match request {
                OcrRequest::DetectText {
                    image,
                    result_sender,
                } => {
                    debug!("Worker: Processing DetectText request.");
                    let result = det.find_text_img(&image);
                    debug!("Worker: DetectText result obtained.");
                    // 发送结果，忽略接收端可能已关闭的错误
                    let _ = result_sender.send(result);
                }
                OcrRequest::GetTextRects {
                    image,
                    result_sender,
                } => {
                    debug!("Worker: Processing GetTextRects request.");
                    let result = det.find_text_rect(&image);
                    debug!("Worker: GetTextRects result obtained.");
                    let _ = result_sender.send(result);
                }
                OcrRequest::GetTextImages {
                    image,
                    result_sender,
                } => {
                    debug!("Worker: Processing GetTextImages request.");
                    let result = det.find_text_img(&image);
                    debug!("Worker: GetTextImages result obtained.");
                    let _ = result_sender.send(result);
                }
                OcrRequest::RecognizeText {
                    image,
                    result_sender,
                } => {
                    debug!("Worker: Processing RecognizeText request.");
                    let result = rec.predict_str(&image);
                    debug!("Worker: RecognizeText result obtained.");
                    let _ = result_sender.send(result);
                }
                OcrRequest::ProcessOcr {
                    image,
                    result_sender,
                } => {
                    let (img_width, img_height) = (image.width(), image.height());
                    info!("Worker: Processing ProcessOcr request. Image size: {}x{}", img_width, img_height);
                    info!("Worker: Starting OCR processing pipeline for ProcessOcr request");
                    warn!("Worker: OCR Request Type: ProcessOcr - Full OCR pipeline starting");

                    // 先检测文本区域
                    debug!("Worker: Calling det.find_text_img for ProcessOcr.");
                    trace!("Worker: Input image format: {:?}", image.color());
                    trace!("Worker: Input image pixel count: {}", img_width * img_height);

                    match det.find_text_img(&image) {
                        Ok(text_images) => {
                            debug!("Worker: det.find_text_img returned {} images. Starting recognition.", text_images.len());
                            info!("Worker: Text detection phase completed. Found {} text regions to recognize.", text_images.len());
                            // 识别每个文本区域
                            let mut results = Vec::with_capacity(text_images.len());
                            for (i, text_img) in text_images.into_iter().enumerate() {
                                let (text_width, text_height) = (text_img.width(), text_img.height());
                                trace!("Worker: Recognizing text for image #{}... Size: {}x{}", i, text_width, text_height);
                                debug!("Worker: Calling rec.predict_str for text image #{} with size {}x{}", i, text_width, text_height);

                                match rec.predict_str(&text_img) {
                                    Ok(text) => results.push(text),
                                    Err(e) => {
                                        error!("Worker: Rec::predict_str failed for image #{}: {:?}", i, e);
                                        error!("Worker: Text image #{} details - size: {}x{}, format: {:?}", i, text_width, text_height, text_img.color());
                                        let _ = result_sender.send(Err(e));
                                        break;
                                    }
                                }
                            }
                            debug!("Worker: All text images recognized for ProcessOcr.");
                            info!("Worker: OCR pipeline completed successfully. Returning {} recognized texts.", results.len());
                            let _ = result_sender.send(Ok(results));
                        }
                        Err(e) => {
                            error!("Worker: det.find_text_img failed for ProcessOcr: {:?}", e);
                            let _ = result_sender.send(Err(e));
                        }
                    }
                }
                OcrRequest::GetTextImagesEfficient {
                    image,
                    result_sender,
                } => {
                    debug!("Worker: Processing GetTextImagesEfficient request.");
                    let result = det.find_text_img_efficient(&image);
                    debug!("Worker: GetTextImagesEfficient result obtained.");
                    let _ = result_sender.send(result);
                }
                OcrRequest::ProcessOcrEfficient {
                    image,
                    result_sender,
                } => {
                    debug!("Worker: Processing ProcessOcrEfficient request.");
                    // 使用高效裁剪先检测文本区域
                    debug!("Worker: Calling det.find_text_img_efficient for ProcessOcrEfficient.");
                    match det.find_text_img_efficient(&image) {
                        Ok(text_images) => {
                            debug!("Worker: det.find_text_img_efficient returned {} images. Starting recognition.", text_images.len());
                            // 识别每个文本区域
                            let mut results = Vec::with_capacity(text_images.len());
                            for (i, text_img) in text_images.into_iter().enumerate() {
                                trace!("Worker: Recognizing text for image #{}...", i);
                                match rec.predict_str(&text_img) {
                                    Ok(text) => results.push(text),
                                    Err(e) => {
                                        error!("Worker: Rec::predict_str failed for image #{}: {:?}", i, e);
                                        let _ = result_sender.send(Err(e));
                                        break;
                                    }
                                }
                            }
                            debug!("Worker: All text images recognized for ProcessOcrEfficient.");
                            let _ = result_sender.send(Ok(results));
                        }
                        Err(e) => {
                            error!("Worker: det.find_text_img_efficient failed for ProcessOcrEfficient: {:?}", e);
                            let _ = result_sender.send(Err(e));
                        }
                    }
                }
                OcrRequest::Shutdown => {
                    info!("Worker: Received Shutdown request, exiting loop.");
                    // 收到关闭请求，退出循环
                    break;
                }
            }
        }
        trace!("OCR worker thread finished.");
        Ok(())
    }

    /// 使用字节数据的工作线程的主处理函数
    ///
    /// Main processing function for the worker thread using byte data
    fn run_worker_with_bytes(
        det_model_data: Vec<u8>,
        rec_model_data: Vec<u8>,
        keys_data: Vec<u8>,
        receiver: Receiver<OcrRequest>,
        rect_border_size: u32,
        merge_boxes: bool,
        merge_threshold: i32,
    ) -> OcrResult<()> {
        trace!("OCR worker thread (from bytes) started.");
        debug!("Worker (bytes): Initializing Det model from bytes.");
        // 直接从字节数据初始化模型
        let mut det = Det::from_bytes(&det_model_data)
            .map_err(|e| {
                error!("Worker (bytes): Det model initialization failed from bytes: {:?}", e);
                e
            })?;
        debug!("Worker (bytes): Det model initialized from bytes.");

        debug!("Worker (bytes): Initializing Rec model from bytes.");
        let mut rec = Rec::from_bytes_with_keys(&rec_model_data, &keys_data)
            .map_err(|e| {
                error!("Worker (bytes): Rec model initialization failed from bytes: {:?}", e);
                e
            })?;
        debug!("Worker (bytes): Rec model initialized from bytes.");

        trace!("Worker (bytes): Entering request processing loop.");
        // 处理请求循环
        for request in receiver {
            trace!("Worker (bytes): Received request: {:?}", request);
            match request {
                OcrRequest::DetectText {
                    image,
                    result_sender,
                } => {
                    debug!("Worker (bytes): Processing DetectText request.");
                    let result = det.find_text_img(&image);
                    debug!("Worker (bytes): DetectText result obtained.");
                    // 发送结果，忽略接收端可能已关闭的错误
                    let _ = result_sender.send(result);
                }
                OcrRequest::GetTextRects {
                    image,
                    result_sender,
                } => {
                    debug!("Worker (bytes): Processing GetTextRects request.");
                    let result = det.find_text_rect(&image);
                    debug!("Worker (bytes): GetTextRects result obtained.");
                    let _ = result_sender.send(result);
                }
                OcrRequest::GetTextImages {
                    image,
                    result_sender,
                } => {
                    debug!("Worker (bytes): Processing GetTextImages request.");
                    let result = det.find_text_img(&image);
                    debug!("Worker (bytes): GetTextImages result obtained.");
                    let _ = result_sender.send(result);
                }
                OcrRequest::RecognizeText {
                    image,
                    result_sender,
                } => {
                    debug!("Worker (bytes): Processing RecognizeText request.");
                    let result = rec.predict_str(&image);
                    debug!("Worker (bytes): RecognizeText result obtained.");
                    let _ = result_sender.send(result);
                }
                OcrRequest::ProcessOcr {
                    image,
                    result_sender,
                } => {
                    debug!("Worker (bytes): Processing ProcessOcr request.");
                    // 先检测文本区域
                    debug!("Worker (bytes): Calling det.find_text_img for ProcessOcr.");
                    match det.find_text_img(&image) {
                        Ok(text_images) => {
                            debug!("Worker (bytes): det.find_text_img returned {} images. Starting recognition.", text_images.len());
                            // 识别每个文本区域
                            let mut results = Vec::with_capacity(text_images.len());
                            for (i, text_img) in text_images.into_iter().enumerate() {
                                trace!("Worker (bytes): Recognizing text for image #{}...", i);
                                match rec.predict_str(&text_img) {
                                    Ok(text) => results.push(text),
                                    Err(e) => {
                                        error!("Worker (bytes): Rec::predict_str failed for image #{}: {:?}", i, e);
                                        let _ = result_sender.send(Err(e));
                                        break;
                                    }
                                }
                            }
                            debug!("Worker (bytes): All text images recognized for ProcessOcr.");
                            let _ = result_sender.send(Ok(results));
                        }
                        Err(e) => {
                            error!("Worker (bytes): det.find_text_img failed for ProcessOcr: {:?}", e);
                            let _ = result_sender.send(Err(e));
                        }
                    }
                }
                OcrRequest::GetTextImagesEfficient {
                    image,
                    result_sender,
                } => {
                    debug!("Worker (bytes): Processing GetTextImagesEfficient request.");
                    let result = det.find_text_img_efficient(&image);
                    debug!("Worker (bytes): GetTextImagesEfficient result obtained.");
                    let _ = result_sender.send(result);
                }
                OcrRequest::ProcessOcrEfficient {
                    image,
                    result_sender,
                } => {
                    debug!("Worker (bytes): Processing ProcessOcrEfficient request.");
                    // 使用高效裁剪先检测文本区域
                    debug!("Worker (bytes): Calling det.find_text_img_efficient for ProcessOcrEfficient.");
                    match det.find_text_img_efficient(&image) {
                        Ok(text_images) => {
                            debug!("Worker (bytes): det.find_text_img_efficient returned {} images. Starting recognition.", text_images.len());
                            // 识别每个文本区域
                            let mut results = Vec::with_capacity(text_images.len());
                            for (i, text_img) in text_images.into_iter().enumerate() {
                                trace!("Worker (bytes): Recognizing text for image #{}...", i);
                                match rec.predict_str(&text_img) {
                                    Ok(text) => results.push(text),
                                    Err(e) => {
                                        error!("Worker (bytes): Rec::predict_str failed for image #{}: {:?}", i, e);
                                        let _ = result_sender.send(Err(e));
                                        break;
                                    }
                                }
                            }
                            debug!("Worker (bytes): All text images recognized for ProcessOcrEfficient.");
                            let _ = result_sender.send(Ok(results));
                        }
                        Err(e) => {
                            error!("Worker (bytes): det.find_text_img_efficient failed for ProcessOcrEfficient: {:?}", e);
                            let _ = result_sender.send(Err(e));
                        }
                    }
                }
                OcrRequest::Shutdown => {
                    info!("Worker (bytes): Received Shutdown request, exiting loop.");
                    // 收到关闭请求，退出循环
                    break;
                }
            }
        }
        trace!("OCR worker thread (from bytes) finished.");
        Ok(())
    }
}

impl Drop for OcrEngine {
    fn drop(&mut self) {
        trace!("OcrEngine::drop called, sending shutdown request.");
        // 发送关闭请求
        let _ = self.request_sender.send(OcrRequest::Shutdown);

        // 等待工作线程完成
        if let Some(handle) = self.worker_handle.take() {
            let _ = handle.join();
            trace!("Worker thread joined successfully.");
        }
    }
}

/// 全局OCR引擎单例
///
/// Global OCR engine singleton
pub struct OcrEngineManager {
    // 私有构造函数，防止直接实例化
    _private: (),
}

// 全局单例实例，使用 Arc<Mutex<>> 确保线程安全
static INSTANCE: once_cell::sync::OnceCell<Arc<Mutex<Option<OcrEngine>>>> =
    once_cell::sync::OnceCell::new();

impl OcrEngineManager {
    /// 初始化全局OCR引擎
    ///
    /// Initialize the global OCR engine
    pub fn initialize(
        det_model_path: impl AsRef<Path>,
        rec_model_path: impl AsRef<Path>,
        keys_path: impl AsRef<Path>,
    ) -> OcrResult<()> {
        trace!("OcrEngineManager::initialize called.");
        let engine = OcrEngine::new(det_model_path, rec_model_path, keys_path)
            .map_err(|e| {
                error!("OcrEngine::new failed during OcrEngineManager::initialize: {:?}", e);
                e
            })?;

        // 获取或初始化全局实例
        let instance = INSTANCE.get_or_init(|| {
            trace!("OcrEngineManager: INSTANCE not yet initialized, creating new Arc<Mutex<None>>.");
            Arc::new(Mutex::new(None))
        });

        // 更新引擎实例
        let mut guard = instance.lock().map_err(|_| {
            error!("OcrEngineManager: Failed to acquire lock on OCR engine manager during initialize.");
            OcrError::EngineError("Failed to acquire lock on OCR engine manager".to_string())
        })?;

        *guard = Some(engine);
        info!("OCR Engine successfully initialized with tracing support.");
        trace!("OcrEngineManager initialized.");
        Ok(())
    }

    /// 使用自定义配置初始化全局OCR引擎
    ///
    /// Initialize the global OCR engine with custom configuration
    pub fn initialize_with_config(
        det_model_path: impl AsRef<Path>,
        rec_model_path: impl AsRef<Path>,
        keys_path: impl AsRef<Path>,
        rect_border_size: u32,
        merge_boxes: bool,
        merge_threshold: i32,
    ) -> OcrResult<()> {
        trace!("OcrEngineManager::initialize_with_config called.");
        let engine = OcrEngine::new_with_config(
            det_model_path,
            rec_model_path,
            keys_path,
            rect_border_size,
            merge_boxes,
            merge_threshold,
        )
            .map_err(|e| {
                error!("OcrEngine::new_with_config failed during OcrEngineManager::initialize_with_config: {:?}", e);
                e
            })?;

        // 获取或初始化全局实例
        let instance = INSTANCE.get_or_init(|| {
            trace!("OcrEngineManager: INSTANCE not yet initialized, creating new Arc<Mutex<None>>.");
            Arc::new(Mutex::new(None))
        });

        // 更新引擎实例
        let mut guard = instance.lock().map_err(|_| {
            error!("OcrEngineManager: Failed to acquire lock on OCR engine manager during initialize_with_config.");
            OcrError::EngineError("Failed to acquire lock on OCR engine manager".to_string())
        })?;

        *guard = Some(engine);
        trace!("OcrEngineManager::initialize_with_config finished.");
        Ok(())
    }

    /// 使用自定义配置和字节数据初始化全局OCR引擎
    ///
    /// Initialize the global OCR engine with custom configuration and byte data
    pub fn initialize_with_config_and_bytes(
        det_model_data: &[u8],
        rec_model_data: &[u8],
        keys_data: &[u8],
        rect_border_size: u32,
        merge_boxes: bool,
        merge_threshold: i32,
    ) -> OcrResult<()> {
        trace!("OcrEngineManager::initialize_with_config_and_bytes called.");
        let engine = OcrEngine::new_with_config_and_bytes(
            det_model_data,
            rec_model_data,
            keys_data,
            rect_border_size,
            merge_boxes,
            merge_threshold,
        )
            .map_err(|e| {
                error!("OcrEngine::new_with_config_and_bytes failed during OcrEngineManager::initialize_with_config_and_bytes: {:?}", e);
                e
            })?;

        // 获取或初始化全局实例
        let instance = INSTANCE.get_or_init(|| {
            trace!("OcrEngineManager: INSTANCE not yet initialized, creating new Arc<Mutex<None>>.");
            Arc::new(Mutex::new(None))
        });

        // 更新引擎实例
        let mut guard = instance.lock().map_err(|_| {
            error!("OcrEngineManager: Failed to acquire lock on OCR engine manager during initialize_with_config_and_bytes.");
            OcrError::EngineError("Failed to acquire lock on OCR engine manager".to_string())
        })?;

        *guard = Some(engine);
        trace!("OcrEngineManager::initialize_with_config_and_bytes finished.");
        Ok(())
    }

    /// 获取全局OCR引擎实例
    ///
    /// Get the global OCR engine instance
    pub fn get_instance() -> OcrResult<Arc<Mutex<Option<OcrEngine>>>> {
        trace!("OcrEngineManager::get_instance called.");
        INSTANCE
            .get()
            .cloned()
            .ok_or_else(|| {
                error!("OcrEngineManager: OCR engine not initialized when get_instance called.");
                OcrError::EngineError("OCR engine not initialized".to_string())
            })
    }

    /// 在图像中检测文本区域
    ///
    /// Detect text regions in the image
    pub fn detect_text(image: DynamicImage) -> OcrResult<Vec<DynamicImage>> {
        trace!("OcrEngineManager::detect_text called.");
        let instance = Self::get_instance()?;
        let guard = instance.lock().map_err(|_| {
            error!("OcrEngineManager: Failed to acquire lock on OCR engine manager during detect_text.");
            OcrError::EngineError("Failed to acquire lock on OCR engine manager".to_string())
        })?;

        let engine = guard
            .as_ref()
            .ok_or_else(|| {
                error!("OcrEngineManager: OCR engine not initialized when detect_text called.");
                OcrError::EngineError("OCR engine not initialized".to_string())
            })?;

        engine.detect_text(image)
    }

    /// 获取文本区域的矩形框
    ///
    /// Get text region rectangles
    pub fn get_text_rects(image: &DynamicImage) -> OcrResult<Vec<Rect>> {
        trace!("OcrEngineManager::get_text_rects called.");
        let instance = Self::get_instance()?;
        let guard = instance.lock().map_err(|_| {
            error!("OcrEngineManager: Failed to acquire lock on OCR engine manager during get_text_rects.");
            OcrError::EngineError("Failed to acquire lock on OCR engine manager".to_string())
        })?;

        let engine = guard
            .as_ref()
            .ok_or_else(|| {
                error!("OcrEngineManager: OCR engine not initialized when get_text_rects called.");
                OcrError::EngineError("OCR engine not initialized".to_string())
            })?;

        engine.get_text_rects(image)
    }

    /// 获取文本区域图像
    ///
    /// Get text region images
    pub fn get_text_images(image: &DynamicImage) -> OcrResult<Vec<DynamicImage>> {
        trace!("OcrEngineManager::get_text_images called.");
        let instance = Self::get_instance()?;
        let guard = instance.lock().map_err(|_| {
            error!("OcrEngineManager: Failed to acquire lock on OCR engine manager during get_text_images.");
            OcrError::EngineError("Failed to acquire lock on OCR engine manager".to_string())
        })?;

        let engine = guard
            .as_ref()
            .ok_or_else(|| {
                error!("OcrEngineManager: OCR engine not initialized when get_text_images called.");
                OcrError::EngineError("OCR engine not initialized".to_string())
            })?;

        engine.get_text_images(image)
    }

    /// 识别图像中的文本
    ///
    /// Recognize text in the image
    pub fn recognize_text(image: DynamicImage) -> OcrResult<String> {
        trace!("OcrEngineManager::recognize_text called.");
        let instance = Self::get_instance()?;
        let guard = instance.lock().map_err(|_| {
            error!("OcrEngineManager: Failed to acquire lock on OCR engine manager during recognize_text.");
            OcrError::EngineError("Failed to acquire lock on OCR engine manager".to_string())
        })?;

        let engine = guard
            .as_ref()
            .ok_or_else(|| {
                error!("OcrEngineManager: OCR engine not initialized when recognize_text called.");
                OcrError::EngineError("OCR engine not initialized".to_string())
            })?;

        engine.recognize_text(image)
    }

    /// 完整的OCR处理，检测并识别图像中的所有文本
    ///
    /// Complete OCR processing, detecting and recognizing all text in the image
    pub fn process_ocr(image: DynamicImage) -> OcrResult<Vec<String>> {
        trace!("OcrEngineManager::process_ocr called.");
        let instance = Self::get_instance()?;
        let guard = instance.lock().map_err(|_| {
            error!("OcrEngineManager: Failed to acquire lock on OCR engine manager during process_ocr.");
            OcrError::EngineError("Failed to acquire lock on OCR engine manager".to_string())
        })?;

        let engine = guard
            .as_ref()
            .ok_or_else(|| {
                error!("OcrEngineManager: OCR engine not initialized when process_ocr called.");
                OcrError::EngineError("OCR engine not initialized".to_string())
            })?;

        engine.process_ocr(image)
    }

    /// 使用高效裁剪获取文本区域图像
    ///
    /// Get text region images using efficient cropping
    pub fn get_text_images_efficient(image: &DynamicImage) -> OcrResult<Vec<DynamicImage>> {
        trace!("OcrEngineManager::get_text_images_efficient called.");
        let instance = Self::get_instance()?;
        let guard = instance.lock().map_err(|_| {
            error!("OcrEngineManager: Failed to acquire lock on OCR engine manager during get_text_images_efficient.");
            OcrError::EngineError("Failed to acquire lock on OCR engine manager".to_string())
        })?;

        let engine = guard
            .as_ref()
            .ok_or_else(|| {
                error!("OcrEngineManager: OCR engine not initialized when get_text_images_efficient called.");
                OcrError::EngineError("OCR engine not initialized".to_string())
            })?;

        engine.get_text_images_efficient(image)
    }

    /// 使用高效裁剪的完整OCR处理
    ///
    /// Complete OCR processing using efficient cropping
    pub fn process_ocr_efficient(image: DynamicImage) -> OcrResult<Vec<String>> {
        trace!("OcrEngineManager::process_ocr_efficient called.");
        let instance = Self::get_instance()?;
        let guard = instance.lock().map_err(|_| {
            error!("OcrEngineManager: Failed to acquire lock on OCR engine manager during process_ocr_efficient.");
            OcrError::EngineError("Failed to acquire lock on OCR engine manager".to_string())
        })?;

        let engine = guard
            .as_ref()
            .ok_or_else(|| {
                error!("OcrEngineManager: OCR engine not initialized when process_ocr_efficient called.");
                OcrError::EngineError("OCR engine not initialized".to_string())
            })?;

        engine.process_ocr_efficient(image)
    }
}
