use image::{DynamicImage, GenericImageView, GrayImage, Luma};
use imageproc::{point::Point, rect::Rect};
use mnn::{BackendConfig, ForwardType, Interpreter, PowerMode, PrecisionMode, ScheduleConfig};
use ndarray::{Array, ArrayBase, Dim, OwnedRepr};
use std::path::Path;
use tracing::{debug, error, info, trace, warn};

use crate::efficient_cropping::{EfficientCropper, ImageRef};
use crate::error::OcrResult;

/// 文本检测模型
///
/// Text detection model that locates text regions in images
pub struct Det {
    interpreter: Interpreter,
    session: Option<mnn::Session>,
    rect_border_size: u32,
    merge_boxes: bool,
    merge_threshold: i32,
    // 缓存张量名称以避免重复查找
    input_tensor_name: Option<String>,
    output_tensor_name: Option<String>,
    // 缓存最后的输入形状，避免不必要的resize操作
    last_input_shape: Option<[i32; 4]>,
}

impl Det {
    /// 默认边界尺寸常量
    /// Default rectangle border size constant
    pub const RECT_BORDER_SIZE: u32 = 10;

    /// 二值化阈值常量
    /// Binary threshold constant
    const THRESHOLD: u8 = 200;

    /// 最小边界框尺寸阈值，用于过滤噪声
    /// Minimum box size threshold for filtering noise
    const MIN_BOX_SIZE: u32 = 5;

    /// 默认的边界框合并阈值
    /// Default threshold for merging text boxes
    pub const DEFAULT_MERGE_THRESHOLD: i32 = 1;

    /// 创建新的文本检测器实例
    ///
    /// Create a new text detector instance
    pub fn new(interpreter: Interpreter) -> Self {
        trace!("Det::new called.");
        // 初始化时不创建会话，推迟到需要时创建
        Self {
            interpreter,
            session: None,
            rect_border_size: Self::RECT_BORDER_SIZE,
            merge_boxes: false,
            merge_threshold: Self::DEFAULT_MERGE_THRESHOLD,
            input_tensor_name: None,
            output_tensor_name: None,
            last_input_shape: None,
        }
    }

    /// 从模型文件创建文本检测器
    ///
    /// Create a text detector from a model file
    pub fn from_file(model_path: impl AsRef<Path>) -> OcrResult<Self> {
        let model_path_str = model_path.as_ref().to_string_lossy().to_string();
        trace!("Det::from_file called with path: {}", model_path_str);
        let interpreter = Interpreter::from_file(model_path)
            .map_err(|e| {
                error!("Interpreter::from_file failed for path {}: {:?}", model_path_str, e);
                e
            })?;
        debug!("Interpreter created from file: {}", model_path_str);
        Ok(Self {
            interpreter,
            session: None,
            rect_border_size: Self::RECT_BORDER_SIZE,
            merge_boxes: false,
            merge_threshold: Self::DEFAULT_MERGE_THRESHOLD,
            input_tensor_name: None,
            output_tensor_name: None,
            last_input_shape: None,
        })
    }

    /// 从内存字节创建文本检测器
    ///
    /// Create a text detector from model bytes in memory
    pub fn from_bytes(model_bytes: impl AsRef<[u8]>) -> OcrResult<Self> {
        let bytes_len = model_bytes.as_ref().len();
        trace!("Det::from_bytes called with {} bytes.", bytes_len);
        let interpreter = Interpreter::from_bytes(model_bytes)
            .map_err(|e| {
                error!("Interpreter::from_bytes failed for {} bytes: {:?}", bytes_len, e);
                e
            })?;
        debug!("Interpreter created from {} bytes.", bytes_len);
        Ok(Self {
            interpreter,
            session: None,
            rect_border_size: Self::RECT_BORDER_SIZE,
            merge_boxes: false,
            merge_threshold: Self::DEFAULT_MERGE_THRESHOLD,
            input_tensor_name: None,
            output_tensor_name: None,
            last_input_shape: None,
        })
    }

    /// 设置文本框边界扩展大小
    ///
    /// Set the text box border extension size
    pub fn with_rect_border_size(mut self, rect_border_size: u32) -> Self {
        self.rect_border_size = rect_border_size;
        self
    }

    /// 设置是否合并相邻文本框
    ///
    /// Set whether to merge adjacent text boxes
    pub fn with_merge_boxes(mut self, merge_boxes: bool) -> Self {
        self.merge_boxes = merge_boxes;
        self
    }

    /// 设置文本框合并阈值
    ///
    /// Set the threshold for merging text boxes
    pub fn with_merge_threshold(mut self, merge_threshold: i32) -> Self {
        self.merge_threshold = merge_threshold;
        self
    }

    /// 在图像中查找文本区域，返回矩形框列表
    ///
    /// Find text regions in the image and return a list of rectangle boxes
    pub fn find_text_rect(&mut self, img: &DynamicImage) -> OcrResult<Vec<Rect>> {
        info!("Det::find_text_rect called for image {}x{}.", img.width(), img.height());
      trace!("Det::find_text_rect: Image format: {:?}", img.color());
        
        debug!("Det::find_text_rect: Starting preprocessing...");
        let input = Self::preprocess(img)?;
        debug!("Det::find_text_rect: Preprocessing finished. Input array shape: {:?}", input.shape());

        info!("Det::find_text_rect: Starting model run for image {}x{}...", img.width(), img.height());
        let output = self.run_model(&input, img.width(), img.height())?;
        info!("Det::find_text_rect: Model run completed. Output image {}x{}.", output.width(), output.height());

        debug!("Det::find_text_rect: Starting box finding...");
        let mut boxes = self.find_box(&output, img.width(), img.height());
        debug!("Det::find_text_rect: Box finding finished. Found {} raw boxes.", boxes.len());

        // 如果启用了边界框合并功能，则合并重叠的边界框
        if self.merge_boxes {
            debug!("Det::find_text_rect: Merging overlapping boxes (threshold: {})...", self.merge_threshold);
            boxes = Self::merge_overlapping_boxes(boxes, self.merge_threshold);
            debug!("Det::find_text_rect: Merging finished. Found {} merged boxes.", boxes.len());
        } else {
            debug!("Det::find_text_rect: Box merging disabled.");
        }
        trace!("Det::find_text_rect finished. Returning {} boxes.", boxes.len());
        Ok(boxes)
    }

    /// 在图像中查找文本区域，返回裁剪后的子图像列表
    ///
    /// Find text regions in the image and return a list of cropped sub-images
    pub fn find_text_img(&mut self, img: &DynamicImage) -> OcrResult<Vec<DynamicImage>> {
        info!("Det::find_text_img called for image {}x{}.", img.width(), img.height());
        trace!("Det::find_text_img: Image format: {:?}", img.color());
        let rects = self.find_text_rect(img)?;
        info!("Det::find_text_img: Found {} text regions to crop.", rects.len());

        // 直接构建结果向量，避免中间集合转换
        let mut results = Vec::with_capacity(rects.len());
        for (i, rect) in rects.into_iter().enumerate() {
            trace!("Det::find_text_img: Cropping image #{} from rect {:?}.", i, rect);
            results.push(img.crop_imm(
                rect.left() as u32,
                rect.top() as u32,
                rect.width(),
                rect.height(),
            ));
        }
        trace!("Det::find_text_img finished. Returned {} cropped images.", results.len());
        Ok(results)
    }

    /// 使用高效裁剪算法的版本
    /// Find text regions using efficient cropping algorithm
    pub fn find_text_img_efficient(&mut self, img: &DynamicImage) -> OcrResult<Vec<DynamicImage>> {
        trace!("Det::find_text_img_efficient called for image {}x{}.", img.width(), img.height());
        let rects = self.find_text_rect(img)?;
        debug!("Det::find_text_img_efficient: Found {} rects.", rects.len());

        if rects.is_empty() {
            trace!("Det::find_text_img_efficient finished. No rects found.");
            return Ok(Vec::new());
        }

        // 使用ImageRef避免不必要的图像克隆
        let image_ref = ImageRef::from(img.clone());
        trace!("Det::find_text_img_efficient: ImageRef created.");

        // 根据矩形数量选择最优的批量裁剪策略
        let results = match rects.len() {
            1 => {
                debug!("Det::find_text_img_efficient: Cropping single image.");
                vec![EfficientCropper::smart_crop(&image_ref, &rects[0])]
            },
            2..=8 => {
                debug!("Det::find_text_img_efficient: Cropping batch of {} images in parallel.", rects.len());
                EfficientCropper::parallel_batch_crop(&image_ref, &rects)
            },
            _ => {
                debug!("Det::find_text_img_efficient: Cropping optimized batch of {} images.", rects.len());
                EfficientCropper::optimized_batch_crop(&image_ref, &rects)
            },
        };

        trace!("Det::find_text_img_efficient finished. Returned {} cropped images.", results.len());
        Ok(results)
    }

    fn preprocess(img: &DynamicImage) -> OcrResult<ArrayBase<OwnedRepr<f32>, Dim<[usize; 4]>>> {
        trace!("Det::preprocess called for image {}x{}.", img.width(), img.height());
        let (w, h) = img.dimensions();
        let pad_w = Self::get_pad_length(w);
        let pad_h = Self::get_pad_length(h);
        debug!("Det::preprocess: Original image dimensions: {}x{}", w, h);
        debug!("Det::preprocess: Padded dimensions: {}x{} (original {}x{}).", pad_w, pad_h, w, h);
        trace!("Det::preprocess: Image color format: {:?}", img.color());
        trace!("Det::preprocess: Total padding: +{} width, +{} height", pad_w - w, pad_h - h);

        // 预分配数组空间
        let mut input = Array::zeros((1, 3, pad_h as usize, pad_w as usize));

        // 归一化参数
        const MEAN: [f32; 3] = [0.485, 0.456, 0.406];
        const STD: [f32; 3] = [0.229, 0.224, 0.225];

        // 转换为RGB格式以便批量处理
        trace!("Det::preprocess: Converting image to RGB8.");
        let rgb_img = img.to_rgb8();
        let img_width = w as usize;
        let img_height = h as usize;
        trace!("Det::preprocess: Image converted to RGB8.");

        // 使用 rayon 并行处理像素，但使用安全的索引方式
        use rayon::prelude::*;
        use std::sync::Mutex;

        // 收集所有需要处理的像素坐标
        let pixel_coords: Vec<(usize, usize)> = (0..img_height)
            .flat_map(|y| (0..img_width).map(move |x| (x, y)))
            .collect();
        trace!("Det::preprocess: Starting parallel pixel processing for {} pixels.", pixel_coords.len());
        trace!("Det::preprocess: Pixel range: x: 0-{}, y: 0-{}", img_width - 1, img_height - 1);
        trace!("Det::preprocess: Image array shape after allocation: {:?}", input.shape());

        let input_mutex = Mutex::new(&mut input);

        // 分批并行处理像素
        pixel_coords.par_chunks(1024).for_each(|chunk| {
            let mut local_updates = Vec::with_capacity(chunk.len());

            for &(x, y) in chunk {
                let pixel = rgb_img.get_pixel(x as u32, y as u32);
                let [r, g, b] = pixel.0;

                // 计算归一化值
                let norm_r = (r as f32 / 255.0 - MEAN[0]) / STD[0];
                let norm_g = (g as f32 / 255.0 - MEAN[1]) / STD[1];
                let norm_b = (b as f32 / 255.0 - MEAN[2]) / STD[2];

                local_updates.push(((0, 0, y, x), norm_r));
                local_updates.push(((0, 1, y, x), norm_g));
                local_updates.push(((0, 2, y, x), norm_b));
            }

            // 批量更新到主数组
            let mut input_guard = input_mutex.lock().unwrap();
            for (coords, value) in local_updates {
                input_guard[coords] = value;
            }
        });
        trace!("Det::preprocess: Parallel pixel processing finished.");

        trace!("Det::preprocess finished.");
        Ok(input)
    }

    fn run_model(
        &mut self,
        input: &ArrayBase<OwnedRepr<f32>, Dim<[usize; 4]>>,
        width: u32,
        height: u32,
    ) -> OcrResult<GrayImage> {
        debug!("Det::run_model: Entering run_model. Input shape: {:?}", input.shape());
        trace!("Det::run_model called. Image {}x{}, input shape: {:?}", width, height, input.shape());
        let pad_w = Self::get_pad_length(width);

        // 优化配置：使用更好的性能配置
        if self.session.is_none() {
            debug!("Det::run_model: Session is none, creating new session.");
            
            debug!("Det::run_model: Creating ScheduleConfig...");
            let mut config = ScheduleConfig::new();
            debug!("Det::run_model: ScheduleConfig created. Setting ForwardType::CPU...");
            config.set_type(ForwardType::CPU);

            debug!("Det::run_model: Creating BackendConfig...");
            let mut backend_config = BackendConfig::new();
            // 使用更低精度以提升性能（如果支持的话，否则使用Normal）
            debug!("Det::run_model: Setting PrecisionMode::Low...");
            backend_config.set_precision_mode(PrecisionMode::Low);
            debug!("Det::run_model: Setting PowerMode::High...");
            backend_config.set_precision_mode(PrecisionMode::Normal); // Try Normal first to be safe? No, keep as is but log it. 
            // Wait, the original code had Low. I will keep Low but log it.
            // Actually, let's switch to Normal temporarily to see if Low is causing issues on some hardware?
            // No, user asked for logs first. I will keep Low but log it.
            backend_config.set_power_mode(PowerMode::High);

            debug!("Det::run_model: Setting backend config to ScheduleConfig...");
            config.set_backend_config(backend_config);

            debug!("Det::run_model: Calling interpreter.create_session(config)...");
            let session = self.interpreter.create_session(config)
                .map_err(|e| {
                    error!("Det::run_model: interpreter.create_session failed: {:?}", e);
                    e
                })?;
            self.session = Some(session);
            info!("Det::run_model: MNN session created successfully.");
        }

        // 获取或缓存输入输出张量名称
        if self.input_tensor_name.is_none() || self.output_tensor_name.is_none() {
            debug!("Det::run_model: Input/Output tensor names not cached, retrieving...");
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

            self.input_tensor_name = Some(input_info.name().to_string());
            self.output_tensor_name = Some(output_info.name().to_string());
            debug!("Det::run_model: Cached input tensor: {}, output tensor: {}", 
                        self.input_tensor_name.as_ref().unwrap(),
                        self.output_tensor_name.as_ref().unwrap());
        }

        let input_tensor_info = self.input_tensor_name.as_ref().unwrap();
        let output_tensor_info = self.output_tensor_name.as_ref().unwrap();

        let input_shape = input.shape();
        let new_shape = [
            input_shape[0] as i32,
            input_shape[1] as i32,
            input_shape[2] as i32,
            input_shape[3] as i32,
        ];

        // 只在形状变化时才重新调整张量大小
        let need_resize = self
            .last_input_shape
            .map(|last_shape| last_shape != new_shape)
            .unwrap_or(true);

        if need_resize {
            debug!("Det::run_model: Resizing session/tensor.");
            debug!("Det::run_model: Input shape changed or first run. Resizing tensor and session.");
            let session = self.session.as_mut().unwrap();
            
            trace!("Det::run_model: Getting input_unresized tensor...");
            let mut input_tensor = unsafe {
                self.interpreter
                    .input_unresized::<f32>(session, input_tensor_info)?
            };
            trace!("Det::run_model: Calling interpreter.resize_tensor()...");
            self.interpreter.resize_tensor(&mut input_tensor, new_shape);
            drop(input_tensor); // input_tensor must be dropped before resize_session
            trace!("Det::run_model: Calling interpreter.resize_session()...");
            self.interpreter.resize_session(session);
            debug!("Det::run_model: Session resized.");

            // 缓存当前形状
            self.last_input_shape = Some(new_shape);
            debug!("Det::run_model: Tensor and session resized to {:?}", new_shape);
        }

        // 填充输入数据并执行推理
        let output_data = {
            let session = self.session.as_mut().unwrap();
            trace!("Det::run_model: Getting input tensor for data filling...");
            debug!("Det::run_model: Getting input tensor.");
            let mut input_tensor = self.interpreter.input::<f32>(session, input_tensor_info)?;

            // 使用输入数据填充张量
            debug!("Det::run_model: Copying data to input tensor.");
            if let Some(flat_data) = input.as_slice() {
                trace!("Det::run_model: Input data is contiguous. Copying with copy_from_slice.");
                let mut host_tensor = input_tensor.create_host_tensor_from_device(false);
                let host_data_mut = host_tensor.host_mut();
                
                // CRITICAL SECTION: Data Copy
                trace!("Det::run_model: Calling host_data_mut.copy_from_slice()...");
                host_data_mut.copy_from_slice(flat_data);
                trace!("Det::run_model: host_data_mut.copy_from_slice() completed.");
                
                trace!("Det::run_model: Calling input_tensor.copy_from_host_tensor()...");
                input_tensor.copy_from_host_tensor(&host_tensor)?;
                trace!("Det::run_model: input_tensor.copy_from_host_tensor() completed.");
            } else {
                trace!("Det::run_model: Input data is not contiguous. Copying element by element.");
                // 只在必要时逐元素复制
                let mut host_tensor = input_tensor.create_host_tensor_from_device(false);
                let host_data_mut = host_tensor.host_mut();
                for (i, val) in input.iter().enumerate() {
                    host_data_mut[i] = *val;
                }
                input_tensor.copy_from_host_tensor(&host_tensor)?;
                trace!("Det::run_model: Element-by-element copy completed.");
            }
            debug!("Det::run_model: Data copy completed.");

            // 运行推理
            debug!("Det::run_model: Calling interpreter.run_session()...");
            debug!("Det::run_model: Running session.");
            self.interpreter.run_session(session)
                .map_err(|e| {
                    error!("Det::run_model: interpreter.run_session failed: {:?}", e);
                    e
                })?;
            debug!("Det::run_model: interpreter.run_session completed.");
            debug!("Det::run_model: Session run completed.");

            // 获取输出并等待计算完成
            trace!("Det::run_model: Getting output tensor...");
            let output = self
                .interpreter
                .output::<f32>(session, output_tensor_info)?;
            trace!("Det::run_model: Calling output.wait()...");
            debug!("Det::run_model: Waiting for output.");
            output.wait(mnn::ffi::MapType::MAP_TENSOR_READ, true);
            trace!("Det::run_model: output.wait() completed.");
            debug!("Det::run_model: Output ready.");

            // 从设备张量创建主机张量并获取数据
            trace!("Det::run_model: Creating host tensor from device...");
            let output_host_tensor = output.create_host_tensor_from_device(true);
            trace!("Det::run_model: Copying output data to Vec...");
            output_host_tensor.host().to_vec() // 复制数据到新的向量
        };
        debug!("Det::run_model: Output data extracted from MNN session.");

        // 构建灰度图像
        trace!("Det::run_model: Building GrayImage from output data...");
        let img = image::ImageBuffer::from_fn(width, height, |x, y| {
            let index = (y * pad_w + x) as usize;
            if index < output_data.len() {
                Luma([(output_data[index] * 255.0).min(255.0) as u8])
            } else {
                Luma([0])
            }
        });
        trace!("Det::run_model: GrayImage built successfully.");
        Ok(img)
    }

    fn find_box(&self, img: &GrayImage, width: u32, height: u32) -> Vec<Rect> {
        trace!("Det::find_box called for image {}x{}.", width, height);
        debug!("Det::find_box: Finding contours with threshold {}.", Self::THRESHOLD);
        let contours =
            imageproc::contours::find_contours_with_threshold::<u32>(img, Self::THRESHOLD);
        debug!("Det::find_box: Found {} raw contours.", contours.len());

        // 先获取所有有效的边界框
        let mut boxes: Vec<Rect> = contours
            .into_iter()
            .filter(|x| x.parent.is_none()) // 只保留外部轮廓
            .filter_map(|x| Self::bounding_rect(&x.points))
            .collect();
        debug!("Det::find_box: Filtered to {} bounding boxes after initial rect calc.", boxes.len());

        // 应用边界扩展
        boxes = boxes
            .into_iter()
            .map(|x| {
                // 扩展边界，确保完全包含文本
                let left = (x.left() - self.rect_border_size as i32).max(0);
                let top = (x.top() - self.rect_border_size as i32).max(0);

                // 确保扩展后的右边界和下边界不超出图像
                let right = (x.right() + self.rect_border_size as i32).min(width as i32 - 1);
                let bottom = (x.bottom() + self.rect_border_size as i32).min(height as i32 - 1);

                // 计算新的宽度和高度
                let rect_width = (right - left + 1) as u32;
                let rect_height = (bottom - top + 1) as u32;

                Rect::at(left, top).of_size(rect_width, rect_height)
            })
            .collect();
        debug!("Det::find_box: Applied border expansion. Resulting {} boxes.", boxes.len());
        trace!("Det::find_box finished.");
        boxes
    }

    /// 合并重叠的边界框
    ///
    /// Merge overlapping bounding boxes
    fn merge_overlapping_boxes(boxes: Vec<Rect>, threshold: i32) -> Vec<Rect> {
        if boxes.is_empty() {
            return boxes;
        }

        let mut result = Vec::new();
        let mut boxes_to_process = boxes;

        while !boxes_to_process.is_empty() {
            let current = boxes_to_process.remove(0);
            let mut merged = current;
            let mut merged_any = false;
            let mut i = 0;

            while i < boxes_to_process.len() {
                if Self::boxes_overlap_with_threshold(&merged, &boxes_to_process[i], threshold) {
                    // 合并边界框
                    merged = Self::merge_boxes(&merged, &boxes_to_process[i]);
                    boxes_to_process.remove(i);
                    merged_any = true;
                } else {
                    i += 1;
                }
            }

            // 如果当前边界框与其他框合并过，把合并后的结果放回处理队列的开头继续处理
            // 这样可以处理可能的传递性重叠（A与B重叠，合并后的AB与C重叠）
            if merged_any {
                boxes_to_process.insert(0, merged);
            } else {
                // 没有可合并的框，将当前结果添加到最终结果中
                result.push(merged);
            }
        }

        result
    }

    /// 判断两个边界框是否左右重叠(考虑阈值)，忽略上下重叠
    ///
    /// Determine whether two bounding boxes overlap horizontally (considering threshold), ignoring vertical overlap
    fn boxes_overlap_with_threshold(a: &Rect, b: &Rect, threshold: i32) -> bool {
        // 扩展边界框的左右边缘，以允许略微左右重叠或接近的框也可以合并
        let a_left = a.left() - threshold;
        let a_right = a.right() + threshold;

        let b_left = b.left() - threshold;
        let b_right = b.right() + threshold;

        // 检查两个边界框在水平方向上是否重叠
        let horizontal_overlap = !(a_right < b_left || b_right < a_left);

        // 检查两个边界框在垂直方向上是否接近（有一定的重叠）
        // 通常文本行内的字符垂直位置可能有小的偏差，但不应差距太大
        // 设置一个合理的垂直重叠阈值：边界框高度的一定比例
        let a_height = a.height() as i32;
        let b_height = b.height() as i32;
        let min_height = a_height.min(b_height);

        // 允许垂直方向上有一定的偏移，比如最大偏移不超过较小框高度的40%
        let vertical_threshold = (min_height as f32 * 0.4) as i32;

        let a_top = a.top();
        let a_bottom = a.bottom();
        let b_top = b.top();
        let b_bottom = b.bottom();

        // 检查两个边界框在垂直方向上的重叠或接近程度
        let vertical_close = if a_top <= b_top {
            a_bottom + vertical_threshold >= b_top // a在上方，底部需要接近b的顶部
        } else {
            b_bottom + vertical_threshold >= a_top // b在上方，底部需要接近a的顶部
        };

        // 只有水平重叠且垂直接近时才考虑合并
        horizontal_overlap && vertical_close
    }

    /// 合并两个边界框，保持高度的独立性
    ///
    /// Merge two bounding boxes, maintaining height independence
    fn merge_boxes(a: &Rect, b: &Rect) -> Rect {
        // 合并左右边界
        let left = a.left().min(b.left());
        let right = a.right().max(b.right());

        // 对于垂直边界，我们有两种策略：
        // 1. 保持各自的高度边界，取并集（完全包含两个框）
        // 2. 计算加权平均值，使得合并后的框高度更平滑

        // 这里采用第一种策略：保持各自高度边界的并集
        let top = a.top().min(b.top());
        let bottom = a.bottom().max(b.bottom());

        let width = (right - left + 1) as u32;
        let height = (bottom - top + 1) as u32;

        Rect::at(left, top).of_size(width, height)
    }

    /// 计算点集的边界框
    ///
    /// Calculate the bounding box of a set of points
    fn bounding_rect(points: &[Point<u32>]) -> Option<Rect> {
        if points.is_empty() {
            return None;
        }

        // 更高效的边界计算
        let mut x_min = points[0].x;
        let mut x_max = points[0].x;
        let mut y_min = points[0].y;
        let mut y_max = points[0].y;

        for p in points.iter().skip(1) {
            x_min = x_min.min(p.x);
            x_max = x_max.max(p.x);
            y_min = y_min.min(p.y);
            y_max = y_max.max(p.y);
        }

        let width = (x_max - x_min) as u32;
        let height = (y_max - y_min) as u32;

        // 使用常量判断最小尺寸
        if width <= Self::MIN_BOX_SIZE || height <= Self::MIN_BOX_SIZE {
            return None;
        }

        Some(Rect::at(x_min as i32, y_min as i32).of_size(width, height))
    }

    #[inline]
    /// 计算填充长度
    ///
    /// Calculate the padding length
    const fn get_pad_length(length: u32) -> u32 {
        let i = length % 32;
        if i == 0 {
            length
        } else {
            length + 32 - i
        }
    }
}

impl Drop for Det {
    fn drop(&mut self) {
        trace!("Det::drop called.");
        if let Some(session) = self.session.take() {
            drop(session);
            trace!("Det::drop: MNN session dropped.");
        } else {
            trace!("Det::drop: No MNN session to drop.");
        }
    }
}
