use std::time::{Duration, Instant, SystemTime};

use anyhow::Context;
use image::RgbImage;
use nokhwa::{
    Camera, NokhwaError, native_api_backend,
    pixel_format::RgbFormat,
    query,
    utils::{
        CameraFormat, CameraIndex, CameraInfo, FrameFormat, RequestedFormat, RequestedFormatType,
    },
};

/// Owns the camera stream. Create it once; capture frames as needed.
pub struct AttentCamera {
    camera: Camera,
    sequence: u64,
}

/// A decoded RGB888 frame, independent of any inference backend.
/// The image remains in memory only as long as its owner keeps it.
pub struct CapturedFrame {
    pub rgb: RgbImage,
    pub meta: CaptureMeta,
}

pub struct CaptureMeta {
    /// Camera index from this process; not stable across reboots or re-enumeration.
    pub camera_index: CameraIndex,
    /// Negotiated stream format (resolution, pixel encoding, nominal FPS).
    pub stream_format: CameraFormat,
    /// Local counter of successfully decoded frames (starts at 1).
    pub sequence: u64,
    /// Wall-clock timestamp recorded after the frame was received.
    pub received_at: SystemTime,
    /// Monotonic timestamp for merging in-process events and measuring intervals.
    pub received_tick: Instant,
    /// Backend-provided timestamp; its time base is backend-specific.
    pub driver_timestamp: Option<Duration>,
    /// Native wire format reported by the capture backend (e.g. MJPEG).
    pub source_format: FrameFormat,
    /// Time spent waiting for and obtaining a frame, not inference time.
    pub capture_duration: Duration,
    /// Time spent decoding to RGB888.
    pub decode_duration: Duration,
}

impl AttentCamera {
    pub fn devices() -> anyhow::Result<Vec<CameraInfo>> {
        let backend = native_api_backend().context("No native camera backend available")?;

        Ok(query(backend)?)
    }

    /// `format` must be advertised by the selected device.
    pub fn new(index: CameraIndex, format: CameraFormat) -> Result<Self, NokhwaError> {
        let request = RequestedFormat::new::<RgbFormat>(RequestedFormatType::Exact(format));
        let mut camera = Camera::new(index, request)?;
        camera.open_stream()?;
        Ok(Self {
            camera,
            sequence: 0,
        })
    }

    /// Negotiates the nearest decodable format instead of requiring an exact match.
    /// This still requires the device to support the requested pixel encoding.
    pub fn new_preferred(index: CameraIndex, format: CameraFormat) -> Result<Self, NokhwaError> {
        let request = RequestedFormat::new::<RgbFormat>(RequestedFormatType::Closest(format));
        let mut camera = Camera::new(index, request)?;
        camera.open_stream()?;
        Ok(Self {
            camera,
            sequence: 0,
        })
    }

    pub fn device(&self) -> &CameraInfo {
        self.camera.info()
    }

    pub fn format(&self) -> CameraFormat {
        self.camera.camera_format()
    }

    pub fn capture(&mut self) -> Result<CapturedFrame, NokhwaError> {
        let capture_start = Instant::now();
        let buffer = self.camera.frame()?;
        let received_tick = Instant::now();
        let received_at = SystemTime::now();
        let capture_duration = received_tick.duration_since(capture_start);

        let driver_timestamp = buffer.capture_timestamp();
        let source_format = buffer.source_frame_format();
        let decode_start = Instant::now();
        let rgb = buffer.decode_image::<RgbFormat>()?;
        let decode_duration = decode_start.elapsed();

        self.sequence = self.sequence.saturating_add(1);
        Ok(CapturedFrame {
            rgb,
            meta: CaptureMeta {
                camera_index: self.camera.index().clone(),
                stream_format: self.camera.camera_format(),
                sequence: self.sequence,
                received_at,
                received_tick,
                driver_timestamp,
                source_format,
                capture_duration,
                decode_duration,
            },
        })
    }
}

// Nokhwa already implements Drop for Camera. Do not add another Drop here.
