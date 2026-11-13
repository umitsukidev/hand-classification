use std::{
    path::{Path, PathBuf},
    sync::Arc,
};

use gpui::{Image, Rgba, rgb};
use serde::Deserialize;

pub const SUPPORTED_IMAGE_EXTENSIONS: &[&str] = &["bmp", "gif", "jpeg", "jpg", "png"];

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum ActiveTab {
    Predict,
    Train,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum HandClass {
    Rock,
    Scissors,
    Paper,
}

impl HandClass {
    pub fn from_directory(directory: &str) -> Option<Self> {
        match directory {
            "rock" => Some(Self::Rock),
            "scissors" => Some(Self::Scissors),
            "paper" => Some(Self::Paper),
            _ => None,
        }
    }

    pub fn symbol(self) -> &'static str {
        match self {
            Self::Rock => "✊",
            Self::Scissors => "✌️",
            Self::Paper => "✋",
        }
    }

    pub fn accent_color(self) -> Rgba {
        match self {
            Self::Rock => rgb(0xe05d44),
            Self::Scissors => rgb(0x2d7ff9),
            Self::Paper => rgb(0x1f9d68),
        }
    }
}

#[derive(Clone, Debug, Deserialize, PartialEq)]
pub struct ClassProbability {
    pub directory: String,
    pub label: String,
    pub probability: f32,
}

#[derive(Clone, Debug, PartialEq)]
pub struct PredictionResult {
    pub image_path: PathBuf,
    pub predicted_label: String,
    pub predicted_directory: String,
    pub probabilities: Vec<ClassProbability>,
}

#[derive(Debug, Deserialize)]
pub struct WorkerResponse {
    pub ok: bool,
    pub error: Option<String>,
    pub predicted_label: Option<String>,
    pub predicted_directory: Option<String>,
    pub probabilities: Option<Vec<ClassProbability>>,
}

#[derive(Clone, Debug, PartialEq)]
pub enum PredictionState {
    Idle,
    Running {
        image_path: PathBuf,
    },
    Ready(PredictionResult),
    Error {
        image_path: Option<PathBuf>,
        message: String,
    },
}

#[derive(Clone, Debug, PartialEq)]
pub struct TrainConfig {
    pub data_dir: PathBuf,
    pub artifacts_dir: PathBuf,
    pub epochs: usize,
    pub batch_size: usize,
    pub validation_split: f32,
    pub seed: u64,
}

impl TrainConfig {
    pub fn new(project_root: &Path) -> Self {
        Self {
            data_dir: project_root.join("src/assets/hand_data"),
            artifacts_dir: project_root.join("artifacts"),
            epochs: 60,
            batch_size: 32,
            validation_split: 0.2,
            seed: 123,
        }
    }
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct DatasetSummary {
    pub rock_count: usize,
    pub scissors_count: usize,
    pub paper_count: usize,
    pub ready: bool,
    pub message: String,
}

#[derive(Clone, Debug)]
pub struct ArtifactsStatus {
    pub model_ready: bool,
    pub training_history_png: Option<Arc<Image>>,
    pub confusion_matrix_png: Option<Arc<Image>>,
}

#[derive(Clone, Debug)]
pub enum TrainEvent {
    Log(String),
    Finished {
        success: bool,
        exit_code: Option<i32>,
    },
    Failed(String),
}

pub fn is_supported_image_path(path: &Path) -> bool {
    path.extension()
        .and_then(|ext| ext.to_str())
        .map(|ext| SUPPORTED_IMAGE_EXTENSIONS.contains(&ext.to_ascii_lowercase().as_str()))
        .unwrap_or(false)
}

pub fn format_percentage(probability: f32) -> String {
    format!("{:.1}%", probability.clamp(0.0, 1.0) * 100.0)
}

pub fn display_path(path: &Path, project_root: &Path) -> String {
    path.strip_prefix(project_root)
        .map(|relative| {
            let text = relative.display().to_string();
            if text.is_empty() {
                ".".to_string()
            } else {
                text
            }
        })
        .unwrap_or_else(|_| path.display().to_string())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn image_extension_and_percentage_helpers() {
        assert!(is_supported_image_path(Path::new("hand.PNG")));
        assert!(is_supported_image_path(Path::new("dir/photo.jpeg")));
        assert!(!is_supported_image_path(Path::new("notes.txt")));
        assert_eq!(format_percentage(0.9234), "92.3%");
        assert_eq!(format_percentage(1.5), "100.0%");
    }

    #[test]
    fn worker_response_deserializes_prediction_payload() {
        let json = r#"{
            "ok": true,
            "image": "hand.png",
            "predicted_label": "グー",
            "predicted_directory": "rock",
            "probabilities": [
                {"directory": "rock", "label": "グー", "probability": 0.9},
                {"directory": "scissors", "label": "チョキ", "probability": 0.06},
                {"directory": "paper", "label": "パー", "probability": 0.04}
            ]
        }"#;
        let parsed: WorkerResponse = serde_json::from_str(json).unwrap();
        assert!(parsed.ok);
        assert_eq!(parsed.predicted_label.as_deref(), Some("グー"));
        assert_eq!(parsed.probabilities.unwrap().len(), 3);
    }
}
