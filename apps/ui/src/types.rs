use std::{
    path::{Path, PathBuf},
    sync::Arc,
};

use gpui::Image;
use serde::Deserialize;

pub const SUPPORTED_IMAGE_EXTENSIONS: &[&str] = &["bmp", "gif", "jpeg", "jpg", "png"];

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum ActiveTab {
    Predict,
    Train,
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
            data_dir: project_root.join("assets/hand_data"),
            artifacts_dir: project_root.join("artifacts"),
            epochs: 60,
            batch_size: 32,
            validation_split: 0.2,
            seed: 123,
        }
    }
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct DatasetClassCount {
    pub directory: String,
    pub image_count: usize,
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct DatasetSummary {
    pub classes: Vec<DatasetClassCount>,
    pub ready: bool,
    pub message: String,
}

#[derive(Clone, Debug)]
pub struct ArtifactsStatus {
    pub model_ready: bool,
    pub class_labels: Vec<String>,
    pub training_history_png: Option<Arc<Image>>,
    pub confusion_matrix_png: Option<Arc<Image>>,
    pub misclassification_report: Option<MisclassificationReport>,
}

#[derive(Clone, Debug, Deserialize)]
pub struct MisclassificationReport {
    pub validation_count: usize,
    #[serde(rename = "misclassified_images")]
    pub images: Vec<MisclassifiedImage>,
    #[serde(default)]
    pub correctly_classified_images: Option<Vec<MisclassifiedImage>>,
    #[serde(default, deserialize_with = "deserialize_confusion_matrix_plot")]
    pub confusion_matrix_plot: Option<ConfusionMatrixPlotGeometry>,
}

#[derive(Clone, Copy, Debug, Deserialize, PartialEq)]
pub struct NormalizedRect {
    pub left: f32,
    pub top: f32,
    pub right: f32,
    pub bottom: f32,
}

#[derive(Clone, Debug, Deserialize, PartialEq)]
pub struct ConfusionMatrixPlotGeometry {
    pub image_width: u32,
    pub image_height: u32,
    pub axes_bounds: NormalizedRect,
}

impl ConfusionMatrixPlotGeometry {
    pub fn is_valid(&self) -> bool {
        let bounds = self.axes_bounds;
        self.image_width > 0
            && self.image_height > 0
            && bounds.left.is_finite()
            && bounds.top.is_finite()
            && bounds.right.is_finite()
            && bounds.bottom.is_finite()
            && bounds.left > 0.0
            && bounds.left < bounds.right
            && bounds.right < 1.0
            && bounds.top > 0.0
            && bounds.top < bounds.bottom
            && bounds.bottom < 1.0
    }

    pub fn aspect_ratio(&self) -> f32 {
        self.image_width as f32 / self.image_height as f32
    }

    pub fn cell_bounds(
        &self,
        row: usize,
        column: usize,
        class_count: usize,
    ) -> Option<NormalizedRect> {
        if !self.is_valid() || class_count == 0 || row >= class_count || column >= class_count {
            return None;
        }

        let bounds = self.axes_bounds;
        let cell_width = (bounds.right - bounds.left) / class_count as f32;
        let cell_height = (bounds.bottom - bounds.top) / class_count as f32;
        let left = bounds.left + cell_width * column as f32;
        let top = bounds.top + cell_height * row as f32;
        Some(NormalizedRect {
            left,
            top,
            right: if column + 1 == class_count {
                bounds.right
            } else {
                left + cell_width
            },
            bottom: if row + 1 == class_count {
                bounds.bottom
            } else {
                top + cell_height
            },
        })
    }
}

fn deserialize_confusion_matrix_plot<'de, D>(
    deserializer: D,
) -> Result<Option<ConfusionMatrixPlotGeometry>, D::Error>
where
    D: serde::Deserializer<'de>,
{
    let value = Option::<serde_json::Value>::deserialize(deserializer)?;
    Ok(value
        .and_then(|value| serde_json::from_value::<ConfusionMatrixPlotGeometry>(value).ok())
        .filter(ConfusionMatrixPlotGeometry::is_valid))
}

#[derive(Clone, Debug, Deserialize, PartialEq, Eq)]
pub struct MisclassifiedImage {
    pub image_path: PathBuf,
    pub true_label: String,
    pub predicted_label: String,
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub enum EvaluationPreviewOrigin {
    MisclassificationList,
    ConfusionMatrixCell {
        true_class_index: usize,
        predicted_class_index: usize,
    },
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub enum EvaluationPreview {
    ConfusionMatrix,
    ConfusionMatrixCell {
        true_class_index: usize,
        predicted_class_index: usize,
    },
    Image {
        image: MisclassifiedImage,
        origin: EvaluationPreviewOrigin,
    },
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

    #[test]
    fn confusion_matrix_cells_map_to_the_plot_region_at_multiple_sizes() {
        let geometry = ConfusionMatrixPlotGeometry {
            image_width: 700,
            image_height: 600,
            axes_bounds: NormalizedRect {
                left: 0.1,
                top: 0.05,
                right: 0.9,
                bottom: 0.85,
            },
        };
        assert!(geometry.is_valid());

        let first_cell = geometry.cell_bounds(0, 0, 3).unwrap();
        let last_cell = geometry.cell_bounds(2, 2, 3).unwrap();
        for (display_width, display_height) in [(700.0, 600.0), (350.0, 300.0), (607.0, 520.0)] {
            assert!((first_cell.left * display_width - 70.0 * display_width / 700.0).abs() < 0.001);
            assert!(
                (first_cell.top * display_height - 30.0 * display_height / 600.0).abs() < 0.001
            );
            assert!(
                (last_cell.right * display_width - 630.0 * display_width / 700.0).abs() < 0.001
            );
            assert!(
                (last_cell.bottom * display_height - 510.0 * display_height / 600.0).abs() < 0.001
            );
        }
        assert!(geometry.cell_bounds(3, 0, 3).is_none());
        assert!(geometry.cell_bounds(0, 0, 0).is_none());
    }

    #[test]
    fn invalid_optional_plot_geometry_does_not_discard_image_report() {
        let report: MisclassificationReport = serde_json::from_str(
            r#"{
                "validation_count": 0,
                "misclassified_images": [],
                "correctly_classified_images": [],
                "confusion_matrix_plot": {
                    "image_width": 700,
                    "image_height": 600,
                    "axes_bounds": {
                        "left": -0.1,
                        "top": 0.1,
                        "right": 0.9,
                        "bottom": 0.9
                    }
                }
            }"#,
        )
        .unwrap();

        assert!(report.confusion_matrix_plot.is_none());
        assert_eq!(report.correctly_classified_images, Some(vec![]));
    }
}
