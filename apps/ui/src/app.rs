use std::path::PathBuf;

use gpui::{
    App, Bounds, Context, Div, FontWeight, PathPromptOptions, Pixels, Task, TitlebarOptions,
    Window, WindowBounds, WindowOptions, colors::Colors, div, prelude::*, px, rgb, size,
};

use crate::{
    backend::{
        TrainingHandle, WorkerClient, inspect_artifacts, inspect_dataset, locate_project_root,
        start_training,
    },
    components::{action_button, status_pill, tab_button},
    predict_view::render_predict_tab,
    train_view::render_train_tab,
    types::{
        ActiveTab, ArtifactsStatus, DatasetSummary, EvaluationPreview, EvaluationPreviewOrigin,
        MisclassifiedImage, PredictionResult, PredictionState, TrainConfig, TrainEvent,
        display_path, is_supported_image_path,
    },
};

const MAX_RECENT_PREDICTIONS: usize = 6;
const MAX_TRAIN_LOG_LINES: usize = 300;
const DEFAULT_TRAIN_EVALUATION_RATIO: f32 = 0.58;
pub(crate) const TRAIN_SPLITTER_WIDTH: f32 = 16.0;
pub(crate) const MIN_TRAIN_LOG_WIDTH: f32 = 260.0;
pub(crate) const MIN_EVALUATION_WIDTH: f32 = 360.0;

#[derive(Clone, Copy)]
pub enum DirectoryTarget {
    Data,
    Artifacts,
}

pub struct HandClassificationApp {
    pub project_root: PathBuf,
    pub active_tab: ActiveTab,
    pub selected_image: Option<PathBuf>,
    pub prediction_state: PredictionState,
    pub recent_predictions: Vec<PredictionResult>,
    pub train_config: TrainConfig,
    pub dataset_summary: DatasetSummary,
    pub artifacts_status: ArtifactsStatus,
    pub train_logs: Vec<String>,
    pub train_status: Option<String>,
    pub train_running: bool,
    pub train_evaluation_ratio: f32,
    pub evaluation_preview: Option<EvaluationPreview>,
    train_handle: Option<TrainingHandle>,
    worker: WorkerClient,
    predict_generation: u64,
    predict_task: Option<Task<()>>,
    train_task: Option<Task<()>>,
    picker_task: Option<Task<()>>,
}

impl Drop for HandClassificationApp {
    fn drop(&mut self) {
        if let Some(handle) = &self.train_handle {
            handle.stop();
        }
        self.worker.shutdown();
    }
}

impl HandClassificationApp {
    pub fn new(_window: &mut Window, _cx: &mut Context<Self>) -> Self {
        Self::with_project_root(locate_project_root())
    }

    pub fn with_project_root(project_root: PathBuf) -> Self {
        let train_config = TrainConfig::new(&project_root);
        let dataset_summary = inspect_dataset(&train_config.data_dir);
        let artifacts_status = inspect_artifacts(&train_config.artifacts_dir);

        Self {
            project_root,
            active_tab: ActiveTab::Predict,
            selected_image: None,
            prediction_state: PredictionState::Idle,
            recent_predictions: Vec::new(),
            train_config,
            dataset_summary,
            artifacts_status,
            train_logs: Vec::new(),
            train_status: None,
            train_running: false,
            train_evaluation_ratio: DEFAULT_TRAIN_EVALUATION_RATIO,
            evaluation_preview: None,
            train_handle: None,
            worker: WorkerClient::default(),
            predict_generation: 0,
            predict_task: None,
            train_task: None,
            picker_task: None,
        }
    }

    fn refresh_workspace_status(&mut self) {
        self.dataset_summary = inspect_dataset(&self.train_config.data_dir);
        self.artifacts_status = inspect_artifacts(&self.train_config.artifacts_dir);
    }

    pub fn switch_tab(&mut self, tab: ActiveTab, cx: &mut Context<Self>) {
        self.active_tab = tab;
        self.refresh_workspace_status();
        cx.notify();
    }

    pub fn select_image_and_predict(&mut self, path: PathBuf, cx: &mut Context<Self>) {
        if !path.is_file() {
            self.prediction_state = PredictionState::Error {
                image_path: Some(path),
                message: "選択されたファイルが見つかりません。".to_string(),
            };
            cx.notify();
            return;
        }

        if !is_supported_image_path(&path) {
            self.prediction_state = PredictionState::Error {
                image_path: Some(path),
                message: "対応していないファイル形式です（JPEG / PNG / BMP / GIF のみ対応）。"
                    .to_string(),
            };
            cx.notify();
            return;
        }

        self.selected_image = Some(path.clone());
        self.prediction_state = PredictionState::Running {
            image_path: path.clone(),
        };
        self.predict_generation = self.predict_generation.wrapping_add(1);
        let generation = self.predict_generation;
        let worker = self.worker.clone();
        let project_root = self.project_root.clone();
        let artifacts_dir = self.train_config.artifacts_dir.clone();

        self.predict_task = Some(cx.spawn(async move |app, cx| {
            let image_for_error = path.clone();
            let result = cx
                .background_spawn(
                    async move { worker.predict(&project_root, &artifacts_dir, &path) },
                )
                .await;

            let _ = app.update(cx, |this, cx| {
                if this.predict_generation != generation {
                    return;
                }
                match result {
                    Ok(prediction) => {
                        this.record_recent_prediction(prediction.clone());
                        this.prediction_state = PredictionState::Ready(prediction);
                        this.artifacts_status = inspect_artifacts(&this.train_config.artifacts_dir);
                    }
                    Err(message) => {
                        this.prediction_state = PredictionState::Error {
                            image_path: Some(image_for_error),
                            message,
                        };
                    }
                }
                this.predict_task = None;
                cx.notify();
            });
        }));

        cx.notify();
    }

    fn record_recent_prediction(&mut self, prediction: PredictionResult) {
        self.recent_predictions
            .retain(|entry| entry.image_path != prediction.image_path);
        self.recent_predictions.insert(0, prediction);
        self.recent_predictions.truncate(MAX_RECENT_PREDICTIONS);
    }

    pub fn clear_selection(&mut self, cx: &mut Context<Self>) {
        self.predict_generation = self.predict_generation.wrapping_add(1);
        self.predict_task = None;
        self.selected_image = None;
        self.prediction_state = PredictionState::Idle;
        cx.notify();
    }

    pub fn pick_image_file(&mut self, cx: &mut Context<Self>) {
        let receiver = cx.prompt_for_paths(PathPromptOptions {
            files: true,
            directories: false,
            multiple: false,
            prompt: Some("判定する画像を選択".into()),
        });

        self.picker_task = Some(cx.spawn(async move |app, cx| {
            let Ok(Ok(Some(paths))) = receiver.await else {
                return;
            };
            let Some(first) = paths.into_iter().next() else {
                return;
            };
            let _ = app.update(cx, |this, cx| {
                this.picker_task = None;
                this.select_image_and_predict(first, cx);
            });
        }));
    }

    pub fn pick_directory(&mut self, target: DirectoryTarget, cx: &mut Context<Self>) {
        let prompt = match target {
            DirectoryTarget::Data => "学習用画像フォルダ（rock / scissors / paper）を選択",
            DirectoryTarget::Artifacts => "モデル・評価結果の保存先フォルダを選択",
        };
        let receiver = cx.prompt_for_paths(PathPromptOptions {
            files: false,
            directories: true,
            multiple: false,
            prompt: Some(prompt.into()),
        });

        self.picker_task = Some(cx.spawn(async move |app, cx| {
            let Ok(Ok(Some(paths))) = receiver.await else {
                return;
            };
            let Some(directory) = paths.into_iter().next() else {
                return;
            };
            let _ = app.update(cx, |this, cx| {
                this.picker_task = None;
                match target {
                    DirectoryTarget::Data => {
                        this.train_config.data_dir = directory;
                    }
                    DirectoryTarget::Artifacts => {
                        this.train_config.artifacts_dir = directory;
                        this.worker.shutdown();
                    }
                }
                this.refresh_workspace_status();
                cx.notify();
            });
        }));
    }

    pub fn begin_training(&mut self, cx: &mut Context<Self>) {
        if self.train_running {
            return;
        }

        self.refresh_workspace_status();
        self.train_logs.clear();
        self.train_status = Some("学習を開始しています…".to_string());

        match start_training(&self.project_root, &self.train_config) {
            Ok((handle, receiver)) => {
                self.train_running = true;
                self.train_handle = Some(handle);
                self.train_task = Some(cx.spawn(async move |app, cx| {
                    while let Ok(event) = receiver.recv().await {
                        let should_stop =
                            matches!(event, TrainEvent::Finished { .. } | TrainEvent::Failed(_));
                        let updated = app.update(cx, |this, cx| {
                            match event {
                                TrainEvent::Log(line) => this.push_train_log(line),
                                TrainEvent::Finished { success, exit_code } => {
                                    this.train_running = false;
                                    this.train_handle = None;
                                    this.train_task = None;
                                    this.worker.shutdown();
                                    this.refresh_workspace_status();
                                    this.train_status = Some(if success {
                                        "学習が完了しました。モデルをすぐに判定へ利用できます。"
                                            .to_string()
                                    } else {
                                        format!(
                                            "学習が終了しました（終了コード: {}）",
                                            exit_code
                                                .map(|code| code.to_string())
                                                .unwrap_or_else(|| "不明".to_string())
                                        )
                                    });
                                }
                                TrainEvent::Failed(message) => {
                                    this.train_running = false;
                                    this.train_handle = None;
                                    this.train_task = None;
                                    this.train_status = Some(message);
                                }
                            }
                            cx.notify();
                        });
                        if updated.is_err() || should_stop {
                            break;
                        }
                    }
                }));
            }
            Err(error) => {
                self.train_status = Some(error);
            }
        }

        cx.notify();
    }

    pub fn stop_training(&mut self, cx: &mut Context<Self>) {
        if let Some(handle) = &self.train_handle {
            handle.stop();
        }
        self.train_status = Some("学習の停止をリクエストしました…".to_string());
        cx.notify();
    }

    fn push_train_log(&mut self, line: String) {
        if self.train_logs.len() >= MAX_TRAIN_LOG_LINES {
            let overflow = self.train_logs.len() + 1 - MAX_TRAIN_LOG_LINES;
            self.train_logs.drain(0..overflow);
        }
        self.train_logs.push(line);
    }

    pub fn resize_train_panes(
        &mut self,
        pointer_x: Pixels,
        container_bounds: Bounds<Pixels>,
        cx: &mut Context<Self>,
    ) {
        self.train_evaluation_ratio = evaluation_ratio_for_pointer(pointer_x, container_bounds);
        cx.notify();
    }

    pub fn show_evaluation_image(
        &mut self,
        image: MisclassifiedImage,
        origin: EvaluationPreviewOrigin,
        cx: &mut Context<Self>,
    ) {
        self.evaluation_preview = Some(EvaluationPreview::Image { image, origin });
        cx.notify();
    }

    pub fn show_confusion_matrix(&mut self, cx: &mut Context<Self>) {
        self.evaluation_preview = Some(EvaluationPreview::ConfusionMatrix);
        cx.notify();
    }

    pub fn show_confusion_matrix_cell(
        &mut self,
        true_class_index: usize,
        predicted_class_index: usize,
        cx: &mut Context<Self>,
    ) {
        self.evaluation_preview = Some(EvaluationPreview::ConfusionMatrixCell {
            true_class_index,
            predicted_class_index,
        });
        cx.notify();
    }

    pub fn return_from_evaluation_preview(&mut self, cx: &mut Context<Self>) {
        self.evaluation_preview = match &self.evaluation_preview {
            Some(EvaluationPreview::ConfusionMatrix) => None,
            Some(EvaluationPreview::ConfusionMatrixCell { .. }) => {
                Some(EvaluationPreview::ConfusionMatrix)
            }
            Some(EvaluationPreview::Image {
                origin: EvaluationPreviewOrigin::MisclassificationList,
                ..
            }) => None,
            Some(EvaluationPreview::Image {
                origin:
                    EvaluationPreviewOrigin::ConfusionMatrixCell {
                        true_class_index,
                        predicted_class_index,
                    },
                ..
            }) => Some(EvaluationPreview::ConfusionMatrixCell {
                true_class_index: *true_class_index,
                predicted_class_index: *predicted_class_index,
            }),
            _ => return,
        };
        cx.notify();
    }

    pub fn close_evaluation_preview(&mut self, cx: &mut Context<Self>) {
        self.evaluation_preview = None;
        cx.notify();
    }

    pub fn adjust_epochs(&mut self, delta: isize, cx: &mut Context<Self>) {
        self.train_config.epochs =
            (self.train_config.epochs as isize + delta).clamp(1, 500) as usize;
        cx.notify();
    }

    pub fn adjust_seed(&mut self, delta: i64, cx: &mut Context<Self>) {
        self.train_config.seed = (self.train_config.seed as i64 + delta).clamp(0, 999_999) as u64;
        cx.notify();
    }

    fn render_header(&self, colors: &Colors, cx: &mut Context<Self>) -> Div {
        let artifacts_label = display_path(&self.train_config.artifacts_dir, &self.project_root);
        let status_color = if self.artifacts_status.model_ready {
            rgb(0x1f9d68)
        } else {
            rgb(0xd97706)
        };
        let status_text = if self.artifacts_status.model_ready {
            "モデル準備完了"
        } else {
            "モデル未学習"
        };

        div()
            .flex()
            .flex_row()
            .items_center()
            .justify_between()
            .gap_4()
            .px_6()
            .py_4()
            .border_b_1()
            .border_color(colors.border)
            .bg(colors.container)
            .child(
                div()
                    .flex()
                    .flex_col()
                    .gap_0p5()
                    .child(
                        div()
                            .text_lg()
                            .font_weight(FontWeight::BOLD)
                            .child("グー・チョキ・パー画像分類"),
                    )
                    .child(
                        div()
                            .text_xs()
                            .text_color(colors.disabled)
                            .child("TensorFlow / Keras CNN + GPUI"),
                    ),
            )
            .child(
                div()
                    .flex()
                    .flex_row()
                    .items_center()
                    .gap_2()
                    .p_1()
                    .rounded_lg()
                    .bg(colors.background)
                    .border_1()
                    .border_color(colors.border)
                    .child(tab_button(
                        "tab-predict",
                        "画像判定",
                        ActiveTab::Predict,
                        self.active_tab,
                        colors,
                        cx,
                    ))
                    .child(tab_button(
                        "tab-train",
                        "モデル学習",
                        ActiveTab::Train,
                        self.active_tab,
                        colors,
                        cx,
                    )),
            )
            .child(
                div()
                    .flex()
                    .flex_row()
                    .items_center()
                    .gap_3()
                    .child(status_pill(status_text, status_color))
                    .child(
                        div()
                            .flex()
                            .flex_row()
                            .items_center()
                            .gap_2()
                            .px_3()
                            .py_1p5()
                            .rounded_lg()
                            .bg(colors.background)
                            .border_1()
                            .border_color(colors.border)
                            .child(
                                div()
                                    .text_xs()
                                    .text_color(colors.disabled)
                                    .child("artifacts:"),
                            )
                            .child(
                                div()
                                    .text_xs()
                                    .max_w(px(160.))
                                    .overflow_hidden()
                                    .child(artifacts_label),
                            )
                            .child(
                                action_button(
                                    "change-artifacts-header",
                                    "変更",
                                    false,
                                    true,
                                    colors,
                                )
                                .text_xs()
                                .px_2()
                                .py_0p5()
                                .on_click(cx.listener(
                                    |this, _, _, cx| {
                                        this.pick_directory(DirectoryTarget::Artifacts, cx);
                                    },
                                )),
                            ),
                    ),
            )
    }
}

fn evaluation_ratio_for_pointer(pointer_x: Pixels, container_bounds: Bounds<Pixels>) -> f32 {
    let pane_width = container_bounds.size.width - px(TRAIN_SPLITTER_WIDTH);
    if pane_width <= px(0.) {
        return DEFAULT_TRAIN_EVALUATION_RATIO;
    }

    let half_pane_width = px(f32::from(pane_width) / 2.0);
    let minimum_log_width = px(MIN_TRAIN_LOG_WIDTH).min(half_pane_width);
    let minimum_evaluation_width = px(MIN_EVALUATION_WIDTH).min(pane_width - minimum_log_width);
    let log_width = (pointer_x - container_bounds.left() - px(TRAIN_SPLITTER_WIDTH / 2.0))
        .clamp(minimum_log_width, pane_width - minimum_evaluation_width);

    1.0 - log_width / pane_width
}

impl Render for HandClassificationApp {
    fn render(&mut self, window: &mut Window, cx: &mut Context<Self>) -> impl IntoElement {
        let colors = Colors::for_appearance(window);

        div()
            .size_full()
            .flex()
            .flex_col()
            .bg(colors.background)
            .text_color(colors.text)
            .child(self.render_header(&colors, cx))
            .child(match self.active_tab {
                ActiveTab::Predict => render_predict_tab(self, &colors, cx),
                ActiveTab::Train => render_train_tab(self, &colors, cx),
            })
    }
}

pub fn app_window_options(cx: &App) -> WindowOptions {
    let bounds = Bounds::centered(None, size(px(1080.), px(760.)), cx);
    WindowOptions {
        window_bounds: Some(WindowBounds::Windowed(bounds)),
        titlebar: Some(TitlebarOptions {
            title: Some("グー・チョキ・パー画像分類".into()),
            ..Default::default()
        }),
        window_min_size: Some(size(px(860.), px(620.))),
        ..Default::default()
    }
}

pub fn run(cx: &mut App) {
    cx.open_window(app_window_options(cx), |window, cx| {
        cx.new(|cx| HandClassificationApp::new(window, cx))
    })
    .expect("Failed to open GPUI window");
    cx.activate(true);
}

#[cfg(test)]
mod tests {
    use super::*;
    use gpui::{TestAppContext, point};
    use std::fs;

    fn temp_dir(name: &str) -> PathBuf {
        let dir = std::env::temp_dir().join(format!(
            "hand_classification_ui_{}_{}",
            name,
            std::process::id()
        ));
        let _ = fs::remove_dir_all(&dir);
        fs::create_dir_all(&dir).unwrap();
        dir
    }

    #[gpui::test]
    fn switching_tabs_and_adjusting_training_hyperparameters(cx: &mut TestAppContext) {
        let root = temp_dir("tabs_and_config");
        let (view, cx) =
            cx.add_window_view(|_, _| HandClassificationApp::with_project_root(root.clone()));

        view.read_with(cx, |view, _| {
            assert_eq!(view.active_tab, ActiveTab::Predict);
            assert_eq!(view.train_config.epochs, 60);
            assert_eq!(view.train_config.batch_size, 32);
            assert!(!view.artifacts_status.model_ready);
        });

        view.update_in(cx, |view, _, cx| {
            view.switch_tab(ActiveTab::Train, cx);
            view.adjust_epochs(10, cx);
            view.adjust_seed(-3, cx);
        });

        view.read_with(cx, |view, _| {
            assert_eq!(view.active_tab, ActiveTab::Train);
            assert_eq!(view.train_config.epochs, 70);
            assert_eq!(view.train_config.seed, 120);
        });

        let _ = fs::remove_dir_all(root);
    }

    #[gpui::test]
    fn training_pane_ratio_can_be_resized_and_survives_tab_changes(cx: &mut TestAppContext) {
        let root = temp_dir("train_pane_ratio");
        let (view, cx) =
            cx.add_window_view(|_, _| HandClassificationApp::with_project_root(root.clone()));
        let container_bounds = Bounds::new(point(px(0.), px(0.)), size(px(800.), px(400.)));

        view.update_in(cx, |view, _, cx| {
            view.resize_train_panes(px(400.), container_bounds, cx);
            view.switch_tab(ActiveTab::Predict, cx);
            view.switch_tab(ActiveTab::Train, cx);
        });

        view.read_with(cx, |view, _| {
            assert!((view.train_evaluation_ratio - 0.5).abs() < 0.001);
        });

        let _ = fs::remove_dir_all(root);
    }

    #[test]
    fn training_pane_ratio_respects_minimum_widths() {
        let bounds = Bounds::new(point(px(0.), px(0.)), size(px(800.), px(400.)));
        let leftmost = evaluation_ratio_for_pointer(px(-100.), bounds);
        let rightmost = evaluation_ratio_for_pointer(px(900.), bounds);
        let pane_width = px(800.) - px(TRAIN_SPLITTER_WIDTH);
        let minimum_evaluation_ratio = px(MIN_EVALUATION_WIDTH) / pane_width;
        let maximum_evaluation_ratio = 1.0 - px(MIN_TRAIN_LOG_WIDTH) / pane_width;

        assert!((leftmost - maximum_evaluation_ratio).abs() < 0.001);
        assert!((rightmost - minimum_evaluation_ratio).abs() < 0.001);
    }

    #[gpui::test]
    fn evaluation_preview_supports_back_and_close(cx: &mut TestAppContext) {
        let root = temp_dir("evaluation_preview");
        let (view, cx) =
            cx.add_window_view(|_, _| HandClassificationApp::with_project_root(root.clone()));
        let image = MisclassifiedImage {
            image_path: root.join("validation.png"),
            true_label: "グー".to_string(),
            predicted_label: "チョキ".to_string(),
        };

        view.update_in(cx, |view, _, cx| {
            view.show_confusion_matrix(cx);
            view.show_confusion_matrix_cell(0, 1, cx);
            view.show_evaluation_image(
                image,
                EvaluationPreviewOrigin::ConfusionMatrixCell {
                    true_class_index: 0,
                    predicted_class_index: 1,
                },
                cx,
            );
            view.return_from_evaluation_preview(cx);
        });
        view.read_with(cx, |view, _| {
            assert_eq!(
                view.evaluation_preview,
                Some(EvaluationPreview::ConfusionMatrixCell {
                    true_class_index: 0,
                    predicted_class_index: 1,
                })
            );
        });

        view.update_in(cx, |view, _, cx| {
            view.return_from_evaluation_preview(cx);
            view.close_evaluation_preview(cx);
        });
        view.read_with(cx, |view, _| {
            assert_eq!(view.evaluation_preview, None);
        });

        view.update_in(cx, |view, _, cx| {
            view.show_evaluation_image(
                MisclassifiedImage {
                    image_path: root.join("misclassified.png"),
                    true_label: "グー".to_string(),
                    predicted_label: "パー".to_string(),
                },
                EvaluationPreviewOrigin::MisclassificationList,
                cx,
            );
            view.return_from_evaluation_preview(cx);
        });
        view.read_with(cx, |view, _| {
            assert_eq!(view.evaluation_preview, None);
        });

        let _ = fs::remove_dir_all(root);
    }

    #[gpui::test]
    fn rejects_missing_or_unsupported_images_without_spawning_worker(cx: &mut TestAppContext) {
        let root = temp_dir("invalid_images");
        let text_file = root.join("notes.txt");
        fs::write(&text_file, "not an image").unwrap();

        let (view, cx) =
            cx.add_window_view(|_, _| HandClassificationApp::with_project_root(root.clone()));

        view.update_in(cx, |view, _, cx| {
            view.select_image_and_predict(root.join("missing.png"), cx);
        });
        view.read_with(cx, |view, _| {
            assert!(matches!(
                view.prediction_state,
                PredictionState::Error { .. }
            ));
        });

        view.update_in(cx, |view, _, cx| {
            view.select_image_and_predict(text_file.clone(), cx);
        });
        view.read_with(cx, |view, _| match &view.prediction_state {
            PredictionState::Error { message, .. } => {
                assert!(message.contains("対応していないファイル形式"));
            }
            other => panic!("expected error, got {other:?}"),
        });

        let _ = fs::remove_dir_all(root);
    }

    #[test]
    fn inspect_dataset_and_artifacts_reflect_disk_state() {
        let root = temp_dir("disk_state");
        let data_dir = root.join("hand_data");
        for class_name in ["rock", "scissors", "paper"] {
            let class_dir = data_dir.join(class_name);
            fs::create_dir_all(&class_dir).unwrap();
            fs::write(class_dir.join("a.png"), b"png").unwrap();
            fs::write(class_dir.join("b.jpg"), b"jpg").unwrap();
        }

        let summary = inspect_dataset(&data_dir);
        assert!(summary.ready);
        assert_eq!(summary.rock_count, 2);
        assert_eq!(summary.scissors_count, 2);
        assert_eq!(summary.paper_count, 2);

        let artifacts_dir = root.join("artifacts");
        fs::create_dir_all(&artifacts_dir).unwrap();
        let initial_artifacts = inspect_artifacts(&artifacts_dir);
        assert!(!initial_artifacts.model_ready);
        assert!(initial_artifacts.misclassification_report.is_none());

        fs::write(artifacts_dir.join("hand_classifier.keras"), b"model").unwrap();
        fs::write(artifacts_dir.join("metadata.json"), b"{}").unwrap();
        assert!(inspect_artifacts(&artifacts_dir).model_ready);

        fs::write(
            artifacts_dir.join("misclassified_images.json"),
            r#"{
                "validation_count": 3,
                "misclassified_images": [{
                    "image_path": "/moved/dataset/rock/example.png",
                    "true_label": "グー",
                    "predicted_label": "チョキ"
                }]
            }"#,
        )
        .unwrap();
        let report_status = inspect_artifacts(&artifacts_dir);
        let report = report_status.misclassification_report.unwrap();
        assert_eq!(report.validation_count, 3);
        assert_eq!(report.images.len(), 1);
        assert_eq!(
            report.images[0].image_path,
            PathBuf::from("/moved/dataset/rock/example.png")
        );
        assert_eq!(report.images[0].true_label, "グー");
        assert_eq!(report.images[0].predicted_label, "チョキ");
        assert!(!report.images[0].image_path.is_file());

        fs::write(
            artifacts_dir.join("misclassified_images.json"),
            r#"{"validation_count": 3, "misclassified_images": []}"#,
        )
        .unwrap();
        let empty_report = inspect_artifacts(&artifacts_dir)
            .misclassification_report
            .unwrap();
        assert_eq!(empty_report.validation_count, 3);
        assert!(empty_report.images.is_empty());

        let _ = fs::remove_dir_all(root);
    }
}
