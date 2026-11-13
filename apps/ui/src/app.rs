use std::path::PathBuf;

use gpui::{
    App, Bounds, Context, Div, FontWeight, PathPromptOptions, Task, TitlebarOptions, Window,
    WindowBounds, WindowOptions, colors::Colors, div, prelude::*, px, rgb, size,
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
        ActiveTab, ArtifactsStatus, DatasetSummary, PredictionResult, PredictionState, TrainConfig,
        TrainEvent, display_path, is_supported_image_path,
    },
};

const MAX_RECENT_PREDICTIONS: usize = 6;
const MAX_TRAIN_LOG_LINES: usize = 300;

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
    use gpui::TestApp;
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

    #[test]
    fn switching_tabs_and_adjusting_training_hyperparameters() {
        let root = temp_dir("tabs_and_config");
        let mut app = TestApp::new();
        let mut window =
            app.open_window(|_, _| HandClassificationApp::with_project_root(root.clone()));

        window.read(|view, _| {
            assert_eq!(view.active_tab, ActiveTab::Predict);
            assert_eq!(view.train_config.epochs, 60);
            assert_eq!(view.train_config.batch_size, 32);
            assert!(!view.artifacts_status.model_ready);
        });

        window.update(|view, _, cx| {
            view.switch_tab(ActiveTab::Train, cx);
            view.adjust_epochs(10, cx);
            view.adjust_seed(-3, cx);
        });

        window.read(|view, _| {
            assert_eq!(view.active_tab, ActiveTab::Train);
            assert_eq!(view.train_config.epochs, 70);
            assert_eq!(view.train_config.seed, 120);
        });

        let _ = fs::remove_dir_all(root);
    }

    #[test]
    fn rejects_missing_or_unsupported_images_without_spawning_worker() {
        let root = temp_dir("invalid_images");
        let text_file = root.join("notes.txt");
        fs::write(&text_file, "not an image").unwrap();

        let mut app = TestApp::new();
        let mut window =
            app.open_window(|_, _| HandClassificationApp::with_project_root(root.clone()));

        window.update(|view, _, cx| {
            view.select_image_and_predict(root.join("missing.png"), cx);
        });
        window.read(|view, _| {
            assert!(matches!(
                view.prediction_state,
                PredictionState::Error { .. }
            ));
        });

        window.update(|view, _, cx| {
            view.select_image_and_predict(text_file.clone(), cx);
        });
        window.read(|view, _| match &view.prediction_state {
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
        assert!(!inspect_artifacts(&artifacts_dir).model_ready);

        fs::write(artifacts_dir.join("hand_classifier.keras"), b"model").unwrap();
        fs::write(artifacts_dir.join("metadata.json"), b"{}").unwrap();
        assert!(inspect_artifacts(&artifacts_dir).model_ready);

        let _ = fs::remove_dir_all(root);
    }
}
