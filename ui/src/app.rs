use std::path::PathBuf;

use gpui::{
    ColorExt, Context, Div, ElementId, ExternalPaths, FontWeight, Image, ObjectFit,
    PathPromptOptions, Rgba, SharedString, Stateful, Task, Window, colors::Colors, div, img,
    prelude::*, px, relative, rgb,
};

use crate::{
    backend::{
        TrainingHandle, WorkerClient, inspect_artifacts, inspect_dataset, locate_project_root,
        start_training,
    },
    types::{
        ActiveTab, ArtifactsStatus, ClassProbability, DatasetSummary, HandClass, PredictionResult,
        PredictionState, TrainConfig, TrainEvent, display_path, format_percentage,
        is_supported_image_path,
    },
};

const MAX_RECENT_PREDICTIONS: usize = 6;
const MAX_TRAIN_LOG_LINES: usize = 300;
const BATCH_SIZE_OPTIONS: &[usize] = &[8, 16, 32, 64];
const VALIDATION_SPLIT_OPTIONS: &[f32] = &[0.1, 0.2, 0.25, 0.3];

pub struct HandClassificationApp {
    project_root: PathBuf,
    active_tab: ActiveTab,
    selected_image: Option<PathBuf>,
    prediction_state: PredictionState,
    recent_predictions: Vec<PredictionResult>,
    train_config: TrainConfig,
    dataset_summary: DatasetSummary,
    artifacts_status: ArtifactsStatus,
    train_logs: Vec<String>,
    train_status: Option<String>,
    train_running: bool,
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
        let project_root = locate_project_root();
        Self::with_project_root(project_root)
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

    fn switch_tab(&mut self, tab: ActiveTab, cx: &mut Context<Self>) {
        self.active_tab = tab;
        self.refresh_workspace_status();
        cx.notify();
    }

    fn select_image_and_predict(&mut self, path: PathBuf, cx: &mut Context<Self>) {
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

    fn clear_selection(&mut self, cx: &mut Context<Self>) {
        self.predict_generation = self.predict_generation.wrapping_add(1);
        self.predict_task = None;
        self.selected_image = None;
        self.prediction_state = PredictionState::Idle;
        cx.notify();
    }

    fn pick_image_file(&mut self, cx: &mut Context<Self>) {
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

    fn pick_directory(&mut self, target: DirectoryTarget, cx: &mut Context<Self>) {
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

    fn begin_training(&mut self, cx: &mut Context<Self>) {
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
                                TrainEvent::Log(line) => {
                                    this.push_train_log(line);
                                }
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

    fn stop_training(&mut self, cx: &mut Context<Self>) {
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

    fn adjust_epochs(&mut self, delta: isize, cx: &mut Context<Self>) {
        let next = (self.train_config.epochs as isize + delta).clamp(1, 500) as usize;
        self.train_config.epochs = next;
        cx.notify();
    }

    fn adjust_seed(&mut self, delta: i64, cx: &mut Context<Self>) {
        let next = (self.train_config.seed as i64 + delta).clamp(0, 999_999) as u64;
        self.train_config.seed = next;
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
                    .child(self.tab_button(
                        "tab-predict",
                        "画像判定",
                        ActiveTab::Predict,
                        colors,
                        cx,
                    ))
                    .child(self.tab_button(
                        "tab-train",
                        "モデル学習",
                        ActiveTab::Train,
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

    fn tab_button(
        &self,
        id: &'static str,
        label: &'static str,
        tab: ActiveTab,
        colors: &Colors,
        cx: &mut Context<Self>,
    ) -> Stateful<Div> {
        let active = self.active_tab == tab;
        div()
            .id(id)
            .px_4()
            .py_1p5()
            .rounded_md()
            .text_sm()
            .font_weight(if active {
                FontWeight::SEMIBOLD
            } else {
                FontWeight::NORMAL
            })
            .cursor_pointer()
            .when(active, |el| {
                el.bg(colors.selected).text_color(colors.selected_text)
            })
            .when(!active, |el| {
                el.text_color(colors.text)
                    .hover(|style| style.bg(colors.separator.opacity(0.45)))
            })
            .child(label)
            .on_click(cx.listener(move |this, _, _, cx| {
                this.switch_tab(tab, cx);
            }))
    }

    fn render_predict_tab(&self, colors: &Colors, cx: &mut Context<Self>) -> Div {
        div()
            .flex_1()
            .min_h(px(0.))
            .flex()
            .flex_col()
            .gap_4()
            .p_6()
            .child(
                div()
                    .flex_1()
                    .min_h(px(0.))
                    .flex()
                    .flex_row()
                    .gap_5()
                    .child(self.render_image_panel(colors, cx))
                    .child(self.render_result_panel(colors, cx)),
            )
            .when(!self.recent_predictions.is_empty(), |container| {
                container.child(self.render_recent_strip(colors, cx))
            })
    }

    fn render_image_panel(&self, colors: &Colors, cx: &mut Context<Self>) -> Div {
        let accent = colors.selected;
        let selected_label = self
            .selected_image
            .as_ref()
            .map(|path| display_path(path, &self.project_root));

        panel_card(colors)
            .flex_1()
            .min_w(px(0.))
            .gap_4()
            .child(
                div()
                    .flex()
                    .flex_row()
                    .items_center()
                    .justify_between()
                    .child(
                        div()
                            .flex()
                            .flex_col()
                            .gap_0p5()
                            .child(
                                div()
                                    .text_base()
                                    .font_weight(FontWeight::SEMIBOLD)
                                    .child("判定する画像"),
                            )
                            .child(div().text_xs().text_color(colors.disabled).child(
                                "ファイルをドロップするか、「画像を選択」から指定します",
                            )),
                    )
                    .child(
                        div()
                            .flex()
                            .flex_row()
                            .items_center()
                            .gap_2()
                            .when(self.selected_image.is_some(), |row| {
                                row.child(
                                    action_button("clear-image", "クリア", false, true, colors)
                                        .on_click(cx.listener(|this, _, _, cx| {
                                            this.clear_selection(cx);
                                        })),
                                )
                            })
                            .child(
                                action_button("pick-image", "画像を選択…", true, true, colors)
                                    .on_click(cx.listener(|this, _, _, cx| {
                                        this.pick_image_file(cx);
                                    })),
                            ),
                    ),
            )
            .child(
                div()
                    .id("image-drop-zone")
                    .flex_1()
                    .min_h(px(280.))
                    .rounded_lg()
                    .border_2()
                    .border_color(colors.border)
                    .bg(colors.background)
                    .overflow_hidden()
                    .flex()
                    .flex_col()
                    .items_center()
                    .justify_center()
                    .p_4()
                    .on_drag_over(move |style, _: &ExternalPaths, _window, _cx| {
                        style.border_color(accent).bg(accent.opacity(0.08))
                    })
                    .on_drop(cx.listener(|this, paths: &ExternalPaths, _window, cx| {
                        if let Some(first) = paths.paths().first() {
                            this.select_image_and_predict(first.clone(), cx);
                        }
                    }))
                    .map(|drop_zone| match &self.selected_image {
                        Some(path) => drop_zone.child(
                            div()
                                .size_full()
                                .flex()
                                .flex_col()
                                .items_center()
                                .justify_center()
                                .gap_3()
                                .child(
                                    div()
                                        .flex_1()
                                        .w_full()
                                        .min_h(px(0.))
                                        .flex()
                                        .items_center()
                                        .justify_center()
                                        .child(
                                            img(path.clone())
                                                .size_full()
                                                .object_fit(ObjectFit::Contain)
                                                .rounded_md(),
                                        ),
                                )
                                .when_some(selected_label, |col, label| {
                                    col.child(
                                        div()
                                            .text_xs()
                                            .text_color(colors.disabled)
                                            .child(label),
                                    )
                                }),
                        ),
                        None => drop_zone.cursor_pointer().on_click(cx.listener(|this, _, _, cx| {
                            this.pick_image_file(cx);
                        })).child(
                            div()
                                .flex()
                                .flex_col()
                                .items_center()
                                .gap_2()
                                .child(div().text_3xl().child("✊ ✌️ ✋"))
                                .child(
                                    div()
                                        .text_sm()
                                        .font_weight(FontWeight::MEDIUM)
                                        .child("ここに画像をドラッグ＆ドロップ"),
                                )
                                .child(
                                    div()
                                        .text_xs()
                                        .text_color(colors.disabled)
                                        .child("またはクリックして画像ファイルを選択（JPEG / PNG / BMP / GIF）"),
                                ),
                        ),
                    }),
            )
    }

    fn render_result_panel(&self, colors: &Colors, cx: &mut Context<Self>) -> Div {
        let mut card = panel_card(colors)
            .w(px(380.))
            .flex_shrink_0()
            .gap_4()
            .child(
                div()
                    .flex()
                    .flex_row()
                    .items_center()
                    .justify_between()
                    .child(
                        div()
                            .text_base()
                            .font_weight(FontWeight::SEMIBOLD)
                            .child("判定結果"),
                    )
                    .when_some(self.selected_image.clone(), |header, path| {
                        let is_running =
                            matches!(self.prediction_state, PredictionState::Running { .. });
                        header.child(
                            action_button("rerun-predict", "再判定", false, !is_running, colors)
                                .text_xs()
                                .px_2p5()
                                .py_1()
                                .on_click(cx.listener(move |this, _, _, cx| {
                                    if !is_running {
                                        this.select_image_and_predict(path.clone(), cx);
                                    }
                                })),
                        )
                    }),
            );

        card = match &self.prediction_state {
            PredictionState::Idle => card.child(
                div()
                    .flex_1()
                    .flex()
                    .flex_col()
                    .justify_center()
                    .gap_3()
                    .p_4()
                    .rounded_lg()
                    .bg(colors.background)
                    .border_1()
                    .border_color(colors.border)
                    .child(
                        div()
                            .text_sm()
                            .font_weight(FontWeight::MEDIUM)
                            .child("画像を選択すると推論を実行します"),
                    )
                    .child(div().text_xs().text_color(colors.disabled).child(
                        if self.artifacts_status.model_ready {
                            "学習済みモデルが検出されました。初回推論時のみ TensorFlow の読み込みに数秒かかります。"
                        } else {
                            "まだ学習済みモデルがありません。「モデル学習」タブから学習を実行するか、artifacts フォルダを指定してください。"
                        },
                    ))
                    .when(!self.artifacts_status.model_ready, |box_el| {
                        box_el.child(
                            action_button("goto-train-idle", "モデル学習タブへ移動", true, true, colors)
                                .on_click(cx.listener(|this, _, _, cx| {
                                    this.switch_tab(ActiveTab::Train, cx);
                                })),
                        )
                    }),
            ),
            PredictionState::Running { .. } => card.child(
                div()
                    .flex_1()
                    .flex()
                    .flex_col()
                    .items_center()
                    .justify_center()
                    .gap_2()
                    .p_5()
                    .rounded_lg()
                    .bg(colors.background)
                    .border_1()
                    .border_color(colors.border)
                    .child(
                        div()
                            .text_base()
                            .font_weight(FontWeight::SEMIBOLD)
                            .child("判定中…"),
                    )
                    .child(
                        div()
                            .text_xs()
                            .text_color(colors.disabled)
                            .child("Keras モデルで画像を解析しています"),
                    ),
            ),
            PredictionState::Ready(result) => {
                let hand_class = HandClass::from_directory(&result.predicted_directory);
                let accent = hand_class.map_or(colors.selected, HandClass::accent_color);
                let symbol = hand_class.map_or("🤖", HandClass::symbol);
                let top_probability = result
                    .probabilities
                    .iter()
                    .find(|entry| entry.directory == result.predicted_directory)
                    .map(|entry| entry.probability)
                    .unwrap_or(0.0);

                card.child(
                    div()
                        .flex()
                        .flex_col()
                        .gap_4()
                        .child(
                            div()
                                .p_5()
                                .rounded_xl()
                                .bg(accent.opacity(0.12))
                                .border_1()
                                .border_color(accent.opacity(0.45))
                                .flex()
                                .flex_row()
                                .items_center()
                                .justify_between()
                                .child(
                                    div()
                                        .flex()
                                        .flex_row()
                                        .items_center()
                                        .gap_3()
                                        .child(div().text_3xl().child(symbol))
                                        .child(
                                            div()
                                                .flex()
                                                .flex_col()
                                                .child(
                                                    div()
                                                        .text_xs()
                                                        .text_color(colors.disabled)
                                                        .child("予測クラス"),
                                                )
                                                .child(
                                                    div()
                                                        .text_2xl()
                                                        .font_weight(FontWeight::BOLD)
                                                        .child(result.predicted_label.clone()),
                                                ),
                                        ),
                                )
                                .child(
                                    div()
                                        .flex()
                                        .flex_col()
                                        .items_end()
                                        .child(
                                            div()
                                                .text_xs()
                                                .text_color(colors.disabled)
                                                .child("信頼度"),
                                        )
                                        .child(
                                            div()
                                                .text_xl()
                                                .font_weight(FontWeight::BOLD)
                                                .text_color(accent)
                                                .child(format_percentage(top_probability)),
                                        ),
                                ),
                        )
                        .child(
                            div()
                                .flex()
                                .flex_col()
                                .gap_3()
                                .p_4()
                                .rounded_lg()
                                .bg(colors.background)
                                .border_1()
                                .border_color(colors.border)
                                .child(
                                    div()
                                        .text_xs()
                                        .font_weight(FontWeight::SEMIBOLD)
                                        .text_color(colors.disabled)
                                        .child("クラス別確率"),
                                )
                                .children(result.probabilities.iter().map(|entry| {
                                    render_probability_row(
                                        entry,
                                        entry.directory == result.predicted_directory,
                                        colors,
                                    )
                                })),
                        ),
                )
            }
            PredictionState::Error { message, .. } => card.child(
                div()
                    .flex()
                    .flex_col()
                    .gap_3()
                    .p_4()
                    .rounded_lg()
                    .bg(rgb(0xdc2626).opacity(0.1))
                    .border_1()
                    .border_color(rgb(0xdc2626).opacity(0.4))
                    .child(
                        div()
                            .text_sm()
                            .font_weight(FontWeight::SEMIBOLD)
                            .text_color(rgb(0xdc2626))
                            .child("判定に失敗しました"),
                    )
                    .child(div().text_xs().child(message.clone()))
                    .when(!self.artifacts_status.model_ready, |box_el| {
                        box_el.child(
                            action_button("goto-train-error", "モデル学習タブへ移動", true, true, colors)
                                .on_click(cx.listener(|this, _, _, cx| {
                                    this.switch_tab(ActiveTab::Train, cx);
                                })),
                        )
                    }),
            ),
        };

        card
    }

    fn render_recent_strip(&self, colors: &Colors, cx: &mut Context<Self>) -> Div {
        panel_card(colors)
            .py_3()
            .px_4()
            .gap_2()
            .child(
                div()
                    .text_xs()
                    .font_weight(FontWeight::SEMIBOLD)
                    .text_color(colors.disabled)
                    .child("最近の判定履歴"),
            )
            .child(
                div().flex().flex_row().flex_wrap().gap_2().children(
                    self.recent_predictions
                        .iter()
                        .enumerate()
                        .map(|(ix, item)| {
                            let path = item.image_path.clone();
                            let file_name = item
                                .image_path
                                .file_name()
                                .and_then(|name| name.to_str())
                                .unwrap_or("image")
                                .to_string();
                            let hand = HandClass::from_directory(&item.predicted_directory);
                            let accent = hand.map_or(colors.selected, HandClass::accent_color);
                            let symbol = hand.map_or("", HandClass::symbol);
                            let is_current = self.selected_image.as_ref() == Some(&item.image_path);

                            div()
                                .id(("recent", ix))
                                .flex()
                                .flex_row()
                                .items_center()
                                .gap_2()
                                .px_3()
                                .py_1p5()
                                .rounded_lg()
                                .bg(colors.background)
                                .border_1()
                                .border_color(if is_current { accent } else { colors.border })
                                .cursor_pointer()
                                .hover(|style| style.bg(colors.separator.opacity(0.35)))
                                .child(div().text_sm().child(symbol))
                                .child(
                                    div()
                                        .text_xs()
                                        .font_weight(FontWeight::MEDIUM)
                                        .child(file_name),
                                )
                                .child(
                                    div()
                                        .text_xs()
                                        .font_weight(FontWeight::SEMIBOLD)
                                        .text_color(accent)
                                        .child(item.predicted_label.clone()),
                                )
                                .on_click(cx.listener(move |this, _, _, cx| {
                                    this.select_image_and_predict(path.clone(), cx);
                                }))
                        }),
                ),
            )
    }

    fn render_train_tab(&self, colors: &Colors, cx: &mut Context<Self>) -> Div {
        div()
            .flex_1()
            .min_h(px(0.))
            .flex()
            .flex_col()
            .gap_4()
            .p_6()
            .child(self.render_train_config_card(colors, cx))
            .child(
                div()
                    .flex_1()
                    .min_h(px(0.))
                    .flex()
                    .flex_row()
                    .gap_5()
                    .child(self.render_train_log_card(colors))
                    .child(self.render_evaluation_card(colors)),
            )
    }

    fn render_train_config_card(&self, colors: &Colors, cx: &mut Context<Self>) -> Div {
        let data_label = display_path(&self.train_config.data_dir, &self.project_root);
        let artifacts_label = display_path(&self.train_config.artifacts_dir, &self.project_root);
        let dataset_color = if self.dataset_summary.ready {
            rgb(0x1f9d68)
        } else {
            rgb(0xd97706)
        };

        panel_card(colors)
            .gap_4()
            .child(
                div()
                    .flex()
                    .flex_row()
                    .items_center()
                    .justify_between()
                    .gap_4()
                    .child(
                        div()
                            .flex()
                            .flex_col()
                            .gap_0p5()
                            .child(
                                div()
                                    .text_base()
                                    .font_weight(FontWeight::SEMIBOLD)
                                    .child("モデル学習設定"),
                            )
                            .child(div().text_xs().text_color(colors.disabled).child(format!(
                                "rock: {}枚 / scissors: {}枚 / paper: {}枚 — {}",
                                self.dataset_summary.rock_count,
                                self.dataset_summary.scissors_count,
                                self.dataset_summary.paper_count,
                                self.dataset_summary.message
                            ))),
                    )
                    .child(
                        div()
                            .flex()
                            .flex_row()
                            .items_center()
                            .gap_3()
                            .child(status_pill(
                                if self.dataset_summary.ready {
                                    "データセット OK"
                                } else {
                                    "データ不足"
                                },
                                dataset_color,
                            ))
                            .child(if self.train_running {
                                action_button("stop-train", "学習を停止", true, true, colors)
                                    .bg(rgb(0xdc2626))
                                    .on_click(cx.listener(|this, _, _, cx| {
                                        this.stop_training(cx);
                                    }))
                            } else {
                                action_button(
                                    "start-train",
                                    "学習を開始",
                                    true,
                                    self.dataset_summary.ready,
                                    colors,
                                )
                                .on_click(cx.listener(
                                    |this, _, _, cx| {
                                        if this.dataset_summary.ready {
                                            this.begin_training(cx);
                                        }
                                    },
                                ))
                            }),
                    ),
            )
            .child(
                div()
                    .flex()
                    .flex_row()
                    .flex_wrap()
                    .gap_4()
                    .child(self.path_selector_field(
                        "pick-data-dir",
                        "学習データ (--data-dir)",
                        data_label,
                        DirectoryTarget::Data,
                        !self.train_running,
                        colors,
                        cx,
                    ))
                    .child(self.path_selector_field(
                        "pick-artifacts-dir",
                        "出力先 (--artifacts-dir)",
                        artifacts_label,
                        DirectoryTarget::Artifacts,
                        !self.train_running,
                        colors,
                        cx,
                    )),
            )
            .child(
                div()
                    .flex()
                    .flex_row()
                    .flex_wrap()
                    .items_center()
                    .gap_6()
                    .child(self.render_epochs_control(colors, cx))
                    .child(self.render_batch_size_control(colors, cx))
                    .child(self.render_validation_split_control(colors, cx))
                    .child(self.render_seed_control(colors, cx)),
            )
    }

    #[allow(clippy::too_many_arguments)]
    fn path_selector_field(
        &self,
        id: &'static str,
        title: &'static str,
        value: String,
        target: DirectoryTarget,
        enabled: bool,
        colors: &Colors,
        cx: &mut Context<Self>,
    ) -> Div {
        div()
            .flex_1()
            .min_w(px(280.))
            .flex()
            .flex_col()
            .gap_1()
            .child(div().text_xs().text_color(colors.disabled).child(title))
            .child(
                div()
                    .flex()
                    .flex_row()
                    .items_center()
                    .justify_between()
                    .gap_2()
                    .px_3()
                    .py_2()
                    .rounded_lg()
                    .bg(colors.background)
                    .border_1()
                    .border_color(colors.border)
                    .child(div().text_xs().overflow_hidden().child(value))
                    .child(
                        action_button(id, "選択…", false, enabled, colors)
                            .text_xs()
                            .px_2p5()
                            .py_1()
                            .on_click(cx.listener(move |this, _, _, cx| {
                                if enabled {
                                    this.pick_directory(target, cx);
                                }
                            })),
                    ),
            )
    }

    fn render_epochs_control(&self, colors: &Colors, cx: &mut Context<Self>) -> Div {
        let enabled = !self.train_running;
        div()
            .flex()
            .flex_col()
            .gap_1()
            .child(div().text_xs().text_color(colors.disabled).child("Epochs"))
            .child(
                div()
                    .flex()
                    .flex_row()
                    .items_center()
                    .gap_1()
                    .child(
                        small_chip("ep-m10", "-10", false, enabled, colors).on_click(cx.listener(
                            move |this, _, _, cx| {
                                if enabled {
                                    this.adjust_epochs(-10, cx);
                                }
                            },
                        )),
                    )
                    .child(
                        small_chip("ep-m1", "-1", false, enabled, colors).on_click(cx.listener(
                            move |this, _, _, cx| {
                                if enabled {
                                    this.adjust_epochs(-1, cx);
                                }
                            },
                        )),
                    )
                    .child(
                        div()
                            .min_w(px(44.))
                            .text_center()
                            .text_sm()
                            .font_weight(FontWeight::SEMIBOLD)
                            .child(self.train_config.epochs.to_string()),
                    )
                    .child(
                        small_chip("ep-p1", "+1", false, enabled, colors).on_click(cx.listener(
                            move |this, _, _, cx| {
                                if enabled {
                                    this.adjust_epochs(1, cx);
                                }
                            },
                        )),
                    )
                    .child(
                        small_chip("ep-p10", "+10", false, enabled, colors).on_click(cx.listener(
                            move |this, _, _, cx| {
                                if enabled {
                                    this.adjust_epochs(10, cx);
                                }
                            },
                        )),
                    ),
            )
    }

    fn render_batch_size_control(&self, colors: &Colors, cx: &mut Context<Self>) -> Div {
        let enabled = !self.train_running;
        div()
            .flex()
            .flex_col()
            .gap_1()
            .child(
                div()
                    .text_xs()
                    .text_color(colors.disabled)
                    .child("Batch Size"),
            )
            .child(div().flex().flex_row().items_center().gap_1().children(
                BATCH_SIZE_OPTIONS.iter().map(|&size| {
                    let active = self.train_config.batch_size == size;
                    small_chip(("bs", size), size.to_string(), active, enabled, colors).on_click(
                        cx.listener(move |this, _, _, cx| {
                            if enabled {
                                this.train_config.batch_size = size;
                                cx.notify();
                            }
                        }),
                    )
                }),
            ))
    }

    fn render_validation_split_control(&self, colors: &Colors, cx: &mut Context<Self>) -> Div {
        let enabled = !self.train_running;
        div()
            .flex()
            .flex_col()
            .gap_1()
            .child(
                div()
                    .text_xs()
                    .text_color(colors.disabled)
                    .child("Validation Split"),
            )
            .child(
                div().flex().flex_row().items_center().gap_1().children(
                    VALIDATION_SPLIT_OPTIONS
                        .iter()
                        .enumerate()
                        .map(|(ix, &split)| {
                            let active = (self.train_config.validation_split - split).abs() < 0.001;
                            small_chip(
                                ("val-split", ix),
                                format!("{split:.2}"),
                                active,
                                enabled,
                                colors,
                            )
                            .on_click(cx.listener(
                                move |this, _, _, cx| {
                                    if enabled {
                                        this.train_config.validation_split = split;
                                        cx.notify();
                                    }
                                },
                            ))
                        }),
                ),
            )
    }

    fn render_seed_control(&self, colors: &Colors, cx: &mut Context<Self>) -> Div {
        let enabled = !self.train_running;
        div()
            .flex()
            .flex_col()
            .gap_1()
            .child(div().text_xs().text_color(colors.disabled).child("Seed"))
            .child(
                div()
                    .flex()
                    .flex_row()
                    .items_center()
                    .gap_1()
                    .child(
                        small_chip("seed-m1", "-1", false, enabled, colors).on_click(cx.listener(
                            move |this, _, _, cx| {
                                if enabled {
                                    this.adjust_seed(-1, cx);
                                }
                            },
                        )),
                    )
                    .child(
                        div()
                            .min_w(px(44.))
                            .text_center()
                            .text_sm()
                            .font_weight(FontWeight::SEMIBOLD)
                            .child(self.train_config.seed.to_string()),
                    )
                    .child(
                        small_chip("seed-p1", "+1", false, enabled, colors).on_click(cx.listener(
                            move |this, _, _, cx| {
                                if enabled {
                                    this.adjust_seed(1, cx);
                                }
                            },
                        )),
                    ),
            )
    }

    fn render_train_log_card(&self, colors: &Colors) -> Div {
        panel_card(colors)
            .flex_1()
            .min_w(px(0.))
            .min_h(px(0.))
            .gap_3()
            .child(
                div()
                    .flex()
                    .flex_row()
                    .items_center()
                    .justify_between()
                    .child(
                        div()
                            .text_sm()
                            .font_weight(FontWeight::SEMIBOLD)
                            .child("学習ログ"),
                    )
                    .when_some(self.train_status.clone(), |row, status| {
                        row.child(div().text_xs().text_color(colors.disabled).child(status))
                    }),
            )
            .child(
                div()
                    .id("train-log-scroll")
                    .flex_1()
                    .min_h(px(0.))
                    .overflow_y_scroll()
                    .p_3()
                    .rounded_lg()
                    .bg(colors.background)
                    .border_1()
                    .border_color(colors.border)
                    .flex()
                    .flex_col()
                    .gap_1()
                    .when(self.train_logs.is_empty(), |box_el| {
                        box_el.child(
                            div().text_xs().text_color(colors.disabled).child(
                                "「学習を開始」を押すと、学習の進捗ログがここに表示されます。",
                            ),
                        )
                    })
                    .children(
                        self.train_logs
                            .iter()
                            .map(|line| div().text_xs().child(line.clone())),
                    ),
            )
    }

    fn render_evaluation_card(&self, colors: &Colors) -> Div {
        panel_card(colors)
            .w(px(420.))
            .flex_shrink_0()
            .min_h(px(0.))
            .gap_3()
            .child(
                div()
                    .text_sm()
                    .font_weight(FontWeight::SEMIBOLD)
                    .child("評価グラフ (artifacts)"),
            )
            .child(
                div()
                    .id("evaluation-scroll")
                    .flex_1()
                    .min_h(px(0.))
                    .overflow_y_scroll()
                    .flex()
                    .flex_col()
                    .gap_3()
                    .child(render_chart_slot(
                        "正解率の推移 (training_history.png)",
                        self.artifacts_status.training_history_png.clone(),
                        colors,
                    ))
                    .child(render_chart_slot(
                        "混同行列 (confusion_matrix.png)",
                        self.artifacts_status.confusion_matrix_png.clone(),
                        colors,
                    )),
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
                ActiveTab::Predict => self.render_predict_tab(&colors, cx),
                ActiveTab::Train => self.render_train_tab(&colors, cx),
            })
    }
}

#[derive(Clone, Copy)]
enum DirectoryTarget {
    Data,
    Artifacts,
}

fn panel_card(colors: &Colors) -> Div {
    div()
        .flex()
        .flex_col()
        .p_5()
        .rounded_xl()
        .bg(colors.container)
        .border_1()
        .border_color(colors.border)
}

fn status_pill(label: &'static str, color: Rgba) -> Div {
    div()
        .px_2p5()
        .py_1()
        .rounded_full()
        .bg(color.opacity(0.15))
        .border_1()
        .border_color(color.opacity(0.4))
        .text_xs()
        .font_weight(FontWeight::SEMIBOLD)
        .text_color(color)
        .child(label)
}

fn action_button(
    id: impl Into<ElementId>,
    label: &'static str,
    primary: bool,
    enabled: bool,
    colors: &Colors,
) -> Stateful<Div> {
    let base = div()
        .id(id)
        .px_3()
        .py_1p5()
        .rounded_lg()
        .text_sm()
        .font_weight(FontWeight::MEDIUM);

    if !enabled {
        return base
            .bg(colors.separator.opacity(0.4))
            .text_color(colors.disabled)
            .child(label);
    }

    if primary {
        base.bg(colors.selected)
            .text_color(colors.selected_text)
            .cursor_pointer()
            .hover(|style| style.opacity(0.9))
            .active(|style| style.opacity(0.8))
            .child(label)
    } else {
        base.bg(colors.background)
            .text_color(colors.text)
            .border_1()
            .border_color(colors.border)
            .cursor_pointer()
            .hover(|style| style.bg(colors.separator.opacity(0.45)))
            .active(|style| style.bg(colors.separator.opacity(0.7)))
            .child(label)
    }
}

fn small_chip(
    id: impl Into<ElementId>,
    label: impl Into<SharedString>,
    active: bool,
    enabled: bool,
    colors: &Colors,
) -> Stateful<Div> {
    let label: SharedString = label.into();
    let base = div()
        .id(id)
        .px_2p5()
        .py_1()
        .rounded_md()
        .text_xs()
        .border_1()
        .border_color(if active {
            colors.selected
        } else {
            colors.border
        });

    if !enabled {
        return base
            .bg(colors.background)
            .text_color(colors.disabled)
            .child(label);
    }

    base.cursor_pointer()
        .when(active, |el| {
            el.bg(colors.selected).text_color(colors.selected_text)
        })
        .when(!active, |el| {
            el.bg(colors.background)
                .text_color(colors.text)
                .hover(|style| style.bg(colors.separator.opacity(0.4)))
        })
        .child(label)
}

fn render_probability_row(entry: &ClassProbability, is_winner: bool, colors: &Colors) -> Div {
    let hand_class = HandClass::from_directory(&entry.directory);
    let accent = hand_class.map_or(colors.selected, HandClass::accent_color);
    let symbol = hand_class.map_or("", HandClass::symbol);

    div()
        .flex()
        .flex_col()
        .gap_1()
        .child(
            div()
                .flex()
                .flex_row()
                .items_center()
                .justify_between()
                .text_sm()
                .child(
                    div()
                        .flex()
                        .flex_row()
                        .items_center()
                        .gap_1p5()
                        .font_weight(if is_winner {
                            FontWeight::SEMIBOLD
                        } else {
                            FontWeight::NORMAL
                        })
                        .child(symbol)
                        .child(entry.label.clone()),
                )
                .child(
                    div()
                        .text_xs()
                        .font_weight(if is_winner {
                            FontWeight::BOLD
                        } else {
                            FontWeight::NORMAL
                        })
                        .text_color(if is_winner { accent } else { colors.disabled })
                        .child(format_percentage(entry.probability)),
                ),
        )
        .child(
            div()
                .h_2()
                .w_full()
                .rounded_full()
                .bg(colors.container)
                .overflow_hidden()
                .child(
                    div()
                        .h_full()
                        .w(relative(entry.probability.clamp(0.0, 1.0)))
                        .rounded_full()
                        .bg(accent),
                ),
        )
}

fn render_chart_slot(
    title: &'static str,
    chart: Option<std::sync::Arc<Image>>,
    colors: &Colors,
) -> Div {
    div()
        .flex()
        .flex_col()
        .gap_1p5()
        .p_3()
        .rounded_lg()
        .bg(colors.background)
        .border_1()
        .border_color(colors.border)
        .child(div().text_xs().text_color(colors.disabled).child(title))
        .child(match chart {
            Some(image) => div()
                .h(px(190.))
                .w_full()
                .flex()
                .items_center()
                .justify_center()
                .child(img(image).size_full().object_fit(ObjectFit::Contain)),
            None => div()
                .h(px(96.))
                .w_full()
                .flex()
                .items_center()
                .justify_center()
                .text_xs()
                .text_color(colors.disabled)
                .child("未生成"),
        })
}

pub fn app_window_options(cx: &gpui::App) -> gpui::WindowOptions {
    let bounds = gpui::Bounds::centered(None, gpui::size(px(1080.), px(760.)), cx);
    gpui::WindowOptions {
        window_bounds: Some(gpui::WindowBounds::Windowed(bounds)),
        titlebar: Some(gpui::TitlebarOptions {
            title: Some("グー・チョキ・パー画像分類".into()),
            ..Default::default()
        }),
        window_min_size: Some(gpui::size(px(860.), px(620.))),
        ..Default::default()
    }
}

pub fn run(cx: &mut gpui::App) {
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
