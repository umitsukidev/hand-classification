use std::sync::Arc;

use gpui::{
    Context, DefiniteLength, Div, DragMoveEvent, FontWeight, Image, ObjectFit, Stateful,
    colors::Colors, div, img, prelude::*, px, relative, rgb,
};

use crate::{
    app::{
        DirectoryTarget, HandClassificationApp, MIN_EVALUATION_WIDTH, MIN_TRAIN_LOG_WIDTH,
        TRAIN_SPLITTER_WIDTH,
    },
    components::{action_button, panel_card, small_chip, status_pill},
    types::{
        ConfusionMatrixPlotGeometry, EvaluationPreview, EvaluationPreviewOrigin,
        MisclassificationReport, MisclassifiedImage, NormalizedRect, display_path,
    },
};

const BATCH_SIZE_OPTIONS: &[usize] = &[8, 16, 32, 64];
const VALIDATION_SPLIT_OPTIONS: &[f32] = &[0.1, 0.2, 0.25, 0.3];
const MISCLASSIFIED_THUMBNAIL_SIZE: f32 = 112.0;
const EVALUATION_CHART_MAX_HEIGHT: f32 = 260.0;
const MISCLASSIFICATION_CARD_WIDTH: f32 = 120.0;

#[derive(Clone, Copy)]
struct DraggedTrainPaneSplitter;

pub fn render_train_tab(
    app: &HandClassificationApp,
    colors: &Colors,
    cx: &mut Context<HandClassificationApp>,
) -> Div {
    div()
        .flex_1()
        .min_h(px(0.))
        .flex()
        .flex_col()
        .gap_4()
        .p_6()
        .child(render_train_config_card(app, colors, cx))
        .child(
            div()
                .flex_1()
                .min_h(px(0.))
                .flex()
                .flex_row()
                .gap_0()
                .on_drag_move::<DraggedTrainPaneSplitter>(cx.listener(
                    |this, event: &DragMoveEvent<DraggedTrainPaneSplitter>, _, cx| {
                        this.resize_train_panes(event.event.position.x, event.bounds, cx);
                    },
                ))
                .child(
                    render_train_log_card(app, colors)
                        .flex_1()
                        .min_w(px(MIN_TRAIN_LOG_WIDTH))
                        .flex_basis(DefiniteLength::Fraction(1.0 - app.train_evaluation_ratio)),
                )
                .child(render_train_pane_splitter(colors))
                .child(
                    render_evaluation_card(app, colors, cx)
                        .flex_1()
                        .min_w(px(MIN_EVALUATION_WIDTH))
                        .flex_basis(DefiniteLength::Fraction(app.train_evaluation_ratio)),
                ),
        )
}

fn render_train_pane_splitter(colors: &Colors) -> Stateful<Div> {
    div()
        .id("train-pane-splitter")
        .w(px(TRAIN_SPLITTER_WIDTH))
        .h_full()
        .flex_shrink_0()
        .flex()
        .justify_center()
        .cursor_col_resize()
        .hover(|style| style.bg(colors.separator.opacity(0.4)))
        .child(div().w(px(1.)).h_full().bg(colors.border))
        .on_drag(DraggedTrainPaneSplitter, |_, _, _, cx| {
            cx.new(|_| gpui::Empty)
        })
}

fn render_train_config_card(
    app: &HandClassificationApp,
    colors: &Colors,
    cx: &mut Context<HandClassificationApp>,
) -> Div {
    let data_label = display_path(&app.train_config.data_dir, &app.project_root);
    let artifacts_label = display_path(&app.train_config.artifacts_dir, &app.project_root);
    let class_counts = app
        .dataset_summary
        .classes
        .iter()
        .map(|class| format!("{}: {}枚", class.directory, class.image_count))
        .collect::<Vec<_>>()
        .join(" / ");
    let dataset_color = if app.dataset_summary.ready {
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
                            "{} — {}",
                            class_counts, app.dataset_summary.message
                        ))),
                )
                .child(
                    div()
                        .flex()
                        .flex_row()
                        .items_center()
                        .gap_3()
                        .child(status_pill(
                            if app.dataset_summary.ready {
                                "データセット OK"
                            } else {
                                "データ不足"
                            },
                            dataset_color,
                        ))
                        .child(if app.train_running {
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
                                app.dataset_summary.ready,
                                colors,
                            )
                            .on_click(cx.listener(|this, _, _, cx| {
                                if this.dataset_summary.ready {
                                    this.begin_training(cx);
                                }
                            }))
                        }),
                ),
        )
        .child(
            div()
                .flex()
                .flex_row()
                .flex_wrap()
                .gap_4()
                .child(path_selector_field(
                    "pick-data-dir",
                    "学習データ (--data-dir)",
                    data_label,
                    DirectoryTarget::Data,
                    !app.train_running,
                    colors,
                    cx,
                ))
                .child(path_selector_field(
                    "pick-artifacts-dir",
                    "出力先 (--artifacts-dir)",
                    artifacts_label,
                    DirectoryTarget::Artifacts,
                    !app.train_running,
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
                .child(render_epochs_control(app, colors, cx))
                .child(render_batch_size_control(app, colors, cx))
                .child(render_validation_split_control(app, colors, cx))
                .child(render_seed_control(app, colors, cx)),
        )
}

fn path_selector_field(
    id: &'static str,
    title: &'static str,
    value: String,
    target: DirectoryTarget,
    enabled: bool,
    colors: &Colors,
    cx: &mut Context<HandClassificationApp>,
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

fn render_epochs_control(
    app: &HandClassificationApp,
    colors: &Colors,
    cx: &mut Context<HandClassificationApp>,
) -> Div {
    let enabled = !app.train_running;
    let delta_button = |id: &'static str, label: &'static str, delta: isize| {
        small_chip(id, label, false, enabled, colors).on_click(cx.listener(
            move |this, _, _, cx| {
                if enabled {
                    this.adjust_epochs(delta, cx);
                }
            },
        ))
    };

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
                .child(delta_button("ep-m10", "-10", -10))
                .child(delta_button("ep-m1", "-1", -1))
                .child(
                    div()
                        .min_w(px(44.))
                        .text_center()
                        .text_sm()
                        .font_weight(FontWeight::SEMIBOLD)
                        .child(app.train_config.epochs.to_string()),
                )
                .child(delta_button("ep-p1", "+1", 1))
                .child(delta_button("ep-p10", "+10", 10)),
        )
}

fn render_batch_size_control(
    app: &HandClassificationApp,
    colors: &Colors,
    cx: &mut Context<HandClassificationApp>,
) -> Div {
    let enabled = !app.train_running;
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
                let active = app.train_config.batch_size == size;
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

fn render_validation_split_control(
    app: &HandClassificationApp,
    colors: &Colors,
    cx: &mut Context<HandClassificationApp>,
) -> Div {
    let enabled = !app.train_running;
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
                        let active = (app.train_config.validation_split - split).abs() < 0.001;
                        small_chip(
                            ("val-split", ix),
                            format!("{split:.2}"),
                            active,
                            enabled,
                            colors,
                        )
                        .on_click(cx.listener(move |this, _, _, cx| {
                            if enabled {
                                this.train_config.validation_split = split;
                                cx.notify();
                            }
                        }))
                    }),
            ),
        )
}

fn render_seed_control(
    app: &HandClassificationApp,
    colors: &Colors,
    cx: &mut Context<HandClassificationApp>,
) -> Div {
    let enabled = !app.train_running;
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
                        .child(app.train_config.seed.to_string()),
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

fn render_train_log_card(app: &HandClassificationApp, colors: &Colors) -> Div {
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
                .when_some(app.train_status.clone(), |row, status| {
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
                .when(app.train_logs.is_empty(), |box_el| {
                    box_el.child(
                        div()
                            .text_xs()
                            .text_color(colors.disabled)
                            .child("「学習を開始」を押すと、学習の進捗ログがここに表示されます。"),
                    )
                })
                .children(
                    app.train_logs
                        .iter()
                        .map(|line| div().text_xs().child(line.clone())),
                ),
        )
}

fn render_evaluation_card(
    app: &HandClassificationApp,
    colors: &Colors,
    cx: &mut Context<HandClassificationApp>,
) -> Div {
    panel_card(colors)
        .w(px(420.))
        .flex_shrink_0()
        .min_h(px(0.))
        .gap_3()
        .child(
            div()
                .text_sm()
                .font_weight(FontWeight::SEMIBOLD)
                .child("評価結果 (artifacts)"),
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
                .child(render_misclassification_report(
                    app.artifacts_status.misclassification_report.as_ref(),
                    colors,
                    cx,
                ))
                .child(render_chart_slot(
                    "正解率の推移 (training_history.png)",
                    app.artifacts_status.training_history_png.clone(),
                    false,
                    colors,
                    cx,
                ))
                .child(render_chart_slot(
                    "混同行列 (confusion_matrix.png)",
                    app.artifacts_status.confusion_matrix_png.clone(),
                    true,
                    colors,
                    cx,
                )),
        )
}

fn render_misclassification_report(
    report: Option<&MisclassificationReport>,
    colors: &Colors,
    cx: &mut Context<HandClassificationApp>,
) -> Div {
    let section = div()
        .flex()
        .flex_col()
        .gap_2()
        .p_3()
        .rounded_lg()
        .bg(colors.background)
        .border_1()
        .border_color(colors.border)
        .child(
            div()
                .text_xs()
                .font_weight(FontWeight::SEMIBOLD)
                .child("誤分類一覧 (misclassified_images.json)"),
        );

    match report {
        Some(report) if report.images.is_empty() => {
            section.child(div().text_xs().text_color(colors.disabled).child(format!(
                "誤分類はありません（検証画像 {} 枚）",
                report.validation_count
            )))
        }
        Some(report) => {
            section
                .child(div().text_xs().text_color(colors.disabled).child(format!(
                    "{} 件 / 検証画像 {} 枚",
                    report.images.len(),
                    report.validation_count
                )))
                .child(div().flex().flex_row().flex_wrap().gap_2().children(
                    report.images.iter().map(|image| {
                        render_misclassified_image(
                            image,
                            EvaluationPreviewOrigin::MisclassificationList,
                            colors,
                            cx,
                        )
                    }),
                ))
        }
        None => section.child(
            div()
                .text_xs()
                .text_color(colors.disabled)
                .child("この学習結果の一覧はありません。学習し直すと表示されます。"),
        ),
    }
}

fn render_misclassified_image(
    image: &MisclassifiedImage,
    origin: EvaluationPreviewOrigin,
    colors: &Colors,
    cx: &mut Context<HandClassificationApp>,
) -> Stateful<Div> {
    let image_for_preview = image.clone();
    let origin_for_preview = origin.clone();
    let image_id = format!("evaluation-image-{origin:?}-{}", image.image_path.display());
    div()
        .id(image_id)
        .flex()
        .flex_col()
        .items_center()
        .gap_2()
        .w(px(MISCLASSIFICATION_CARD_WIDTH))
        .p_2()
        .rounded_md()
        .border_1()
        .border_color(colors.border)
        .child(if image.image_path.is_file() {
            div()
                .w(px(MISCLASSIFIED_THUMBNAIL_SIZE))
                .h(px(MISCLASSIFIED_THUMBNAIL_SIZE))
                .flex()
                .items_center()
                .justify_center()
                .child(
                    img(image.image_path.clone())
                        .size_full()
                        .object_fit(ObjectFit::Contain),
                )
        } else {
            div()
                .w(px(MISCLASSIFIED_THUMBNAIL_SIZE))
                .h(px(MISCLASSIFIED_THUMBNAIL_SIZE))
                .flex()
                .items_center()
                .justify_center()
                .rounded_md()
                .bg(colors.background)
                .text_xs()
                .text_color(colors.disabled)
                .child("画像なし")
        })
        .child(
            div()
                .w_full()
                .flex()
                .flex_col()
                .gap_1()
                .child(div().text_xs().child(format!("正解: {}", image.true_label)))
                .child(
                    div()
                        .text_xs()
                        .child(format!("予測: {}", image.predicted_label)),
                ),
        )
        .cursor_pointer()
        .hover(|style| style.bg(colors.separator.opacity(0.2)))
        .on_click(cx.listener(move |this, _, _, cx| {
            this.show_evaluation_image(image_for_preview.clone(), origin_for_preview.clone(), cx);
        }))
}

fn render_chart_slot(
    title: &'static str,
    chart: Option<Arc<Image>>,
    open_confusion_matrix: bool,
    colors: &Colors,
    cx: &mut Context<HandClassificationApp>,
) -> Stateful<Div> {
    let can_open = open_confusion_matrix && chart.is_some();
    let card = div()
        .id(title)
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
                .w_full()
                .max_h(px(EVALUATION_CHART_MAX_HEIGHT))
                .flex()
                .items_center()
                .justify_center()
                .child(
                    img(image)
                        .w_full()
                        .max_h(px(EVALUATION_CHART_MAX_HEIGHT))
                        .object_fit(ObjectFit::Contain),
                ),
            None => div()
                .h(px(MISCLASSIFIED_THUMBNAIL_SIZE))
                .w_full()
                .flex()
                .items_center()
                .justify_center()
                .text_xs()
                .text_color(colors.disabled)
                .child("未生成"),
        });

    if can_open {
        card.cursor_pointer()
            .hover(|style| style.bg(colors.separator.opacity(0.2)))
            .on_click(cx.listener(|this, _, _, cx| this.show_confusion_matrix(cx)))
    } else {
        card
    }
}

pub fn render_evaluation_preview(
    app: &HandClassificationApp,
    preview: EvaluationPreview,
    colors: &Colors,
    cx: &mut Context<HandClassificationApp>,
) -> Stateful<Div> {
    let back_label = evaluation_preview_back_label(&preview);
    let (title, description, body) = match preview {
        EvaluationPreview::Image { image, .. } => (
            "検証画像",
            format!(
                "正解: {} / 予測: {}",
                image.true_label, image.predicted_label
            ),
            render_evaluation_image_preview(&image, &app.project_root, colors),
        ),
        EvaluationPreview::ConfusionMatrix => {
            let report = app.artifacts_status.misclassification_report.as_ref();
            let class_count = app.artifacts_status.class_labels.len();
            let plot_geometry =
                report.and_then(|report| clickable_confusion_matrix_geometry(report, class_count));
            let can_click_cells =
                plot_geometry.is_some() && app.artifacts_status.confusion_matrix_png.is_some();
            let description = if can_click_cells {
                "混同行列のセルをクリックすると、その検証画像を表示します。".to_string()
            } else {
                "保存済みの混同行列を拡大表示しています。セル内画像を開くには、この形式で再学習してください。"
                    .to_string()
            };
            let body = render_confusion_matrix_image_preview(
                app.artifacts_status.confusion_matrix_png.clone(),
                plot_geometry,
                class_count,
                colors,
                cx,
            );
            ("混同行列", description, body)
        }
        EvaluationPreview::ConfusionMatrixCell {
            true_class_index,
            predicted_class_index,
        } => {
            let body = match app.artifacts_status.misclassification_report.as_ref() {
                Some(report) if report.correctly_classified_images.is_some() => {
                    render_matrix_cell_gallery(
                        report,
                        true_class_index,
                        predicted_class_index,
                        &app.artifacts_status.class_labels,
                        colors,
                        cx,
                    )
                }
                _ => render_cell_images_unavailable(colors),
            };
            (
                "セル内の検証画像",
                format!(
                    "正解: {} / 予測: {}",
                    matrix_class_label(&app.artifacts_status.class_labels, true_class_index,),
                    matrix_class_label(&app.artifacts_status.class_labels, predicted_class_index,)
                ),
                body,
            )
        }
    };

    let panel = div()
        .id("evaluation-preview-panel")
        .w_full()
        .h_full()
        .max_w(px(960.))
        .max_h(px(700.))
        .flex()
        .flex_col()
        .gap_4()
        .p_5()
        .rounded_xl()
        .bg(colors.container)
        .border_1()
        .border_color(colors.border)
        .on_click(|_, _, cx| cx.stop_propagation())
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
                        .gap_1()
                        .child(
                            div()
                                .text_base()
                                .font_weight(FontWeight::SEMIBOLD)
                                .child(title),
                        )
                        .child(
                            div()
                                .text_xs()
                                .text_color(colors.disabled)
                                .child(description),
                        ),
                )
                .child(
                    div()
                        .flex()
                        .flex_row()
                        .gap_2()
                        .when_some(back_label, |buttons, back_label| {
                            buttons.child(
                                action_button(
                                    "evaluation-preview-back",
                                    back_label,
                                    false,
                                    true,
                                    colors,
                                )
                                .on_click(cx.listener(
                                    |this, _, _, cx| {
                                        this.return_from_evaluation_preview(cx);
                                    },
                                )),
                            )
                        })
                        .child(
                            action_button(
                                "evaluation-preview-close",
                                "閉じる",
                                false,
                                true,
                                colors,
                            )
                            .on_click(cx.listener(|this, _, _, cx| {
                                this.close_evaluation_preview(cx);
                            })),
                        ),
                ),
        )
        .child(
            div()
                .id("evaluation-preview-body")
                .flex_1()
                .min_h(px(0.))
                .overflow_y_scroll()
                .child(body),
        );

    div()
        .id("evaluation-preview-overlay")
        .absolute()
        .inset_0()
        .size_full()
        .p_5()
        .flex()
        .items_center()
        .justify_center()
        .bg(rgb(0x020617).opacity(0.76))
        .occlude()
        .on_click(cx.listener(|this, _, _, cx| {
            this.close_evaluation_preview(cx);
        }))
        .child(panel)
}

fn evaluation_preview_back_label(preview: &EvaluationPreview) -> Option<&'static str> {
    match preview {
        EvaluationPreview::ConfusionMatrix
        | EvaluationPreview::Image {
            origin: EvaluationPreviewOrigin::MisclassificationList,
            ..
        } => None,
        EvaluationPreview::ConfusionMatrixCell { .. } => Some("混同行列に戻る"),
        EvaluationPreview::Image {
            origin: EvaluationPreviewOrigin::ConfusionMatrixCell { .. },
            ..
        } => Some("セルの画像一覧に戻る"),
    }
}

fn clickable_confusion_matrix_geometry(
    report: &MisclassificationReport,
    class_count: usize,
) -> Option<&ConfusionMatrixPlotGeometry> {
    if class_count < 2 {
        return None;
    }
    report.correctly_classified_images.as_ref()?;
    report
        .confusion_matrix_plot
        .as_ref()
        .filter(|geometry| geometry.is_valid())
}

fn render_clickable_confusion_matrix_image(
    image: Arc<Image>,
    geometry: &ConfusionMatrixPlotGeometry,
    class_count: usize,
    colors: &Colors,
    cx: &mut Context<HandClassificationApp>,
) -> Div {
    const MAX_IMAGE_HEIGHT: f32 = 520.0;
    let mut hit_targets = Vec::with_capacity(class_count * class_count);
    for true_class_index in 0..class_count {
        for predicted_class_index in 0..class_count {
            let cell = geometry
                .cell_bounds(true_class_index, predicted_class_index, class_count)
                .expect("valid confusion matrix cell bounds");
            hit_targets.push(render_confusion_matrix_cell_hit_target(
                cell,
                true_class_index,
                predicted_class_index,
                colors,
                cx,
            ));
        }
    }

    div().w_full().flex().justify_center().child(
        div()
            .id("confusion-matrix-image-hit-target")
            .debug_selector(|| "confusion-matrix-image-test".to_string())
            .relative()
            .w_full()
            .max_w(px(MAX_IMAGE_HEIGHT * geometry.aspect_ratio()))
            .max_h(px(MAX_IMAGE_HEIGHT))
            .aspect_ratio(geometry.aspect_ratio())
            .flex_shrink_0()
            .child(img(image).size_full().object_fit(ObjectFit::Fill))
            .children(hit_targets),
    )
}

fn render_confusion_matrix_cell_hit_target(
    cell: NormalizedRect,
    true_class_index: usize,
    predicted_class_index: usize,
    colors: &Colors,
    cx: &mut Context<HandClassificationApp>,
) -> Stateful<Div> {
    div()
        .id(format!(
            "confusion-matrix-cell-{true_class_index}-{predicted_class_index}"
        ))
        .debug_selector(|| {
            format!("confusion-matrix-cell-{true_class_index}-{predicted_class_index}")
        })
        .absolute()
        .left(relative(cell.left))
        .top(relative(cell.top))
        .w(relative(cell.right - cell.left))
        .h(relative(cell.bottom - cell.top))
        .cursor_pointer()
        .hover(|style| style.bg(colors.selected.opacity(0.08)))
        .on_click(cx.listener(move |this, _, _, cx| {
            this.show_confusion_matrix_cell(true_class_index, predicted_class_index, cx);
        }))
}

fn render_matrix_cell_gallery(
    report: &MisclassificationReport,
    true_class_index: usize,
    predicted_class_index: usize,
    class_labels: &[String],
    colors: &Colors,
    cx: &mut Context<HandClassificationApp>,
) -> Div {
    let origin = EvaluationPreviewOrigin::ConfusionMatrixCell {
        true_class_index,
        predicted_class_index,
    };
    let images = matrix_cell_images(
        report,
        true_class_index,
        predicted_class_index,
        class_labels,
    );

    if images.is_empty() {
        return div().flex().flex_col().gap_3().child(
            div()
                .text_sm()
                .child("このセルに該当する検証画像はありません。"),
        );
    }

    div()
        .flex()
        .flex_col()
        .gap_3()
        .child(
            div()
                .text_xs()
                .text_color(colors.disabled)
                .child(format!("{} 枚", images.len())),
        )
        .child(
            div().flex().flex_row().flex_wrap().gap_2().children(
                images
                    .iter()
                    .map(|image| render_misclassified_image(image, origin.clone(), colors, cx)),
            ),
        )
}

fn render_cell_images_unavailable(colors: &Colors) -> Div {
    div()
        .flex()
        .flex_col()
        .gap_2()
        .p_4()
        .rounded_lg()
        .bg(colors.background)
        .child(
            div()
                .text_sm()
                .child("この評価結果には正しく分類した画像一覧がありません。"),
        )
        .child(
            div()
                .text_xs()
                .text_color(colors.disabled)
                .child("再学習すると、混同行列の各セルから対象画像を確認できます。"),
        )
}

fn matrix_cell_images(
    report: &MisclassificationReport,
    true_class_index: usize,
    predicted_class_index: usize,
    class_labels: &[String],
) -> Vec<MisclassifiedImage> {
    report
        .images
        .iter()
        .chain(report.correctly_classified_images.iter().flatten())
        .filter(|image| {
            matrix_class_index(&image.true_label, class_labels) == Some(true_class_index)
                && matrix_class_index(&image.predicted_label, class_labels)
                    == Some(predicted_class_index)
        })
        .cloned()
        .collect()
}

fn matrix_class_index(label: &str, class_labels: &[String]) -> Option<usize> {
    class_labels
        .iter()
        .position(|class_label| class_label == label)
}

fn matrix_class_label(class_labels: &[String], index: usize) -> &str {
    class_labels
        .get(index)
        .map(String::as_str)
        .unwrap_or("不明")
}

fn render_evaluation_image_preview(
    image: &MisclassifiedImage,
    project_root: &std::path::Path,
    colors: &Colors,
) -> Div {
    let image_body = if image.image_path.is_file() {
        div()
            .w_full()
            .max_h(px(520.))
            .flex()
            .items_center()
            .justify_center()
            .child(
                img(image.image_path.clone())
                    .w_full()
                    .max_h(px(520.))
                    .object_fit(ObjectFit::Contain),
            )
    } else {
        div()
            .w_full()
            .h(px(280.))
            .flex()
            .items_center()
            .justify_center()
            .rounded_lg()
            .bg(colors.background)
            .text_sm()
            .text_color(colors.disabled)
            .child("画像ファイルが見つかりません。")
    };

    div()
        .flex()
        .flex_col()
        .items_center()
        .gap_4()
        .child(image_body)
        .child(
            div()
                .w_full()
                .flex()
                .flex_col()
                .gap_2()
                .child(
                    div()
                        .text_sm()
                        .font_weight(FontWeight::MEDIUM)
                        .child(format!(
                            "正解: {} → 予測: {}",
                            image.true_label, image.predicted_label
                        )),
                )
                .child(
                    div()
                        .text_xs()
                        .text_color(colors.disabled)
                        .child(display_path(&image.image_path, project_root)),
                ),
        )
}

fn render_confusion_matrix_image_preview(
    chart: Option<Arc<Image>>,
    geometry: Option<&ConfusionMatrixPlotGeometry>,
    class_count: usize,
    colors: &Colors,
    cx: &mut Context<HandClassificationApp>,
) -> Div {
    let can_click_cells = geometry.is_some() && chart.is_some();
    let preview = div()
        .flex()
        .flex_col()
        .gap_3()
        .child(match (chart, geometry) {
            (Some(image), Some(geometry)) => {
                render_clickable_confusion_matrix_image(image, geometry, class_count, colors, cx)
            }
            (Some(image), None) => div()
                .w_full()
                .max_h(px(520.))
                .flex()
                .items_center()
                .justify_center()
                .child(
                    img(image)
                        .w_full()
                        .max_h(px(520.))
                        .object_fit(ObjectFit::Contain),
                ),
            (None, _) => div()
                .w_full()
                .h(px(280.))
                .flex()
                .items_center()
                .justify_center()
                .text_sm()
                .text_color(colors.disabled)
                .child("混同行列が見つかりません。"),
        });

    if can_click_cells {
        preview
    } else {
        preview.child(
            div()
                .text_xs()
                .text_color(colors.disabled)
                .child("この評価結果にはセルを画像一覧へ対応づける情報がありません。再学習すると混同行列から確認できます。"),
        )
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::path::PathBuf;

    #[test]
    fn preview_back_labels_depend_on_navigation_context() {
        assert_eq!(
            evaluation_preview_back_label(&EvaluationPreview::ConfusionMatrix),
            None
        );
        assert_eq!(
            evaluation_preview_back_label(&EvaluationPreview::Image {
                image: MisclassifiedImage {
                    image_path: PathBuf::from("validation.png"),
                    true_label: "グー".to_string(),
                    predicted_label: "チョキ".to_string(),
                },
                origin: EvaluationPreviewOrigin::MisclassificationList,
            }),
            None
        );
        assert_eq!(
            evaluation_preview_back_label(&EvaluationPreview::ConfusionMatrixCell {
                true_class_index: 0,
                predicted_class_index: 1,
            }),
            Some("混同行列に戻る")
        );
        assert_eq!(
            evaluation_preview_back_label(&EvaluationPreview::Image {
                image: MisclassifiedImage {
                    image_path: PathBuf::from("validation.png"),
                    true_label: "グー".to_string(),
                    predicted_label: "チョキ".to_string(),
                },
                origin: EvaluationPreviewOrigin::ConfusionMatrixCell {
                    true_class_index: 0,
                    predicted_class_index: 1,
                },
            }),
            Some("セルの画像一覧に戻る")
        );
    }

    #[test]
    fn matrix_cell_images_selects_only_the_requested_true_and_predicted_classes() {
        let class_labels = ["alpha", "beta", "gamma", "delta"]
            .map(str::to_string)
            .to_vec();
        let report = MisclassificationReport {
            validation_count: 4,
            images: vec![
                MisclassifiedImage {
                    image_path: PathBuf::from("alpha-beta.png"),
                    true_label: "alpha".to_string(),
                    predicted_label: "beta".to_string(),
                },
                MisclassifiedImage {
                    image_path: PathBuf::from("invalid.png"),
                    true_label: "unknown".to_string(),
                    predicted_label: "delta".to_string(),
                },
            ],
            correctly_classified_images: Some(vec![
                MisclassifiedImage {
                    image_path: PathBuf::from("alpha-alpha.png"),
                    true_label: "alpha".to_string(),
                    predicted_label: "alpha".to_string(),
                },
                MisclassifiedImage {
                    image_path: PathBuf::from("delta-delta.png"),
                    true_label: "delta".to_string(),
                    predicted_label: "delta".to_string(),
                },
            ]),
            confusion_matrix_plot: None,
        };

        assert_eq!(matrix_cell_images(&report, 0, 1, &class_labels).len(), 1);
        assert_eq!(
            matrix_cell_images(&report, 0, 1, &class_labels)[0].image_path,
            PathBuf::from("alpha-beta.png")
        );
        assert_eq!(
            matrix_cell_images(&report, 0, 0, &class_labels)[0].image_path,
            PathBuf::from("alpha-alpha.png")
        );
        assert_eq!(
            matrix_cell_images(&report, 3, 3, &class_labels)[0].image_path,
            PathBuf::from("delta-delta.png")
        );
        assert!(matrix_cell_images(&report, 2, 3, &class_labels).is_empty());
    }
}
