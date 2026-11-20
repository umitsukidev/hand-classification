use std::sync::Arc;

use gpui::{
    Context, DefiniteLength, Div, DragMoveEvent, FontWeight, Image, ObjectFit, Stateful,
    colors::Colors, div, img, prelude::*, px, rgb,
};

use crate::{
    app::{
        DirectoryTarget, HandClassificationApp, MIN_EVALUATION_WIDTH, MIN_TRAIN_LOG_WIDTH,
        TRAIN_SPLITTER_WIDTH,
    },
    components::{action_button, panel_card, small_chip, status_pill},
    types::{
        EvaluationPreview, EvaluationPreviewOrigin, MisclassificationReport, MisclassifiedImage,
        display_path,
    },
};

const BATCH_SIZE_OPTIONS: &[usize] = &[8, 16, 32, 64];
const VALIDATION_SPLIT_OPTIONS: &[f32] = &[0.1, 0.2, 0.25, 0.3];
const MISCLASSIFIED_THUMBNAIL_SIZE: f32 = 112.0;
const EVALUATION_CHART_MAX_HEIGHT: f32 = 260.0;
const MISCLASSIFICATION_CARD_WIDTH: f32 = 128.0;

#[derive(Clone, Copy)]
struct DraggedTrainPaneSplitter;

pub fn render_train_tab(
    app: &HandClassificationApp,
    colors: &Colors,
    cx: &mut Context<HandClassificationApp>,
) -> Div {
    div()
        .relative()
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
        .when_some(app.evaluation_preview.clone(), |tab, preview| {
            tab.child(render_evaluation_preview(app, preview, colors, cx))
        })
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
                            "rock: {}枚 / scissors: {}枚 / paper: {}枚 — {}",
                            app.dataset_summary.rock_count,
                            app.dataset_summary.scissors_count,
                            app.dataset_summary.paper_count,
                            app.dataset_summary.message
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
        Some(report) => section
            .child(div().text_xs().text_color(colors.disabled).child(format!(
                "{} 件 / 検証画像 {} 枚",
                report.images.len(),
                report.validation_count
            )))
            .child(
                div().flex().flex_row().flex_wrap().gap_2().children(
                    report
                        .images
                        .iter()
                        .map(|image| render_misclassified_image(image, colors, cx)),
                ),
            ),
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
    colors: &Colors,
    cx: &mut Context<HandClassificationApp>,
) -> Stateful<Div> {
    let image_for_preview = image.clone();
    div()
        .id(format!(
            "misclassified-image-{}",
            image.image_path.display()
        ))
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
            this.show_evaluation_image(
                image_for_preview.clone(),
                EvaluationPreviewOrigin::MisclassificationList,
                cx,
            );
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

fn render_evaluation_preview(
    app: &HandClassificationApp,
    preview: EvaluationPreview,
    colors: &Colors,
    cx: &mut Context<HandClassificationApp>,
) -> Stateful<Div> {
    let (title, description, body, back_label) = match preview {
        EvaluationPreview::Image { image, origin } => {
            let back_label = if matches!(
                origin,
                EvaluationPreviewOrigin::ConfusionMatrixCell { .. }
            ) {
                "混同行列に戻る"
            } else {
                "戻る"
            };
            (
                "検証画像",
                format!("正解: {} / 予測: {}", image.true_label, image.predicted_label),
                render_evaluation_image_preview(&image, &app.project_root, colors),
                back_label,
            )
        }
        EvaluationPreview::ConfusionMatrix => (
            "混同行列",
            "検証データの正解クラスと予測クラスを表示しています。".to_string(),
            render_confusion_matrix_image_preview(
                app.artifacts_status.confusion_matrix_png.clone(),
                colors,
            ),
            "戻る",
        ),
        EvaluationPreview::ConfusionMatrixCell {
            true_class_index,
            predicted_class_index,
        } => (
            "セル内の検証画像",
            format!(
                "正解: {} / 予測: {}",
                matrix_class_label(true_class_index),
                matrix_class_label(predicted_class_index)
            ),
            div()
                .flex()
                .flex_col()
                .gap_3()
                .child(div().text_sm().child(
                    "この保存結果にはセル別の検証画像がありません。画像情報を含む学習結果を作成してください。",
                )),
            "混同行列に戻る",
        ),
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
                        .child(
                            action_button(
                                "evaluation-preview-back",
                                back_label,
                                false,
                                true,
                                colors,
                            )
                            .on_click(cx.listener(|this, _, _, cx| {
                                this.return_from_evaluation_preview(cx);
                            })),
                        )
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

fn render_confusion_matrix_image_preview(chart: Option<Arc<Image>>, colors: &Colors) -> Div {
    div()
        .flex()
        .flex_col()
        .gap_3()
        .child(match chart {
            Some(image) => div()
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
            None => div()
                .w_full()
                .h(px(280.))
                .flex()
                .items_center()
                .justify_center()
                .text_sm()
                .text_color(colors.disabled)
                .child("混同行列が見つかりません。"),
        })
        .child(
            div()
                .text_xs()
                .text_color(colors.disabled)
                .child("セルから検証画像を表示するには、画像情報を含む学習結果が必要です。"),
        )
}

fn matrix_class_label(index: usize) -> &'static str {
    ["グー", "チョキ", "パー"]
        .get(index)
        .copied()
        .unwrap_or("不明")
}
