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
    types::{MisclassificationReport, MisclassifiedImage, display_path},
};

const BATCH_SIZE_OPTIONS: &[usize] = &[8, 16, 32, 64];
const VALIDATION_SPLIT_OPTIONS: &[f32] = &[0.1, 0.2, 0.25, 0.3];

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
                    render_evaluation_card(app, colors)
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

fn render_evaluation_card(app: &HandClassificationApp, colors: &Colors) -> Div {
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
                    &app.project_root,
                    colors,
                ))
                .child(render_chart_slot(
                    "正解率の推移 (training_history.png)",
                    app.artifacts_status.training_history_png.clone(),
                    colors,
                ))
                .child(render_chart_slot(
                    "混同行列 (confusion_matrix.png)",
                    app.artifacts_status.confusion_matrix_png.clone(),
                    colors,
                )),
        )
}

fn render_misclassification_report(
    report: Option<&MisclassificationReport>,
    project_root: &std::path::Path,
    colors: &Colors,
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
            .children(
                report
                    .images
                    .iter()
                    .map(|image| render_misclassified_image(image, project_root, colors)),
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
    project_root: &std::path::Path,
    colors: &Colors,
) -> Div {
    let path = display_path(&image.image_path, project_root);
    div()
        .flex()
        .flex_row()
        .items_center()
        .gap_3()
        .p_2()
        .rounded_md()
        .border_1()
        .border_color(colors.border)
        .child(if image.image_path.is_file() {
            div()
                .w(px(64.))
                .h(px(64.))
                .flex_shrink_0()
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
                .w(px(64.))
                .h(px(64.))
                .flex_shrink_0()
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
                .flex_1()
                .min_w(px(0.))
                .flex()
                .flex_col()
                .gap_1()
                .child(div().text_xs().child(path))
                .child(
                    div()
                        .text_xs()
                        .font_weight(FontWeight::MEDIUM)
                        .child(format!(
                            "正解: {} → 予測: {}",
                            image.true_label, image.predicted_label
                        )),
                ),
        )
}

fn render_chart_slot(title: &'static str, chart: Option<Arc<Image>>, colors: &Colors) -> Div {
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
