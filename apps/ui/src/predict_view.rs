use gpui::{
    ColorExt, Context, Div, ExternalPaths, FontWeight, ObjectFit, colors::Colors, div, img,
    prelude::*, px, relative, rgb,
};

use crate::{
    app::HandClassificationApp,
    components::{action_button, panel_card},
    types::{
        ActiveTab, ClassProbability, HandClass, PredictionState, display_path, format_percentage,
    },
};

pub fn render_predict_tab(
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
        .child(
            div()
                .flex_1()
                .min_h(px(0.))
                .flex()
                .flex_row()
                .gap_5()
                .child(render_image_panel(app, colors, cx))
                .child(render_result_panel(app, colors, cx)),
        )
        .when(!app.recent_predictions.is_empty(), |container| {
            container.child(render_recent_strip(app, colors, cx))
        })
}

fn render_image_panel(
    app: &HandClassificationApp,
    colors: &Colors,
    cx: &mut Context<HandClassificationApp>,
) -> Div {
    let accent = colors.selected;
    let selected_label = app
        .selected_image
        .as_ref()
        .map(|path| display_path(path, &app.project_root));

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
                        .child(
                            div()
                                .text_xs()
                                .text_color(colors.disabled)
                                .child("ファイルをドロップするか、「画像を選択」から指定します"),
                        ),
                )
                .child(
                    div()
                        .flex()
                        .flex_row()
                        .items_center()
                        .gap_2()
                        .when(app.selected_image.is_some(), |row| {
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
                .map(|drop_zone| match &app.selected_image {
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
                                col.child(div().text_xs().text_color(colors.disabled).child(label))
                            }),
                    ),
                    None => drop_zone
                        .cursor_pointer()
                        .on_click(cx.listener(|this, _, _, cx| {
                            this.pick_image_file(cx);
                        }))
                        .child(
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
                                    div().text_xs().text_color(colors.disabled).child(
                                        "またはクリックして画像ファイルを選択（JPEG / PNG / BMP / GIF）",
                                    ),
                                ),
                        ),
                }),
        )
}

fn render_result_panel(
    app: &HandClassificationApp,
    colors: &Colors,
    cx: &mut Context<HandClassificationApp>,
) -> Div {
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
                .when_some(app.selected_image.clone(), |header, path| {
                    let is_running =
                        matches!(app.prediction_state, PredictionState::Running { .. });
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

    card = match &app.prediction_state {
        PredictionState::Idle => card.child(render_idle_state(app, colors, cx)),
        PredictionState::Running { .. } => card.child(render_running_state(colors)),
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
                                        div().text_xs().text_color(colors.disabled).child("信頼度"),
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
                .when(!app.artifacts_status.model_ready, |box_el| {
                    box_el.child(
                        action_button(
                            "goto-train-error",
                            "モデル学習タブへ移動",
                            true,
                            true,
                            colors,
                        )
                        .on_click(cx.listener(|this, _, _, cx| {
                            this.switch_tab(ActiveTab::Train, cx);
                        })),
                    )
                }),
        ),
    };

    card
}

fn render_idle_state(
    app: &HandClassificationApp,
    colors: &Colors,
    cx: &mut Context<HandClassificationApp>,
) -> Div {
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
            if app.artifacts_status.model_ready {
                "学習済みモデルが検出されました。初回推論時のみ TensorFlow の読み込みに数秒かかります。"
            } else {
                "まだ学習済みモデルがありません。「モデル学習」タブから学習を実行するか、artifacts フォルダを指定してください。"
            },
        ))
        .when(!app.artifacts_status.model_ready, |box_el| {
            box_el.child(
                action_button(
                    "goto-train-idle",
                    "モデル学習タブへ移動",
                    true,
                    true,
                    colors,
                )
                .on_click(cx.listener(|this, _, _, cx| {
                    this.switch_tab(ActiveTab::Train, cx);
                })),
            )
        })
}

fn render_running_state(colors: &Colors) -> Div {
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
        )
}

fn render_recent_strip(
    app: &HandClassificationApp,
    colors: &Colors,
    cx: &mut Context<HandClassificationApp>,
) -> Div {
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
        .child(div().flex().flex_row().flex_wrap().gap_2().children(
            app.recent_predictions.iter().enumerate().map(|(ix, item)| {
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
                let is_current = app.selected_image.as_ref() == Some(&item.image_path);

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
        ))
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
