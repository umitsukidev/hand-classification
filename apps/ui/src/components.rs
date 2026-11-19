use gpui::{
    Context, Div, ElementId, FontWeight, Rgba, SharedString, Stateful, colors::Colors, div,
    prelude::*,
};

use crate::{app::HandClassificationApp, types::ActiveTab};

pub fn panel_card(colors: &Colors) -> Div {
    div()
        .flex()
        .flex_col()
        .p_5()
        .rounded_xl()
        .bg(colors.container)
        .border_1()
        .border_color(colors.border)
}

pub fn status_pill(label: &'static str, color: Rgba) -> Div {
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

pub fn action_button(
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

pub fn small_chip(
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

pub fn tab_button(
    id: &'static str,
    label: &'static str,
    tab: ActiveTab,
    current_tab: ActiveTab,
    colors: &Colors,
    cx: &mut Context<HandClassificationApp>,
) -> Stateful<Div> {
    let active = current_tab == tab;
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
