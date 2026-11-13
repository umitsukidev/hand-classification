mod app;
mod backend;
mod components;
mod predict_view;
mod train_view;
mod types;

use gpui::QuitMode;

fn main() {
    gpui_platform::application()
        .with_quit_mode(QuitMode::LastWindowClosed)
        .run(app::run);
}
