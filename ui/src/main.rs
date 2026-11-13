mod app;
mod backend;
mod types;

use gpui::QuitMode;

fn main() {
    gpui_platform::application()
        .with_quit_mode(QuitMode::LastWindowClosed)
        .run(app::run);
}
