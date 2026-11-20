use std::{
    fs,
    io::{BufRead, BufReader, Read, Write},
    path::{Path, PathBuf},
    process::{Child, ChildStdin, ChildStdout, Command, Stdio},
    sync::{Arc, Mutex},
    thread,
};

use async_channel::Receiver;
use gpui::{Image, ImageFormat};
use serde::{Deserialize, Serialize};

use crate::types::{
    ArtifactsStatus, DatasetClassCount, DatasetSummary, MisclassificationReport, PredictionResult,
    TrainConfig, TrainEvent, WorkerResponse, is_supported_image_path,
};

pub fn locate_project_root() -> PathBuf {
    if let Ok(current_dir) = std::env::current_dir() {
        for ancestor in current_dir.ancestors() {
            if ancestor.join("pyproject.toml").is_file() {
                return ancestor.to_path_buf();
            }
        }
    }

    for ancestor in Path::new(env!("CARGO_MANIFEST_DIR")).ancestors() {
        if ancestor.join("pyproject.toml").is_file() {
            return ancestor.to_path_buf();
        }
    }

    PathBuf::from(".")
}

pub fn build_cli_command(project_root: &Path) -> Command {
    let venv_bin = if cfg!(windows) {
        project_root.join(".venv/Scripts/hand-classification.exe")
    } else {
        project_root.join(".venv/bin/hand-classification")
    };

    let mut command = if venv_bin.is_file() {
        Command::new(venv_bin)
    } else {
        let mut cmd = Command::new("uv");
        cmd.arg("run").arg("hand-classification");
        cmd
    };

    command
        .current_dir(project_root)
        .env("TF_CPP_MIN_LOG_LEVEL", "2")
        .env("PYTHONUTF8", "1")
        .env("PYTHONUNBUFFERED", "1");
    command
}

fn count_images_recursive(directory: &Path) -> usize {
    let Ok(entries) = fs::read_dir(directory) else {
        return 0;
    };
    let mut count = 0;
    for entry in entries.flatten() {
        let Ok(file_type) = entry.file_type() else {
            continue;
        };
        let path = entry.path();
        if file_type.is_dir() {
            count += count_images_recursive(&path);
        } else if path.is_file() && is_supported_image_path(&path) {
            count += 1;
        }
    }
    count
}

pub fn inspect_dataset(data_dir: &Path) -> DatasetSummary {
    if !data_dir.is_dir() {
        return DatasetSummary {
            classes: Vec::new(),
            ready: false,
            message: "データフォルダが見つかりません".to_string(),
        };
    }

    let mut class_directories = fs::read_dir(data_dir)
        .into_iter()
        .flatten()
        .flatten()
        .filter_map(|entry| {
            let path = entry.path();
            path.is_dir()
                .then(|| (path, entry.file_name().to_string_lossy().into_owned()))
        })
        .collect::<Vec<_>>();
    class_directories.sort_by(|left, right| left.1.cmp(&right.1));
    let classes = class_directories
        .into_iter()
        .map(|(path, directory)| {
            let image_count = count_images_recursive(&path);
            DatasetClassCount {
                directory,
                image_count,
            }
        })
        .collect::<Vec<_>>();

    if classes.len() < 2 {
        let class_count = classes.len();
        return DatasetSummary {
            classes,
            ready: false,
            message: format!(
                "クラスディレクトリが2つ以上必要です（現在: {}個）",
                class_count
            ),
        };
    }

    let ready = classes.iter().all(|class| class.image_count >= 2);
    let message = if ready {
        "学習可能（各クラス2枚以上）".to_string()
    } else {
        "各クラスに2枚以上の画像が必要です".to_string()
    };

    DatasetSummary {
        classes,
        ready,
        message,
    }
}

fn load_png_image(path: &Path) -> Option<Arc<Image>> {
    let bytes = fs::read(path).ok()?;
    Some(Arc::new(Image::from_bytes(ImageFormat::Png, bytes)))
}

fn load_misclassification_report(artifacts_dir: &Path) -> Option<MisclassificationReport> {
    let artifact_path = artifacts_dir.join("misclassified_images.json");
    let bytes = fs::read(artifact_path).ok()?;
    serde_json::from_slice(&bytes).ok()
}

#[derive(Deserialize)]
struct ClassifierMetadata {
    class_directories: Vec<String>,
    class_labels: Vec<String>,
}

fn load_class_labels(artifacts_dir: &Path) -> Vec<String> {
    let metadata_path = artifacts_dir.join("metadata.json");
    let Ok(bytes) = fs::read(metadata_path) else {
        return Vec::new();
    };
    let Ok(metadata) = serde_json::from_slice::<ClassifierMetadata>(&bytes) else {
        return Vec::new();
    };
    if metadata.class_directories.len() < 2
        || metadata.class_directories.len() != metadata.class_labels.len()
    {
        return Vec::new();
    }
    metadata.class_labels
}

pub fn inspect_artifacts(artifacts_dir: &Path) -> ArtifactsStatus {
    let model_ready = artifacts_dir.join("hand_classifier.keras").is_file()
        && artifacts_dir.join("metadata.json").is_file();

    ArtifactsStatus {
        model_ready,
        class_labels: load_class_labels(artifacts_dir),
        training_history_png: load_png_image(&artifacts_dir.join("training_history.png")),
        confusion_matrix_png: load_png_image(&artifacts_dir.join("confusion_matrix.png")),
        misclassification_report: load_misclassification_report(artifacts_dir),
    }
}

struct ServeProcess {
    child: Child,
    stdin: ChildStdin,
    stdout: BufReader<ChildStdout>,
}

impl ServeProcess {
    fn spawn(project_root: &Path, artifacts_dir: &Path) -> Result<Self, String> {
        let mut command = build_cli_command(project_root);
        command
            .arg("serve")
            .arg("--artifacts-dir")
            .arg(artifacts_dir)
            .stdin(Stdio::piped())
            .stdout(Stdio::piped())
            .stderr(Stdio::piped());

        let mut child = command
            .spawn()
            .map_err(|error| format!("Python ワーカーを起動できませんでした: {error}"))?;

        let stdin = child
            .stdin
            .take()
            .ok_or_else(|| "ワーカーの標準入力を開けませんでした。".to_string())?;
        let stdout = child
            .stdout
            .take()
            .ok_or_else(|| "ワーカーの標準出力を開けませんでした。".to_string())?;

        // Drain stderr so the child process never blocks on a full stderr pipe.
        if let Some(stderr) = child.stderr.take() {
            thread::spawn(move || {
                let mut reader = BufReader::new(stderr);
                let mut discard = String::new();
                while reader.read_line(&mut discard).unwrap_or(0) > 0 {
                    discard.clear();
                }
            });
        }

        Ok(Self {
            child,
            stdin,
            stdout: BufReader::new(stdout),
        })
    }

    fn request<T: Serialize>(&mut self, payload: &T) -> Result<WorkerResponse, String> {
        let mut line = serde_json::to_string(payload)
            .map_err(|error| format!("リクエストのシリアライズに失敗しました: {error}"))?;
        line.push('\n');

        self.stdin
            .write_all(line.as_bytes())
            .and_then(|()| self.stdin.flush())
            .map_err(|error| format!("ワーカーへの送信に失敗しました: {error}"))?;

        let mut response_line = String::new();
        let bytes_read = self
            .stdout
            .read_line(&mut response_line)
            .map_err(|error| format!("ワーカーからの応答読み取りに失敗しました: {error}"))?;

        if bytes_read == 0 {
            return Err("Python ワーカーが予期せず終了しました。".to_string());
        }

        serde_json::from_str::<WorkerResponse>(response_line.trim()).map_err(|error| {
            format!(
                "ワーカーの応答 JSON を解釈できませんでした: {error} ({})",
                response_line.trim()
            )
        })
    }
}

impl Drop for ServeProcess {
    fn drop(&mut self) {
        let _ = self.child.kill();
        let _ = self.child.wait();
    }
}

#[derive(Clone, Default)]
pub struct WorkerClient {
    process: Arc<Mutex<Option<ServeProcess>>>,
}

#[derive(Serialize)]
struct PredictRequest<'a> {
    op: &'static str,
    image: &'a str,
    artifacts_dir: &'a str,
}

impl WorkerClient {
    pub fn predict(
        &self,
        project_root: &Path,
        artifacts_dir: &Path,
        image_path: &Path,
    ) -> Result<PredictionResult, String> {
        let request = PredictRequest {
            op: "predict",
            image: &image_path.to_string_lossy(),
            artifacts_dir: &artifacts_dir.to_string_lossy(),
        };

        let mut guard = self
            .process
            .lock()
            .map_err(|_| "ワーカー状態ロックの取得に失敗しました。".to_string())?;

        let response = match guard.as_mut() {
            Some(proc_handle) => match proc_handle.request(&request) {
                Ok(response) => response,
                Err(_) => {
                    *guard = None;
                    let mut fresh = ServeProcess::spawn(project_root, artifacts_dir)?;
                    let resp = fresh.request(&request)?;
                    *guard = Some(fresh);
                    resp
                }
            },
            None => {
                let mut fresh = ServeProcess::spawn(project_root, artifacts_dir)?;
                let resp = fresh.request(&request)?;
                *guard = Some(fresh);
                resp
            }
        };

        if !response.ok {
            return Err(response
                .error
                .unwrap_or_else(|| "推論中に不明なエラーが発生しました。".to_string()));
        }

        let predicted_label = response
            .predicted_label
            .ok_or_else(|| "応答に predicted_label が含まれていません。".to_string())?;
        let predicted_directory = response
            .predicted_directory
            .ok_or_else(|| "応答に predicted_directory が含まれていません。".to_string())?;
        let probabilities = response
            .probabilities
            .ok_or_else(|| "応答に probabilities が含まれていません。".to_string())?;

        Ok(PredictionResult {
            image_path: image_path.to_path_buf(),
            predicted_label,
            predicted_directory,
            probabilities,
        })
    }

    pub fn shutdown(&self) {
        if let Ok(mut guard) = self.process.lock() {
            *guard = None;
        }
    }
}

#[derive(Clone)]
pub struct TrainingHandle {
    child: Arc<Mutex<Option<Child>>>,
}

impl TrainingHandle {
    pub fn stop(&self) {
        if let Ok(mut guard) = self.child.lock()
            && let Some(child) = guard.as_mut()
        {
            let _ = child.kill();
        }
    }
}

fn strip_ansi_sequences(input: &str) -> String {
    let mut output = String::with_capacity(input.len());
    let mut chars = input.chars().peekable();
    while let Some(ch) = chars.next() {
        if ch == '\x1b' {
            if chars.peek() == Some(&'[') {
                chars.next();
                for next in chars.by_ref() {
                    if ('@'..='~').contains(&next) {
                        break;
                    }
                }
            }
            continue;
        }
        output.push(ch);
    }
    output
}

fn clean_log_line(raw: &str) -> Option<String> {
    let stripped = strip_ansi_sequences(raw);
    let cleaned = stripped
        .chars()
        .filter(|ch| !ch.is_control() || *ch == '\t')
        .collect::<String>()
        .trim()
        .to_string();

    if cleaned.is_empty() {
        return None;
    }

    Some(cleaned)
}

fn stream_pipe<R: Read + Send + 'static>(
    reader: R,
    sender: async_channel::Sender<TrainEvent>,
) -> thread::JoinHandle<()> {
    thread::spawn(move || {
        let mut reader = BufReader::new(reader);
        let mut buffer = Vec::new();
        let mut chunk = [0_u8; 1024];

        while let Ok(read_bytes) = reader.read(&mut chunk) {
            if read_bytes == 0 {
                break;
            }
            for &byte in &chunk[..read_bytes] {
                if byte == b'\n' || byte == b'\r' {
                    let text = String::from_utf8_lossy(&buffer);
                    if let Some(cleaned) = clean_log_line(&text) {
                        let _ = sender.send_blocking(TrainEvent::Log(cleaned));
                    }
                    buffer.clear();
                } else {
                    buffer.push(byte);
                }
            }
        }

        if !buffer.is_empty() {
            let text = String::from_utf8_lossy(&buffer);
            if let Some(cleaned) = clean_log_line(&text) {
                let _ = sender.send_blocking(TrainEvent::Log(cleaned));
            }
        }
    })
}

pub fn start_training(
    project_root: &Path,
    config: &TrainConfig,
) -> Result<(TrainingHandle, Receiver<TrainEvent>), String> {
    let mut command = build_cli_command(project_root);
    command
        .arg("train")
        .arg("--data-dir")
        .arg(&config.data_dir)
        .arg("--artifacts-dir")
        .arg(&config.artifacts_dir)
        .arg("--epochs")
        .arg(config.epochs.to_string())
        .arg("--batch-size")
        .arg(config.batch_size.to_string())
        .arg("--validation-split")
        .arg(format!("{:.2}", config.validation_split))
        .arg("--seed")
        .arg(config.seed.to_string())
        .stdin(Stdio::null())
        .stdout(Stdio::piped())
        .stderr(Stdio::piped());

    let mut child = command
        .spawn()
        .map_err(|error| format!("学習プロセスを起動できませんでした: {error}"))?;

    let stdout = child.stdout.take();
    let stderr = child.stderr.take();
    let child_arc = Arc::new(Mutex::new(Some(child)));
    let wait_child = child_arc.clone();

    let (sender, receiver) = async_channel::unbounded();

    let stdout_handle = stdout.map(|pipe| stream_pipe(pipe, sender.clone()));
    let stderr_handle = stderr.map(|pipe| stream_pipe(pipe, sender.clone()));

    thread::spawn(move || {
        if let Some(handle) = stdout_handle {
            let _ = handle.join();
        }
        if let Some(handle) = stderr_handle {
            let _ = handle.join();
        }

        let outcome = wait_child
            .lock()
            .ok()
            .and_then(|mut guard| guard.take())
            .map(|mut child| child.wait());

        match outcome {
            Some(Ok(status)) => {
                let _ = sender.send_blocking(TrainEvent::Finished {
                    success: status.success(),
                    exit_code: status.code(),
                });
            }
            Some(Err(error)) => {
                let _ = sender.send_blocking(TrainEvent::Failed(format!(
                    "学習プロセスの終了待機に失敗しました: {error}"
                )));
            }
            None => {
                let _ = sender.send_blocking(TrainEvent::Failed(
                    "学習プロセスの状態を取得できませんでした。".to_string(),
                ));
            }
        }
    });

    Ok((TrainingHandle { child: child_arc }, receiver))
}
