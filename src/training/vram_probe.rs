//! VRAM/内存探测：用与 Learner 相同的一步训练（前向 + 损失 + 反向）估计配置是否可行。

use std::io::Write;
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::Arc;
use std::time::Duration;

use burn::tensor::{Int, Tensor, TensorData};
use burn::tensor::backend::AutodiffBackend;
use burn::train::TrainStep;

use crate::core::model::{Model, ModelConfig};
use crate::TextBatch;

fn flush_stdout() {
    let _ = std::io::stdout().flush();
}

fn print_flush(message: &str) {
    println!("{}", message);
    flush_stdout();
}

/// 探测阶段长时间无输出时，每隔几秒打印一行，避免误以为死机。
struct ProbeHeartbeat {
    done: Arc<AtomicBool>,
}

impl ProbeHeartbeat {
    fn start() -> Self {
        let done = Arc::new(AtomicBool::new(false));
        let done_flag = Arc::clone(&done);
        std::thread::spawn(move || {
            let mut tick: u32 = 0;
            loop {
                std::thread::sleep(Duration::from_secs(3));
                if done_flag.load(Ordering::Acquire) {
                    break;
                }
                tick += 1;
                print_flush(&format!(
                    "     … 仍在执行（约 {}s）— 当前为显存探测，非正式训练，无 Burn TUI",
                    tick * 3
                ));
            }
        });
        Self { done }
    }
}

impl Drop for ProbeHeartbeat {
    fn drop(&mut self) {
        self.done.store(true, Ordering::Release);
    }
}

fn run_training_step_once<B: AutodiffBackend>(
    model: &Model<B>,
    device: &B::Device,
    batch_size: usize,
    seq_len: usize,
    vocab_size: usize,
) -> bool {
    if batch_size == 0 || seq_len == 0 {
        return false;
    }
    let token = if vocab_size <= 1 { 0 } else { 1i32.min((vocab_size - 1) as i32) };
    let inputs_flat = vec![token; batch_size * seq_len];
    let targets_flat = inputs_flat.clone();
    let inputs = Tensor::<B, 2, Int>::from_data(TensorData::new(inputs_flat, [batch_size, seq_len]), device);
    let targets = Tensor::<B, 2, Int>::from_data(TensorData::new(targets_flat, [batch_size, seq_len]), device);
    let mask_data = vec![1u8; batch_size * seq_len];
    let mask = Tensor::<B, 2, Int>::from_data(TensorData::new(mask_data, [batch_size, seq_len]), device);
    let batch = TextBatch { inputs, targets, mask };
    let result = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
        let out = model.step(batch);
        drop(out);
    }));
    match result {
        Ok(()) => true,
        Err(_) => {
            print_flush("     ⚠ WGPU 内部错误（OOM/验证失败），正在重置模型状态…");
            false
        }
    }
}

/// 按顺序尝试多组 `(batch_size, seq_len)`。
///
/// - **相同 `seq_len` 时复用同一套模型**，避免每组都 `init`（减少 WGPU 重复编译 / 长时间无输出）。
/// - 每次打印后立即 `flush`，避免看起来像「卡住」。
/// - 成功时打印明确提示并返回该组配置。
///
/// 注意：部分 GPU 驱动在 OOM 时直接 abort 进程而无法被 `catch_unwind` 捕获。
#[allow(unused_assignments)]
pub fn probe_first_fitting_config<B: AutodiffBackend>(
    device: &B::Device,
    model_config: &ModelConfig,
    configs: &[(usize, usize)],
) -> Option<(usize, usize)> {
    let mut cached_seq: Option<usize> = None;
    let mut model: Option<Model<B>> = None;

    for &(batch_size, seq_len) in configs {
        print_flush(&format!("  尝试: 物理 batch = {}, seq_len = {} …", batch_size, seq_len));

        if batch_size == 0 || seq_len == 0 {
            print_flush("  ✗ 跳过（无效尺寸）");
            continue;
        }

        if cached_seq != Some(seq_len) || model.is_none() {
            print_flush(&format!(
                "     （seq_len={}：正在构建模型；WGPU 首次可能编译 shader，需等待一段时间属正常）",
                seq_len
            ));
            let mut cfg = model_config.clone();
            cfg.max_seq_len = seq_len;
            let m = {
                let _hb = ProbeHeartbeat::start();
                cfg.init::<B>(device)
            };
            drop(model.take());
            model = Some(m);
            cached_seq = Some(seq_len);
        }

        print_flush("     执行一步前向+反向（显存探测，非 Learner 训练循环）…");

        let step_ok = {
            let _hb = ProbeHeartbeat::start();
            let model_ref = model.as_ref().expect("model after cache fill");
            run_training_step_once(model_ref, device, batch_size, seq_len, model_config.vocab_size)
        };

        if step_ok {
            print_flush(&format!(
                "  ✓ 成功: 物理 batch = {}, seq_len = {} — 一步训练（前向+反向）已完成。",
                batch_size, seq_len
            ));
            print_flush("  🎯 找到合适的显存配置，即将进入正式训练阶段...");
            print_flush("  💡 接下来将显示 Burn 训练 TUI 和完整的 epoch 训练日志");
            return Some((batch_size, seq_len));
        }
        print_flush("  ✗ 失败（OOM、panic 或错误），尝试更小配置…");
        drop(model.take());
        cached_seq = None;
    }
    None
}

/// 单次探测（每次都会完整 `init` 模型；适合测试。批量探测请用 [`probe_first_fitting_config`]。）
pub fn probe_training_step_fits<B: AutodiffBackend>(
    device: &B::Device,
    model_config: &ModelConfig,
    batch_size: usize,
    seq_len: usize,
) -> bool {
    let mut cfg = model_config.clone();
    cfg.max_seq_len = seq_len;
    let model = cfg.init::<B>(device);
    run_training_step_once(&model, device, batch_size, seq_len, cfg.vocab_size)
}
