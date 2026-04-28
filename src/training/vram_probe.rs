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
    
    // 创建全1的attention_mask，表示所有位置都被关注
    let attention_mask_data = vec![1i32; batch_size * seq_len];
    let attention_mask = Tensor::<B, 2, Int>::from_data(TensorData::new(attention_mask_data, [batch_size, seq_len]), device);
    
    // 创建全0的token_type_ids，表示单一序列
    let token_type_ids_data = vec![0i32; batch_size * seq_len];
    let token_type_ids = Tensor::<B, 2, Int>::from_data(TensorData::new(token_type_ids_data, [batch_size, seq_len]), device);
    
    let batch = TextBatch { inputs, targets, mask, attention_mask, token_type_ids, images: None };
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
/// - Configs 必须按**从小到大排序**，函数将从最小配置开始尝试。
/// - **失败即停**：一旦某个配置失败（WGPU OOM），WGPU 状态被破坏，立即停止探测。
/// - **返回最佳成功配置**：即失败前最后一个成功的配置。
/// - **相同 seq_len 时复用模型**，减少 WGPU 重复编译。
///
/// 注意：WGPU OOM 会破坏设备状态，`catch_unwind` 只能捕获 Rust panic，
/// 无法恢复 WGPU 内部状态，因此失败后必须停止探测。
#[allow(unused_assignments)]
pub fn probe_first_fitting_config<B: AutodiffBackend>(
    device: &B::Device,
    model_config: &ModelConfig,
    configs: &[(usize, usize)],
) -> Option<(usize, usize)> {
    let mut cached_seq: Option<usize> = None;
    let mut model: Option<Model<B>> = None;
    let mut best_config: Option<(usize, usize)> = None;

    for &(batch_size, seq_len) in configs {
        print_flush(&format!("  尝试: 物理 batch = {}, seq_len = {} …", batch_size, seq_len));

        if batch_size == 0 || seq_len == 0 {
            print_flush("  ✗ 跳过（无效尺寸）");
            continue;
        }

        let need_new_model = cached_seq != Some(seq_len) || model.is_none();
        if need_new_model {
            print_flush(&format!(
                "     （seq_len={}：正在构建模型；WGPU 首次可能编译 shader，需等待一段时间属正常）",
                seq_len
            ));
            drop(model.take());
            cached_seq = None;

            let mut cfg = model_config.clone();
            cfg.max_seq_len = seq_len;

            let init_result = {
                let _hb = ProbeHeartbeat::start();
                std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
                    cfg.init::<B>(device)
                }))
            };

            match init_result {
                Ok(m) => {
                    model = Some(m);
                    cached_seq = Some(seq_len);
                }
                Err(_) => {
                    print_flush("  ✗ 模型构建失败（WGPU 状态异常），停止探测");
                    break;
                }
            }
        }

        print_flush("     执行一步前向+反向（显存探测，非 Learner 训练循环）…");

        let step_ok = {
            let _hb = ProbeHeartbeat::start();
            let model_ref = model.as_ref().expect("model after cache fill");
            run_training_step_once(model_ref, device, batch_size, seq_len, model_config.vocab_size)
        };

        if step_ok {
            print_flush(&format!(
                "  ✓ 成功: batch={}, seq_len={}",
                batch_size, seq_len
            ));
            best_config = Some((batch_size, seq_len));
        } else {
            print_flush(&format!(
                "  ✗ 失败: batch={}, seq_len={} — 达到显存上限，停止探测",
                batch_size, seq_len
            ));
            break;
        }
    }

    if let Some((batch_size, seq_len)) = best_config {
        // 强制遗忘模型，避免 Drop 时 WGPU 已损坏导致 panic
        // 内存会泄漏，但进程即将重启，因此可接受
        if let Some(m) = model.take() {
            std::mem::forget(m);
        }

        print_flush("");
        print_flush("==========================================");
        print_flush(&format!(
            "🎯 显存探测完成！最佳配置: batch={}, seq_len={}",
            batch_size, seq_len
        ));
        print_flush("==========================================");
        return Some((batch_size, seq_len));
    }

    print_flush("");
    print_flush("==========================================");
    print_flush("⚠️ 显存探测失败：连最小配置都无法在 GPU 上运行");
    print_flush("==========================================");
    print_flush(&format!(
        "模型参数总量: {}（约 {:.1}M）",
        model_config.num_params(),
        model_config.num_params() as f64 / 1_000_000.0
    ));
    print_flush("建议：");
    print_flush("  1. 使用更小的模型（--model-size 10m 或 30m）");
    print_flush("  2. 使用 CPU 后端训练（--backend cpu）");
    print_flush("  3. 手工设置参数并跳过探测（--no-auto-vram --batch-size 1 --max-seq-len 16）");
    print_flush("==========================================");

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
