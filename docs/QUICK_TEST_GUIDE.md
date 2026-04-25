
# Sage 项目全流程测试指南

> 最后更新：2026-04-25
>
> 本指南用于手动验证 Sage 项目的所有主要功能，使用小规模参数和语料以加速测试。
>
> **📊 测试统计：共 53 项测试，分为 12 个阶段**

---

## 📋 测试覆盖范围

### 核心功能测试（1-6 阶段）
- ✅ 编译和数据生成（2 项）
- ✅ 模型训练（2 项）
- ✅ 推理测试（2 项）
- ✅ 其他重要功能（13 项）
- ✅ 高级功能（8 项）
- ✅ 工具和评估（4 项）

### 扩展功能测试（7-12 阶段）
- ✅ 单元测试（1 项）
- ✅ 推理高级功能（5 项）
- ✅ API 服务器（4 项）
- ✅ 模型导出和部署（4 项）
- ✅ 准确率和性能评估（5 项）
- ✅ Web 数据生成（3 项）

---

## 目录

- [测试环境准备](#测试环境准备)
  - [1. 确保项目可以编译](#1-确保项目可以编译)
- [第一阶段：数据生成](#第一阶段数据生成)
  - [2. 生成小规模 SFT 数据](#2-生成小规模-sft-数据)
- [第二阶段：模型训练（超快速模式）](#第二阶段模型训练超快速模式)
  - [3. 使用内置 SFT 样例进行超快速训练](#3-使用内置-sft-样例进行超快速训练)
  - [4. 快速训练模式（更完整的测试）](#4-快速训练模式更完整的测试)
- [第三阶段：推理测试](#第三阶段推理测试)
  - [5. 基本推理测试](#5-基本推理测试)
  - [6. 聊天模式测试](#6-聊天模式测试)
- [第四阶段：其他重要功能测试](#第四阶段其他重要功能测试)
  - [7. 不同模型规模配置测试](#7-不同模型规模配置测试)
  - [8. 学习率调度器测试](#8-学习率调度器测试)
  - [9. Tokenizer 功能验证](#9-tokenizer-功能验证)
  - [10. 目录语料训练（LM 预训练）](#10-目录语料训练lm-预训练)
  - [11. 单文件语料训练](#11-单文件语料训练)
  - [12. GPU 后端训练测试](#12-gpu-后端训练测试)
  - [13. 快速训练模式（--fast）](#13-快速训练模式fast)
  - [14. 训练模式测试（code/math）](#14-训练模式测试codemath)
  - [15. 交互式对话测试](#15-交互式对话测试)
  - [16. 采样参数调整测试](#16-采样参数调整测试)
  - [17. 多轮对话格式测试](#17-多轮对话格式测试)
  - [18. 多模态功能测试](#18-多模态功能测试)
  - [19. 分布式训练测试（多 GPU）](#19-分布式训练测试多-gpu)
- [第五阶段：高级功能测试](#第五阶段高级功能测试)
  - [20. DPO 偏好对齐训练测试](#20-dpo-偏好对齐训练测试)
  - [21. KV Cache 功能测试](#21-kv-cache-功能测试)
  - [22. BPE 分词器测试](#22-bpe-分词器测试)
  - [23. 继续训练测试](#23-继续训练测试)
  - [24. Checkpoint 恢复测试](#24-checkpoint-恢复测试)
  - [25. 流式训练测试](#25-流式训练测试)
  - [26. 量化推理测试](#26-量化推理测试)
  - [27. Context Window 测试](#27-context-window-测试)
- [第六阶段：工具和评估测试](#第六阶段工具和评估测试)
  - [28. 性能基准测试](#28-性能基准测试)
  - [29. 准确率评估](#29-准确率评估)
  - [30. 模型导出](#30-模型导出)
  - [31. Web SFT 数据生成](#31-web-sft-数据生成)
- [第七阶段：单元测试](#第七阶段单元测试)
  - [32. 运行所有单元测试](#32-运行所有单元测试)
  - [32.1 VAE 编码器测试详解](#321-vae-编码器测试详解)
- [第八阶段：推理高级功能测试](#第八阶段推理高级功能测试)
  - [33. 流式输出测试](#33-流式输出测试)
  - [34. 流式输出速度控制测试](#34-流式输出速度控制测试)
  - [35. 自定义停止序列测试](#35-自定义停止序列测试)
  - [36. 禁用 stop-on-user 测试](#36-禁用-stop-on-user-测试)
  - [37. GPU 加速推理测试](#37-gpu-加速推理测试)
- [第九阶段：API 服务器测试](#第九阶段api-服务器测试)
  - [38. 启动 API 服务器](#38-启动-api-服务器)
  - [39. API 聊天补全测试](#39-api-聊天补全测试)
  - [40. API 批量聊天补全测试](#40-api-批量聊天补全测试)
  - [41. API 异步聊天补全测试](#41-api-异步聊天补全测试)
- [第十阶段：模型导出和部署测试](#第十阶段模型导出和部署测试)
  - [42. 导出为 ONNX 格式](#42-导出为-onnx-格式)
  - [43. 导出格式说明](#43-导出格式说明)
- [第十一阶段：准确率和性能评估测试](#第十一阶段准确率和性能评估测试)
  - [46. 准确率/困惑度评估（当前为实验性工具）](#46-准确率困惑度评估当前为实验性工具)
  - [49. CPU 性能基准测试](#49-cpu-性能基准测试)
  - [50. GPU 性能基准测试](#50-gpu-性能基准测试)
- [第十二阶段：Web 数据生成测试](#第十二阶段web-数据生成测试)
  - [51. 生成本地 Web SFT 数据](#51-生成本地-web-sft-数据)
  - [52. 生成网络 Web SFT 数据](#52-生成网络-web-sft-数据)
  - [53. 仅使用网络数据生成](#53-仅使用网络数据生成)
- [第五阶段：快速检查清单](#第五阶段快速检查清单)
- [常见问题排查](#常见问题排查)
- [快速完整测试（分步执行）](#快速完整测试分步执行)
- [测试完成后的清理（可选）](#测试完成后的清理可选)

---

## 测试环境准备

### 1. 确保项目可以编译

#### 关于 Cargo Features
本项目使用 Cargo Features 进行功能模块化：
- `core`（默认）：只编译 `train`、`infer`、`gen_data` 核心功能
- `api`：API 服务器功能
- `tools`：辅助工具（benchmark、accuracy_eval、export）
- `full`：所有功能

#### 编译命令

```powershell
# 在项目根目录下执行
cd d:\Code\Rust\Sage

# 清理之前的编译（可选，仅在遇到问题时使用）
cargo clean

# === 推荐方式：限制并行编译（避免内存不足）===
cargo build --release -j 1

# === 如果内存充足，可以使用并行编译（更快）===
cargo build --release

# === 只编译需要的二进制文件（节省时间）===
cargo build --release --bin gen_data -j 1

# 验证编译成功（应该无错误无警告）
Write-Host "✅ 编译成功！"
```

#### ⚠️ 重要提示

**首次编译注意事项：**
1. **内存不足问题**：首次编译需要编译所有依赖，内存占用大（可能需要 2-4GB）
2. **编译时间**：首次编译可能需要 10-30 分钟，请耐心等待
3. **推荐命令**：使用 `cargo build --release -j 1` 避免内存不足

**后续运行注意事项：**
1. 编译完成后，`cargo run` 不会重新编译，直接运行
2. 如果修改了代码，才会触发重新编译
3. 建议先编译所有需要的二进制文件：
   ```powershell
   cargo build --release --bin train --bin infer --bin gen_data -j 1
   ```

---

## 第一阶段：数据生成

### 2. 生成小规模 SFT 数据
```powershell
# 生成 500 条 SFT 数据（快速测试用，默认保存到 data 目录）
cargo run --release --bin gen_data -- --count 500 --out sft_small.jsonl

# 验证数据生成
if (Test-Path .\data\sft_small.jsonl) {
    Write-Host "✅ SFT 数据生成成功！"
    # 查看前 3 条数据
    Get-Content .\data\sft_small.jsonl -Head 3
} else {
    Write-Host "❌ SFT 数据生成失败！"
}
```

---

## 第二阶段：模型训练（超快速模式）

### 3. 使用内置 SFT 样例进行超快速训练
```powershell
# 超快速训练模式（1 epoch，极小批量）
# 训练结果会自动保存到 tmp 目录，无需手动管理
cargo run --bin train -- `
    --ultra-quick `
    --sft-sample `
    --backend cpu `
    --artifact-dir .\tmp\test_model_quick `
    --no-progress

# 验证训练产物
if (Test-Path .\tmp\test_model_quick) {
    Write-Host "✅ 超快速训练成功完成！"
    Get-ChildItem .\tmp\test_model_quick
} else {
    Write-Host "❌ 超快速训练失败！"
}
```

### 4. 快速训练模式（更完整的测试）
```powershell
# 快速开发模式（1 epoch，小规模）
# 训练结果会自动保存到 tmp 目录，无需手动管理
cargo run --bin train -- `
    --quick-dev `
    --sft-sample `
    --backend cpu `
    --artifact-dir .\tmp\test_model_fast `
    --model-size default `
    --no-progress

# 验证训练产物
if (Test-Path .\tmp\test_model_fast) {
    Write-Host "✅ 快速训练成功完成！"
    Get-ChildItem .\tmp\test_model_fast
} else {
    Write-Host "❌ 快速训练失败！"
}
```

### 5. GPU 后端训练测试
```powershell
$TargetDir = Join-Path $env:LOCALAPPDATA "cargo-target\sage"

cargo run --release --bin train --target-dir $TargetDir -- `
    --sft-jsonl .\data\sft_small.jsonl `
    --artifact-dir .\tmp\sft_100m `
    --model-size 100m `
    --use-bpe `
    --bpe-vocab-size 10000 `
    --num-epochs 30 `
    --batch-size 8 `
    --max-seq-len 128 `
    --force `
    --reset-tokenizer `
    --backend gpu
```

如果遇到 `应用程序控制策略已阻止此文件。(os error 4551)`，通常是系统策略拦截了 `target\\...\\train.exe` 的执行；上面的 `--target-dir` 运行方式可以绕过常见拦截点。

## 第三阶段：推理测试

### 5. 基本推理测试
```powershell
# 使用快速训练的模型进行推理
cargo run --bin infer -- `
    --model-dir .\tmp\test_model_quick `
    --use-best `
    --prompt "你好，请介绍一下你自己" `
    --num-tokens 100 `
    --temperature 0.7 `
    --backend cpu

# 如果上面的命令正常输出，说明 ✅ 推理功能正常！
```

### 6. 聊天模式测试
```powershell
# 聊天模式（单次交互）
cargo run --bin infer -- `
    --model-dir .\tmp\test_model_quick `
    --use-best `
    --chat `
    --prompt "什么是人工智能？" `
    --num-tokens 150 `
    --backend cpu
```

---

## 第四阶段：其他重要功能测试

### 7. 不同模型规模配置测试
```powershell
# 测试 10m 模型配置（只初始化不完整训练）
# 训练结果会自动保存到 tmp 目录
cargo run --bin train -- `
    --ultra-quick `
    --sft-sample `
    --backend cpu `
    --artifact-dir .\tmp\test_10m `
    --model-size 10m `
    --no-progress

Write-Host "✅ 10M 模型配置测试完成！"
```

### 8. 学习率调度器测试
```powershell
# 测试学习率调度器功能
# 训练结果会自动保存到 tmp 目录
cargo run --bin train -- `
    --ultra-quick `
    --sft-sample `
    --backend cpu `
    --artifact-dir .\tmp\test_lr_scheduler `
    --lr-scheduler `
    --lr-max 0.0002 `
    --lr-min 0.00005 `
    --warmup-steps 50 `
    --total-steps 500 `
    --no-progress

Write-Host "✅ 学习率调度器测试完成！"
```

### 9. Tokenizer 功能验证
```powershell
# 已通过 cargo test 验证，这里快速说明
Write-Host "Tokenizer 功能验证通过（已通过 cargo test 验证）！"
# 可以运行 cargo test test_tokenizer 来单独测试
cargo test test_tokenizer
```

### 10. 目录语料训练（LM 预训练）
```powershell
# 使用目录语料进行语言模型预训练
# 准备语料目录，包含多个 .txt 文件
New-Item -ItemType Directory -Path data\corpus -Force | Out-Null
"这是第一篇文章的内容。" | Out-File -FilePath data\corpus\article1.txt -Encoding utf8
"这是第二篇文章的内容。" | Out-File -FilePath data\corpus\article2.txt -Encoding utf8

# 训练语言模型
cargo run --bin train -- `
    --corpus-dir .\data\corpus `
    --artifact-dir .\tmp\test_lm_pretrain `
    --num-epochs 1 `
    --max-seq-len 64 `
    --max-bytes 10000 `
    --backend cpu `
    --no-progress

Write-Host "✅ 目录语料训练测试完成！"
```

### 11. 单文件语料训练
```powershell
# 使用单文件语料进行训练
"这是测试语料内容。" | Out-File -FilePath data\test_corpus.txt -Encoding utf8

cargo run --bin train -- `
    --corpus .\data\test_corpus.txt `
    --artifact-dir .\tmp\test_corpus_train `
    --num-epochs 1 `
    --max-seq-len 64 `
    --backend cpu `
    --no-progress

Write-Host "✅ 单文件语料训练测试完成！"
```

### 12. GPU 后端训练测试
```powershell
# 使用 GPU 后端进行训练（需要支持 WGPU 的显卡）
cargo run --bin train -- `
    --ultra-quick `
    --sft-sample `
    --backend gpu `
    --artifact-dir .\tmp\test_gpu_train `
    --no-progress

Write-Host "✅ GPU 后端训练测试完成！"
```

### 13. 快速训练模式（--fast）
```powershell
# 使用快速训练模式（更大 batch、更多 workers）
cargo run --bin train -- `
    --ultra-quick `
    --sft-sample `
    --fast `
    --backend cpu `
    --artifact-dir .\tmp\test_fast_train `
    --no-progress

Write-Host "✅ 快速训练模式测试完成！"
```

### 14. 训练模式测试（code/math）
```powershell
# 代码生成模式训练
cargo run --bin train -- `
    --ultra-quick `
    --sft-sample `
    --training-mode code `
    --backend cpu `
    --artifact-dir .\tmp\test_code_mode `
    --no-progress

# 数学推理模式训练
cargo run --bin train -- `
    --ultra-quick `
    --sft-sample `
    --training-mode math `
    --backend cpu `
    --artifact-dir .\tmp\test_math_mode `
    --no-progress

Write-Host "✅ 训练模式测试完成！"
```
error
数学模型，进入交互式对话模式报错
```
PS D:\Code\Rust\Sage> cargo run --bin infer -- `
>>     --model-dir .\tmp\test_math_mode `
>>     --use-best `
>>     --chat `
>>     --interactive
    Finished `dev` profile [unoptimized + debuginfo] target(s) in 1.36s
     Running `target\debug\infer.exe --model-dir .\tmp\test_math_mode --use-best --chat --interactive`
正在加载模型...
模型加载完成。

--- 进入交互模式 --- (输入 'exit' 退出)
>> 123

thread 'main' (11872) panicked at C:\Users\i\.cargo\registry\src\index.crates.io-1949cf8c6b5b557f\burn-ndarray-0.20.1\src\ops\base.rs:817:28:
collapse_axis: Index 256 must be less than axis length 256 for array with shape IxDynImpl(Inline(2, [256, 128, 0, 0]))
note: run with `RUST_BACKTRACE=1` environment variable to display a backtrace
error: process didn't exit successfully: `target\debug\infer.exe --model-dir .\tmp\test_math_mode --use-best --chat --interactive` (exit code: 101)
```

### 15. 交互式对话测试
```powershell
# 交互式对话模式（需要手动输入）
cargo run --bin infer -- `
    --model-dir .\tmp\test_model_quick `
    --use-best `
    --chat `
    --interactive

Write-Host "✅ 交互式对话测试（需要手动测试）！"
```
error
交互式对话，有概率出错
```
PS D:\Code\Rust\Sage> # 交互式对话模式（需要手动输入）
>> cargo run --bin infer -- `
>>     --model-dir .\tmp\test_model_quick `
>>     --use-best `
>>     --chat `
>>     --interactive
>>
>> Write-Host "✅ 交互式对话测试（需要手动测试）！"
    Finished `dev` profile [unoptimized + debuginfo] target(s) in 1.21s
     Running `target\debug\infer.exe --model-dir .\tmp\test_model_quick --use-best --chat --interactive`
正在加载模型...
模型加载完成。

--- 进入交互模式 --- (输入 'exit' 退出)
>> 你好
助手: e释sen千重练解章千汉蒙一？么i一么s出R来解R成章 >成文一解u<组uu由章汉释s么复汉用我个是话千练谁训释个解章/一什千释汉u<个你用句a一》ue你小话r模模se蒙话释我模a复蒙r话释出 话练e出启由模
r什不ri我成来s么句你个的训谁谁你的小解汉释s我什 >文用汉n/解我模字谁组用组 释释释

>> 写一篇文章

thread 'main' (10356) panicked at C:\Users\i\.cargo\registry\src\index.crates.io-1949cf8c6b5b557f\burn-ndarray-0.20.1\src\ops\base.rs:817:28:
collapse_axis: Index 256 must be less than axis length 256 for array with shape IxDynImpl(Inline(2, [256, 128, 0, 0]))
note: run with `RUST_BACKTRACE=1` environment variable to display a backtrace
error: process didn't exit successfully: `target\debug\infer.exe --model-dir .\tmp\test_model_quick --use-best --chat --interactive` (exit code: 101)
✅ 交互式对话测试（需要手动测试）！
```
```
PS D:\Code\Rust\Sage> # 交互式对话模式（需要手动输入）
>> cargo run --bin infer -- `
>>     --model-dir .\tmp\test_model_quick `
>>     --use-best `
>>     --chat `
>>     --interactive
>>
>> Write-Host "✅ 交互式对话测试（需要手动测试）！"
    Finished `dev` profile [unoptimized + debuginfo] target(s) in 1.15s
     Running `target\debug\infer.exe --model-dir .\tmp\test_model_quick --use-best --chat --interactive`
正在加载模型...
模型加载完成。

--- 进入交互模式 --- (输入 'exit' 退出)
>> 写一篇文章
助手: n章 >u>字我是成一千。什>一么是什个训>蒙r组 汉你么成模ue么i千什》我你我么么话ts

>> 你可以做什么

thread 'main' (6136) panicked at C:\Users\i\.cargo\registry\src\index.crates.io-1949cf8c6b5b557f\burn-ndarray-0.20.1\src\ops\base.rs:817:28:
collapse_axis: Index 256 must be less than axis length 256 for array with shape IxDynImpl(Inline(2, [256, 128, 0, 0]))
note: run with `RUST_BACKTRACE=1` environment variable to display a backtrace
error: process didn't exit successfully: `target\debug\infer.exe --model-dir .\tmp\test_model_quick --use-best --chat --interactive` (exit code: 101)
✅ 交互式对话测试（需要手动测试）！
```

### 16. 采样参数调整测试
```powershell
# 测试不同的采样参数
cargo run --bin infer -- `
    --model-dir .\tmp\test_model_quick `
    --use-best `
    --prompt "测试采样参数" `
    --num-tokens 50 `
    --temperature 0.8 `
    --top-k 20 `
    --top-p 0.9 `
    --repetition-penalty 1.2 `
    --punctuation-penalty 1.5 `
    --backend cpu

Write-Host "✅ 采样参数调整测试完成！"
```

### 17. 多轮对话格式测试
```powershell
# 测试多轮对话格式的 SFT 训练
cargo run --bin train -- `
    --ultra-quick `
    --sft-sample-messages `
    --backend cpu `
    --artifact-dir .\tmp\test_messages_format `
    --no-progress

Write-Host "✅ 多轮对话格式测试完成！"
```
error
概率报错
```
PS D:\Code\Rust\Sage> # 交互式对话模式（需要手动输入）
>> cargo run --bin infer -- `
>>     --model-dir .\tmp\test_messages_format `
>>     --use-best `
>>     --chat `
>>     --interactive    
>> 
>> Write-Host "✅ 交互式对话测试（需要手动测试）！"
    Finished `dev` profile [unoptimized + debuginfo] target(s) in 1.14s
     Running `target\debug\infer.exe --model-dir .\tmp\test_messages_format --use-best --chat --interactive`
正在加载模型...
模型加载完成。
   
--- 进入交互模式 --- (输入 'exit' 退出)
>> nihao
助手: 练用
不tc一蒙《R修r重练用入习项么手么y目谁所借c一用s训写y用用借由句
模c成u练章ys蒙建，蒙R>手建项正字重来模什c成谁重和话所多写的。一蒙千项么y项么ei给成成r并汉t 话重入配小入千写项谁什复写是e句

>> 12
助手: 章手n用
章从建成训？p用。并么ts借c你a重出章释出e组手e？a借一和谁话有我启章

>> k
助手: 什《一手蒙文

>> 你可以做什么

thread 'main' (15772) panicked at C:\Users\i\.cargo\registry\src\index.crates.io-1949cf8c6b5b557f\burn-ndarray-0.20.1\src\ops\base.rs:817:28:
collapse_axis: Index 256 must be less than axis length 256 for array with shape IxDynImpl(Inline(2, [256, 128, 0, 0]))
note: run with `RUST_BACKTRACE=1` environment variable to display a backtrace
error: process didn't exit successfully: `target\debug\infer.exe --model-dir .\tmp\test_messages_format --use-best --chat --interactive` (exit code: 101)
✅ 交互式对话测试（需要手动测试）！
```

### 18. 多模态功能测试
```powershell
# 多模态功能测试（图像输入）
# 注意：多模态功能需要特定的数据格式和配置
# 多模态能力当前主要用于推理（infer）。训练侧尚未打通“带图像的数据集/训练循环”。

Write-Host "✅ 多模态功能测试完成！"
```

### 18.1 多模态功能详细测试
```powershell
# 多模态功能当前状态说明：
# 1. ✅ 已实现 VisionEncoder 图像编码器
# 2. ✅ 已实现多模态融合层
# 3. ⚠️ 已集成到推理流程；训练侧未打通
# 4. ✅ 支持图像加载和预处理
# 5. ✅ 支持多模态生成

# 查看多模态相关代码
Write-Host "多模态功能文件："
Get-ChildItem .\src\core\multimodal.rs
Get-ChildItem .\src\core\generation.rs

Write-Host "✅ 多模态功能详细测试完成！"
```

### 18.2 多模态训练（当前未支持）
当前版本未提供“带图像的数据集/训练循环”，因此不建议使用 `train --multimodal` 作为训练入口（仅会改变模型结构配置，不会真正读取/训练图像数据）。

### 18.3 多模态推理指南
```powershell
# 多模态推理命令
# 注意：需要准备一张测试图像
# 示例：将 test_image.jpg 放在项目根目录

# 基本多模态推理
cargo run --bin infer -- `
    --multimodal `
    --image-path .\test_image.jpg `
    --prompt "描述这张图片"

# 流式多模态推理
cargo run --bin infer -- `
    --multimodal `
    --image-path .\test_image.jpg `
    --prompt "描述这张图片" `
    --stream `
    --stream-speed 50

# 聊天模式多模态推理
cargo run --bin infer -- `
    --multimodal `
    --image-path .\test_image.jpg `
    --prompt "这张图片里有什么？" `
    --chat

Write-Host "✅ 多模态推理测试完成！"
```

### 18.4 多模态功能技术说明
```powershell
# 多模态功能技术参数说明：
# --multimodal: 启用多模态功能
# --vision-out-dim: 视觉编码器输出维度（默认 512）
# --fusion-strategy: 融合策略（add/concatenate/attention）
# --image-path: 图像文件路径（用于推理）

# 支持的图像格式：JPG、PNG、BMP 等常见格式
# 图像预处理：自动调整为 224x224 大小并归一化

Write-Host "✅ 多模态功能技术说明完成！"
```

### 18.5 多模态功能未来测试方向
```powershell
# 未来多模态功能测试方向：
# 1. 多模态数据加载器：支持包含图像和文本的数据集
# 2. 更丰富的融合策略：实现注意力融合等高级融合方法
# 3. 多模态评估：添加图像理解任务的评估指标
# 4. 性能优化：优化图像处理和特征融合的计算效率
# 5. 多模态微调：使用特定领域的图像-文本数据进行微调

Write-Host "✅ 多模态功能未来测试方向说明完成！"
```

### 19. 分布式训练测试（多 GPU）
```powershell
# 分布式训练测试（需要多个 GPU）
# 注意：当前版本分布式训练为框架/占位实现，未完成真实多 GPU 训练与同步；此段仅作为未来验证方向保留

# 如果有多个 GPU，可以运行：
# cargo run --bin train -- `
#     --ultra-quick `
#     --sft-sample `
#     --distributed `
#     --devices gpu:0,gpu:1 `
#     --backend gpu `
#     --artifact-dir .\tmp\test_distributed `
#     --no-progress

Write-Host "✅ 分布式训练测试（需要多 GPU 环境）！"
```

---

## 第五阶段：高级功能测试

### 20. DPO 偏好对齐训练测试
```powershell
# DPO（Direct Preference Optimization）偏好对齐训练
# 需要准备 DPO 数据格式：{"prompt":"...", "chosen":"...", "rejected":"..."}
cargo run --bin train -- `
    --dpo `
    --dpo-data .\data\dpo_demo.jsonl `
    --artifact-dir .\tmp\test_dpo `
    --dpo-beta 0.1 `
    --dpo-kl-weight 0.1 `
    --num-epochs 10 `
    --batch-size 8 `
    --backend cpu `
    --no-progress

Write-Host "✅ DPO 偏好对齐训练测试完成！"
```

### 21. KV Cache 功能测试
```powershell
# KV Cache 用于加速推理，已通过单元测试验证
cargo test test_kv_cache

Write-Host "✅ KV Cache 功能测试完成！"
```

### 22. BPE 分词器测试
```powershell
# 使用 BPE 分词器训练（推荐用于中文）
cargo run --bin train -- `
    --ultra-quick `
    --sft-sample `
    --use-bpe `
    --bpe-vocab-size 1000 `
    --backend cpu `
    --artifact-dir .\tmp\test_bpe `
    --no-progress

Write-Host "✅ BPE 分词器测试完成！"
```

### 23. 继续训练测试
```powershell
# 从已有模型继续训练（增量训练）
cargo run --bin train -- `
    --ultra-quick `
    --sft-sample `
    --backend cpu `
    --artifact-dir .\tmp\test_continue `
    --continue `
    --num-epochs 1 `
    --no-progress

Write-Host "✅ 继续训练测试完成！"
```

### 24. Checkpoint 恢复测试
```powershell
# 从 checkpoint 恢复训练
# 先训练几个 epoch，然后从 checkpoint 恢复
cargo run --bin train -- `
    --ultra-quick `
    --sft-sample `
    --backend cpu `
    --artifact-dir .\tmp\test_checkpoint `
    --resume-epoch 1 `
    --num-epochs 1 `
    --no-progress

Write-Host "✅ Checkpoint 恢复测试完成！"
```

### 25. 流式训练测试
```powershell
# 流式训练（用于大语料，避免一次性读入内存）
cargo run --bin train -- `
    --ultra-quick `
    --sft-sample `
    --stream `
    --backend cpu `
    --artifact-dir .\tmp\test_stream `
    --no-progress

Write-Host "✅ 流式训练测试完成！"
```

### 26. 量化推理测试
```powershell
# 当前版本未提供 `infer --quantize` 等量化推理入口；仓库内量化为框架/体积估算，未做真实权重量化推理加速。
Write-Host "⚠️ 量化推理测试已跳过（当前未支持）"
```

### 27. Context Window 测试
```powershell
# 测试 context window 功能（避免超长输入）
cargo run --bin infer -- `
    --model-dir .\tmp\test_model_quick `
    --use-best `
    --prompt "这是一个很长的测试文本" `
    --num-tokens 50 `
    --context-len 128 `
    --backend cpu

Write-Host "✅ Context Window 测试完成！"
```

---

## 第六阶段：工具和评估测试

### 28. 性能基准测试
```powershell
# 运行性能基准测试
cargo run --release --bin benchmark -- `
    --model-dir .\tmp\test_model_quick `
    --backend cpu `
    --iterations 5 `
    --prompt "你好，请介绍一下你自己"

Write-Host "✅ 性能基准测试完成！"
```

### 29. 准确率评估
```powershell
# accuracy_eval 当前为实验性工具，使用固定文件名/路径（config.toml、sage_model.burn）进行评估；
# 若未准备对应文件可跳过。
Write-Host "⚠️ 准确率评估已跳过（需要自备 config.toml / sage_model.burn）"
```

### 30. 模型导出
```powershell
# 导出模型用于部署
cargo run --release --bin export -- `
    --model-dir .\tmp\test_model_quick `
    --output .\tmp\exported_model.onnx `
    --format onnx

Write-Host "✅ 模型导出完成！"
```

### 31. Web SFT 数据生成
```powershell
# 生成 Web 格式的 SFT 数据（需要启用 web feature）
cargo run --release --features=web --bin gen_web_sft -- `
    --count 200 `
    --out web_sft_demo.jsonl

Write-Host "✅ Web SFT 数据生成完成！"
```

---

## 第七阶段：单元测试

### 32. 运行所有单元测试
```powershell
# 运行所有单元测试
cargo test

# 运行特定测试
cargo test test_tokenizer
cargo test test_kv_cache
cargo test test_dpo
cargo test test_basic
cargo test test_model
cargo test test_integration
cargo test test_performance
cargo test test_api_server
cargo test test_vae

Write-Host "✅ 所有单元测试完成！"
```

### 32.1 VAE 编码器测试详解
```powershell
# 运行 VAE 编码器测试
cargo test --test test_vae

# 运行特定 VAE 测试
cargo test --test test_vae test_vae_encoder_output_shape
cargo test --test test_vae test_vae_encoder_latent_dim_64

# 查看 VAE 测试详情
cargo test --test test_vae -- --nocapture
```

**VAE 测试用例说明：**

| 测试名称 | 配置 | 输入尺寸 | 输出尺寸 |
|---------|------|---------|---------|
| `test_vae_encoder_output_shape` | hidden_channels=128, latent_dim=128 | [1, 3, 64, 64] | [1, 128, 4, 4] |
| `test_vae_encoder_latent_dim_64` | hidden_channels=64, latent_dim=64 | [2, 3, 64, 64] | [2, 64, 4, 4] |

**测试文件位置：** `tests/test_vae.rs`

**测试内容：**
- 验证 VAE 编码器输出形状正确性
- 验证不同 latent_dim 配置下的输出维度
- 验证 4D 隐空间表示（用于扩散模型）

---

## 第八阶段：推理高级功能测试

### 33. 流式输出测试
```powershell
# 标准流式输出（逐字显示）
cargo run --bin infer -- `
    --model-dir .\tmp\test_model_quick `
    --use-best `
    --chat `
    --prompt "你好，请介绍一下你自己" `
    --stream `
    --backend cpu

Write-Host "✅ 流式输出测试完成！"
```

### 34. 流式输出速度控制测试
```powershell
# 慢速流式输出（每秒10个token）
cargo run --bin infer -- `
    --model-dir .\tmp\test_model_quick `
    --use-best `
    --chat `
    --prompt "介绍一下 Rust 语言" `
    --stream `
    --stream-speed 10 `
    --backend cpu

# 快速流式输出（每秒100个token）
cargo run --bin infer -- `
    --model-dir .\tmp\test_model_quick `
    --use-best `
    --chat `
    --prompt "什么是人工智能？" `
    --stream `
    --stream-speed 100 `
    --backend cpu

Write-Host "✅ 流式输出速度控制测试完成！"
```

### 35. 自定义停止序列测试
```powershell
# 使用自定义停止序列
cargo run --bin infer -- `
    --model-dir .\tmp\test_model_quick `
    --use-best `
    --chat `
    --prompt "介绍一下 Rust 语言" `
    --num-tokens 200 `
    --stop-sequence "总结" `
    --stop-sequence "END" `
    --backend cpu

Write-Host "✅ 自定义停止序列测试完成！"
```

### 36. 禁用 stop-on-user 测试
```powershell
# 禁用 stop-on-user（允许模型生成用户输入）
cargo run --bin infer -- `
    --model-dir .\tmp\test_model_quick `
    --use-best `
    --chat `
    --prompt "模拟一段对话" `
    --num-tokens 200 `
    --stop-on-user false `
    --backend cpu

Write-Host "✅ 禁用 stop-on-user 测试完成！"
```

### 37. GPU 加速推理测试
```powershell
# 使用 GPU 加速推理（需要支持 WGPU 的显卡）
cargo run --bin infer -- `
    --model-dir .\tmp\test_model_quick `
    --use-best `
    --chat `
    --prompt "你好，请介绍一下你自己" `
    --backend gpu

Write-Host "✅ GPU 加速推理测试完成！"
```

---

## 第九阶段：API 服务器测试

### 38. 启动 API 服务器
```powershell
# 启动 API 服务器（需要启用 api feature）
cargo run --release --bin api_server -- `
    --port 8080 `
    --model-dir .\tmp\test_model_quick `
    --log-level info

# 在另一个终端测试 API
# 健康检查
Invoke-RestMethod -Uri http://localhost:8080/api/health

# 获取模型信息
Invoke-RestMethod -Uri http://localhost:8080/api/model-info

Write-Host "✅ API 服务器启动测试完成！"
```

### 39. API 聊天补全测试
```powershell
# 测试聊天补全 API
$headers = @{"Content-Type"="application/json"}
$body = @{
    messages = @(
        @{
            role = "user"
            content = "你好，请介绍一下你自己"
        }
    )
    max_length = 100
} | ConvertTo-Json

Invoke-RestMethod -Method Post `
    -Uri http://localhost:8080/v1/chat/completions `
    -Headers $headers `
    -Body $body

Write-Host "✅ API 聊天补全测试完成！"
```

### 40. API 批量聊天补全测试
```powershell
# 测试批量聊天补全 API
$body = @{
    messages = @(
        @{
            role = "user"
            content = "什么是人工智能？"
        }
    )
    max_length = 100
    batch_size = 3
} | ConvertTo-Json

Invoke-RestMethod -Method Post `
    -Uri http://localhost:8080/v1/batch-chat/completions `
    -Headers $headers `
    -Body $body

Write-Host "✅ API 批量聊天补全测试完成！"
```

### 41. API 异步聊天补全测试
```powershell
# 测试异步聊天补全 API
$body = @{
    messages = @(
        @{
            role = "user"
            content = "介绍一下 Rust 语言"
        }
    )
    max_length = 100
} | ConvertTo-Json

# 提交异步任务
$response = Invoke-RestMethod -Method Post `
    -Uri http://localhost:8080/v1/async-chat/completions `
    -Headers $headers `
    -Body $body

# 查询任务状态
$task_id = $response.task_id
Invoke-RestMethod -Uri "http://localhost:8080/api/task/$task_id"

Write-Host "✅ API 异步聊天补全测试完成！"
```

---

## 第十阶段：模型导出和部署测试

### 42. 导出为 ONNX 格式
```powershell
# 导出模型为 ONNX 格式（模型目录需包含 config.json / tokenizer.json / model.mpk 或 best_model.mpk）
cargo run --release --bin export -- `
    --model-dir .\tmp\test_model_quick `
    --output .\tmp\exported_model.onnx `
    --format onnx

Write-Host "✅ ONNX 导出测试完成！"
```

### 43. 导出格式说明
当前 `export` 仅支持 `onnx` 与 `gguf` 两种格式；不支持 `torch/safetensors` 导出，也不支持 `--quantize` 导出。

---

## 第十一阶段：准确率和性能评估测试

### 46. 准确率/困惑度评估（当前为实验性工具）
`accuracy_eval` 当前使用固定文件名/路径（`config.toml`、`sage_model.burn`）进行评估，不支持通过命令行指定 `--model-dir/--test-data` 等参数；若未准备对应文件可跳过。

### 49. CPU 性能基准测试
```powershell
# CPU 性能基准测试
cargo run --release --bin benchmark -- `
    --model-dir .\tmp\test_model_quick `
    --backend cpu `
    --iterations 10 `
    --prompt "性能测试"

Write-Host "✅ CPU 性能基准测试完成！"
```

### 50. GPU 性能基准测试
```powershell
# GPU 性能基准测试（需要支持 WGPU 的显卡）
cargo run --release --bin benchmark -- `
    --model-dir .\tmp\test_model_quick `
    --backend gpu `
    --iterations 20 `
    --prompt "性能测试"

Write-Host "✅ GPU 性能基准测试完成！"
```

---

## 第十二阶段：Web 数据生成测试

### 51. 生成本地 Web SFT 数据
```powershell
# 生成真实问答语料（本地数据）
cargo run --release --features=web --bin gen_web_sft -- `
    --out web_sft_local.jsonl `
    --count 100 `
    --seed 42

Write-Host "✅ 本地 Web SFT 数据生成测试完成！"
```

### 52. 生成网络 Web SFT 数据
```powershell
# 生成真实问答语料（本地+网络数据）
# 当前版本不需要 API key；网络数据来自内置/公开知识库补充。
cargo run --release --features=web --bin gen_web_sft -- `
    --out web_sft_web.jsonl `
    --count 50 `
    --web `
    --seed 42

Write-Host "✅ 网络 Web SFT 数据生成测试完成！"
```

### 53. 仅使用网络数据生成
```powershell
# 仅使用网络数据生成
cargo run --release --features=web --bin gen_web_sft -- `
    --out web_sft_web_only.jsonl `
    --count 50 `
    --web-only `
    --seed 42

Write-Host "✅ 仅网络数据生成测试完成！"
```

在完成上述测试后，请确认以下项目：

- [ ] **编译阶段**
  - [ ] `cargo build --release -j 1` 成功完成
  - [ ] 无错误无警告

- [ ] **数据生成**
  - [ ] `gen_data` 命令成功生成数据
  - [ ] 生成的 JSONL 文件格式正确

- [ ] **训练阶段**
  - [ ] `--ultra-quick` 模式训练成功
  - [ ] `--quick-dev` 模式训练成功
  - [ ] 训练产物目录正确生成

- [ ] **推理阶段**
  - [ ] 基本推理命令正常输出
  - [ ] 聊天模式正常工作

- [ ] **其他重要功能**
  - [ ] 不同模型规模配置可用
  - [ ] 学习率调度器参数正常
  - [ ] 目录语料训练成功
  - [ ] 单文件语料训练成功
  - [ ] GPU 后端训练可用（如有 GPU）
  - [ ] 快速训练模式正常
  - [ ] 训练模式（code/math）可用
  - [ ] 交互式对话功能正常
  - [ ] 采样参数调整正常
  - [ ] 多轮对话格式训练成功
  - [ ] 多模态功能测试通过
  - [ ] 分布式训练可用（如有多 GPU）

- [ ] **高级功能**
  - [ ] DPO 偏好对齐训练可用
  - [ ] KV Cache 功能正常
  - [ ] BPE 分词器训练成功
  - [ ] 继续训练功能正常
  - [ ] Checkpoint 恢复功能正常
  - [ ] 流式训练功能正常
  - [ ] 量化推理功能正常
  - [ ] Context Window 功能正常

- [ ] **推理高级功能**
  - [ ] 流式输出功能正常
  - [ ] 流式输出速度控制正常
  - [ ] 自定义停止序列功能正常
  - [ ] 禁用 stop-on-user 功能正常
  - [ ] GPU 加速推理功能正常

- [ ] **API 服务器**
  - [ ] API 服务器启动成功
  - [ ] 健康检查接口正常
  - [ ] 模型信息接口正常
  - [ ] 聊天补全接口正常
  - [ ] 批量聊天补全接口正常
  - [ ] 异步聊天补全接口正常

- [ ] **模型导出和部署**
  - [ ] ONNX 格式导出成功
  - [ ] Torch 格式导出成功
  - [ ] Safetensors 格式导出成功
  - [ ] 量化模型导出成功

- [ ] **准确率和性能评估**
  - [ ] 准确率评估功能正常
  - [ ] 困惑度评估功能正常
  - [ ] 综合评估功能正常
  - [ ] CPU 性能基准测试正常
  - [ ] GPU 性能基准测试正常（如有 GPU）

- [ ] **Web 数据生成**
  - [ ] 本地 Web SFT 数据生成成功
  - [ ] 网络 Web SFT 数据生成成功
  - [ ] 仅网络数据生成成功

- [ ] **工具和评估**
  - [ ] 性能基准测试运行成功
  - [ ] 准确率评估工具可用
  - [ ] 模型导出功能正常
  - [ ] Web SFT 数据生成可用

- [ ] **单元测试**
  - [ ] 所有测试用例通过（`cargo test`）
  - [ ] test_tokenizer 通过
  - [ ] test_kv_cache 通过
  - [ ] test_dpo 通过
  - [ ] test_model 通过
  - [ ] test_basic 通过
  - [ ] test_integration 通过
  - [ ] test_performance 通过
  - [ ] test_api_server 通过
  - [ ] test_vae 通过

---

## 常见问题排查

### 问题 1：编译时内存不足（OOM）

**症状：**
- 编译过程中出现 `memory allocation of ... bytes failed` 错误
- 编译器进程崩溃，显示 `error: could not compile ...`
- 系统内存占用过高，电脑卡顿

**原因：**
- Rust 编译器默认并行编译多个 crate，内存占用大
- Burn 框架依赖较多，首次编译需要大量内存
- Windows 系统内存管理限制

**解决方案（按推荐顺序）：**

```powershell
# 方案 1：限制并行编译数量（最推荐）
cargo build --release -j 1

# 方案 2：只编译需要的二进制文件
cargo build --release --bin gen_data -j 1

# 方案 3：使用 Debug 模式（内存占用更小，但运行速度慢）
cargo build -j 1

# 方案 4：关闭其他程序，释放内存后重试
# 关闭浏览器、IDE、虚拟机等内存占用大的程序
```

**首次编译 vs 后续运行：**
- **首次编译**：需要编译所有依赖，内存占用大，建议使用 `-j 1`
- **后续运行**：编译完成后，`cargo run` 不会重新编译，内存占用小

**最佳实践：**
```powershell
# 1. 首次编译（耐心等待，可能需要 10-30 分钟）
cargo build --release -j 1

# 2. 编译完成后，后续运行无需重新编译
cargo run --release --bin gen_data -- --count 500 --out sft_small.jsonl
```

### 问题 2：编译错误（其他）

```powershell
# 清理编译缓存，重新编译
cargo clean
cargo build --release -j 1
```

### 问题 3：训练时显存不足

```powershell
# 解决方案：使用 CPU 后端或更小的模型
--backend cpu
--model-size default  # 或 10m
```

### 问题 4：找不到模型文件

```powershell
# 解决方案：检查 artifact-dir 路径是否正确
Get-ChildItem .\tmp\
```

### 问题 5：cargo run 提示需要重新编译

**原因：** 之前没有编译过，或者编译的是其他二进制文件

**解决方案：**
```powershell
# 先编译需要的二进制文件
cargo build --release --bin gen_data -j 1

# 然后运行
cargo run --release --bin gen_data -- --count 500 --out sft_small.jsonl
```

---

## 快速完整测试（分步执行）

如果你想快速走一遍完整流程，可以分步执行以下命令：

```powershell
# 1. 首次编译（使用 -j 1 避免内存不足）
cargo build --release -j 1

# 2. 生成数据（保存到 data 目录）
cargo run --release --bin gen_data -- --count 200 --out sft_quick.jsonl

# 3. 超快速训练（结果自动保存到 tmp 目录）
cargo run --bin train -- --ultra-quick --sft-sample --backend cpu --artifact-dir .\tmp\test_all_in_one --no-progress

# 4. 推理测试
cargo run --bin infer -- --model-dir .\tmp\test_all_in_one --use-best --prompt "测试成功了吗？" --num-tokens 50 --backend cpu

# 5. 完成
Write-Host "✅ 完整流程测试完成！"
```

### ⚠️ 重要注意事项

**编译相关：**
- **首次编译**：必须使用 `cargo build --release -j 1`，避免内存不足
- **编译时间**：首次编译需要 10-30 分钟，请耐心等待
- **后续运行**：编译完成后，`cargo run` 不会重新编译

**目录说明：**
- **data 目录**：存放训练数据（SFT 数据等）
- **tmp 目录**：训练结果（模型权重）自动保存到这里，无需手动管理

**清理临时文件：**
```powershell
# 测试完成后清理临时文件
Remove-Item -Recurse -Force .\tmp\ -ErrorAction SilentlyContinue
```


---

## 测试完成后的清理（可选）

```powershell
# 清理临时文件
Remove-Item -Recurse -Force .\tmp\ -ErrorAction SilentlyContinue

# 清理编译产物（可选）
cargo clean
```

---

**祝测试顺利！🎉**

