# 主流大模型与推理框架版本适配对应关系（2026年8月）

## 一、概述

2024—2026 年是大语言模型（LLM）架构演进最剧烈的时期。以 **MoE（混合专家）**、**MLA（多头潜在注意力）**、**DSA（DeepSeek 稀疏注意力）**、**FP8/NVFP4/MXFP4 量化**、**MTP（多 token 预测）** 和 **超长上下文（128K–10M）** 为代表的技术创新，倒逼推理框架以近乎月级的节奏快速跟进。**模型与框架版本错配** 是生产环境中最常见的事故根源——轻则性能不达标、功能缺失（如思考模式无法触发、工具调用解析错误），重则 OOM 崩溃、精度损失（量化 kernel 不匹配）、甚至直接报错无法加载（架构未注册）。

本报告系统梳理了主流模型族与推理框架的版本对应关系，核心交付物为 **第四节「模型 × 框架适配对应关系总表」**，并给出关键适配注意事项与选型建议。

**符号说明**：✅ = 原生支持；⚠️ = 部分支持 / 有已知问题 / 需特定配置；❌ = 不支持 / 待适配；待确认 = 数据不足未下定论。

---

## 二、主流模型版本概览

### 2.1 Meta Llama

| 模型 | 发布时间 | 参数规模 | 架构特点 | 上下文 |
|------|----------|----------|----------|--------|
| Llama 3.1 | 2024-07-23 | 8B / 70B / 405B | Dense decoder-only, GQA, LlamaForCausalLM | 128K |
| Llama 3.2 (text) | 2024-09-25 | 1B / 3B | Dense, LlamaForCausalLM | 128K |
| Llama 3.2 Vision | 2024-09-25 | 11B / 90B | 多模态 ViT 图像编码器, MllamaForConditionalGeneration | 128K |
| Llama 3.3 | 2024-12-06 | 70B | Dense, 同 3.1 70B 架构, 由 405B 蒸馏 | 128K |
| Llama 4 Scout | 2025-04-05 | 109B 总 / 17B 激活 | MoE 16 专家, 原生多模态早期融合, iRoPE | 10M |
| Llama 4 Maverick | 2025-04-05 | 400B 总 / 17B 激活 | MoE 128 专家, 原生多模态, 官方 FP8 权重 | 1M |
| Llama 4 Behemoth | 2025-04（预告） | ~2T 总 / 288B 激活 | MoE 16 专家, 多模态含视频 | ~1M（权重未发布） |

### 2.2 Alibaba Qwen

| 模型 | 发布时间 | 参数规模 | 架构特点 | 上下文 |
|------|----------|----------|----------|--------|
| Qwen2 | 2024-07 | Dense 0.5B–72B; MoE 57B-A14B | GQA, RoPE base 1M, SwiGLU, QKV bias, DCA+YaRN | 128K |
| Qwen2.5 | 2024-09 | Dense 0.5B–72B; Coder/Math 变体 | 同 Qwen2 骨架, 32K 原生可扩 128K | 128K |
| QwQ-32B | 2024-11-28 | 32B Dense | 推理模型, 长 CoT, 基于 Qwen2.5-32B | 32K |
| Qwen3 | 2025-04-29 | Dense 0.6B–32B; MoE 30B-A3B / 235B-A22B | QK-Norm, 去 QKV bias, MoE 128 专家 Top-8, **混合思考模式** | 128K (8B/32B 原生 32K) |
| Qwen3-Next | 2025-09-10 | 80B-A3B MoE | 混合注意力 75% Gated DeltaNet + 25% Full, 512 专家, MTP | 256K→1M |
| Qwen3-VL | 2025-09-23 | Dense 2B–32B; MoE 30B-A3B / 235B-A22B | Interleaved-MRoPE, DeepStack ViT-G/14, 每尺寸含 Instruct+Thinking | 256K→1M |

### 2.3 DeepSeek

| 模型 | 发布时间 | 参数规模 | 架构特点 | 上下文 |
|------|----------|----------|----------|--------|
| DeepSeek-V2 | 2024-05-07 | 236B 总 / 8.9B 激活 | **首创 MLA** + DeepSeekMoE, KV 压缩~93% | 128K |
| DeepSeek-V2.5 | 2024-09-05 | 236B / 21B 激活 | V2-Chat + Coder-V2 统一合并 | 128K |
| DeepSeek-V3 | 2024-12-26 | 671B 总 / 37B 激活 | MoE + MLA, **FP8 混合精度训练**, MTP, 无辅助损失负载均衡 | 128K |
| DeepSeek-R1 | 2025-01-20 | 671B / 37B 激活 | 推理模型, 基于 V3 Base, GRPO 四阶段训练 | 128K |
| DeepSeek-R1-0528 | 2025-05-28 | 671B / 37B 激活 | R1 静默升级, 上下文扩至~164K | ~164K |
| DeepSeek-V3.1 | 2025-08-21 | 671B / 37B 激活 | **混合推理**(thinking/non-thinking), UE8M0 FP8, CoT 压缩 | 128K |
| DeepSeek-V3.2-Exp | 2025-09-29 | 671B / 37B 激活 | **DSA 稀疏注意力**, O(kL) 复杂度, ~1.6% 传统注意力计算量 | 128K |
| DeepSeek-V3.2 | 2025-12-01 | 671B / 37B 激活 | DSA 稳定版, V3.2-Speciale 高算力推理变体 | 128K (160K Ascend) |
| DeepSeek-V4 | 2026-04-24 预览 / 2026-08-13 稳定 | V4-Flash 284B / 13B 激活 | 继续 MLA + MoE + MTP, Dynamo 框架优化 | — |

### 2.4 Mistral AI

| 模型 | 发布时间 | 参数规模 | 架构特点 | 上下文 |
|------|----------|----------|----------|--------|
| Mistral 7B | 2023-09-27 | 7.3B Dense | GQA, SWA(4096), SwiGLU | 8K |
| Mixtral 8x7B | 2023-12-11 | ~47B / ~13B 激活 | SMoE 8 专家 Top-2 | 32K |
| Mixtral 8x22B | 2024-04-10 | 141B / ~39B 激活 | SMoE 8 专家 | 64K |
| Mistral Large 2 | 2024-07-24 | 123B Dense | Tekken 分词器, 函数调用 | 128K |
| Codestral 22B | 2024-05-29 | 22.2B Dense | FIM, 80+ 编程语言 | 32K |
| Codestral Mamba 7B | 2024-07 | 7.3B | **Mamba2 SSM** 架构 | 128K |
| Pixtral 12B | 2024-09-17 | 12.4B | 首个多模态, 12B LLM + 400M ViT | 128K |
| Ministral 3B/8B | 2024-10-16 | 3B / 8B Dense | 边缘优化, SWA | 128K |
| Mistral Small 3.1 | 2025-03-17 | 24B + ViT | 多模态, 函数调用 | 128K |
| Magistral Small | 2025-06-10 | 24B Dense | 推理模型, [THINK] 标签 | 128K |
| Mistral Large 3 | 2025-12-02 | 675B / 41B 激活 MoE + 2.5B ViT | 与 NVIDIA/Red Hat/vLLM 共研, 原生多模态 | 256K |
| Devstral Small 2 | 2025-12-09 | 24B Dense | 代理编程, SWE-bench 68% | 256K |

### 2.5 Zhipu GLM (Z.ai / THUDM)

| 模型 | 发布时间 | 参数规模 | 架构特点 | 上下文 |
|------|----------|----------|----------|--------|
| ChatGLM-6B | 2023-03 | 6B Dense | 前缀-LM, 双向注意力 gmask | — |
| ChatGLM2-6B | 2023-06 | 6B Dense | FlashAttention, MQA, 32K | 32K |
| ChatGLM3-6B | 2023-10 | 6B Dense | GQA, 函数调用 | 32K |
| GLM-4 (旗舰, 闭源) | 2024-01-16 | 未公开 | 多模态, 工具, RAG | ~2M(API) |
| GLM-4-9B | 2024-06-06 | 9B Dense | GlmForCausalLM, GQA, RoPE, 双语 | 128K |
| GLM-4V-9B | 2024-06 | 9B | 视觉语言, VQA/OCR | 8K |
| GLM-4-32B-0414 | 2025-04-14 | 32B / 9B Dense | Glm4ForCausalLM, RL+推理合成数据 | 128K |
| GLM-Z1-32B-0414 | 2025-04-14 | 32B / 9B | 推理模型, 冷启动+扩展 RL | 128K |
| GLM-4.5 | 2025-07-28 | 355B 总 / 32B 激活 MoE | Glm4MoeForCausalLM, QK-Norm, **MTP 头**, 128 专家 Top-8, 混合思考 | 128K (96K 输出) |
| GLM-4.6 | 2025-09-30 | ~355B / 32B 激活 MoE | Muon 优化器, 改进 MTP, 472T tokens | 200K 输入 / 128K 输出 |

### 2.6 Google Gemma、Microsoft Phi 及其他

| 模型 | 发布时间 | 参数规模 | 架构特点 | 上下文 |
|------|----------|----------|----------|--------|
| Gemma 2 | 2024-06-27 | 2B / 9B / 27B | 交替 SWA+全局注意力, logit soft-capping | 8K |
| Gemma 3 | 2025-03-12 | 1B / 4B / 12B / 27B | 5:1 local/global, SigLIP 多模态 | 128K |
| Gemma 3n | 2025-06 | E2B / E4B | MatFormer 嵌套, PLE, 多模态(文+图+音+视频) | — |
| Gemma 4 | 2026-05 | e2B–31B | 多模态(视觉+工具+思考+音频), MTP | 256K |
| Phi-3 | 2024-04 | 3.8B / 7B / 14B | GQA, 合成数据训练 | 128K |
| Phi-3.5 | 2024-08 | mini 3.8B / MoE 41.9B / vision 4.2B | MoE 16 专家 Top-2 | 128K |
| Phi-4 | 2024-12-12 | 14B | 合成数据策略, 9.8T tokens | 16K |
| Phi-4-mini / multimodal | 2025-02 | 3.8B / 5.6B | Mixture-of-LoRAs 多模态 | 128K |
| Yi / Yi-1.5 | 2023-11 / 2024-05 | 6B–34B | Llama 架构兼容, 双语 | 4K |
| Baichuan2 | 2023-09-06 | 7B / 13B | W_pack QKV, NormHead, max-z loss | — |
| MiniCPM | 2024-01 | 2.4B / V系列~8B | 量化感知训练, 边缘优化 | — |

---

## 三、主流推理框架版本概览

### 3.1 vLLM

> 高吞吐 LLM 推理与服务引擎，核心为 PagedAttention。截至 2026-08 已至 **v0.27.1**，约每月一个 minor 版本。

| 版本 | 发布时间 | 关键变化 | 新增支持模型 |
|------|----------|----------|-------------|
| v0.6.0 系列 | 2024-09 至 2024-12 | chunked prefill, 多 LoRA, FP8; v0.6.6 初始 DeepSeek-V3 | DeepSeek-V2/V2-Lite, Qwen2.5, Mixtral, Gemma 2, 初始 V3 |
| v0.7.0 系列 | 2025-01 至 2025-02 | 推测解码(实验), V1 引擎 alpha, MLA 快速路径, FP8 KV cache | DeepSeek-V3(原生 MLA), DeepSeek-R1 |
| v0.8.0 系列 | 2025-03 至 2025-05 | V1 引擎默认, FlashMLA, EP/DP, Ngram 推测解码 | Gemma 3, QwQ-32B, Mistral Small 3.1 |
| v0.8.3 | 2025-04-06 | Llama 4 Day-0(V1 only), 滑窗注意力 | Llama 4 Scout/Maverick |
| v0.8.5 | 2025-04-29 | Qwen3 dense + MoE | Qwen3, Qwen3MoE |
| v0.9.0 系列 | 2025-05 至 2025-07 | PyTorch 2.7, PP, EAGLE3/Medusa, DeepSeek-R1-FP4 | DeepSeek-V3.1, MiMo-7B |
| v0.10.0 系列 | 2025-07 至 2025-09 | GLM-4.5 MoE | GLM-4.5 / GLM-4.5-Air |
| v0.11.0 系列 | 2025-10 至 2025-11 | DeepSeek-V3.2 架构 | DeepSeek-V3.2 |
| v0.12–v0.18 | 2025-12 至 2026-04 | GLM5 零代码, EPLB, FA4, B300/GB300 | Qwen3-Next, GLM5 |
| v0.19.0 | 2026-04-07 | Model Runner V2 生产, ViT CUDA graph | **Gemma 4** Day-0 |
| v0.20–v0.21 | 2026-05 至 2026-06 | 2-bit KV cache, CUDA 13.0 | Qwen3.6-35B-A3B |
| v0.25–v0.26 | 2026-07 | DeepSeek-V4 性能推升, 序列并行 | Mistral-Large-3(675B), Kimi K3, Qwen3.5 |
| v0.27.0–v0.27.1 | 2026-08-10 | PyTorch 2.13, FA4 on SM100, DeepSeek-V4 路由 kernel | Kimi K3, Qwen3.5, K-EXAONE-2.0 |

### 3.2 SGLang

> LMSYS/sgl-project 团队开发的高性能框架，核心为 RadixAttention。截至 2026-08 已至 **v0.5.17**。**DeepSeek 官方推荐推理引擎**。

| 版本 | 发布时间 | 关键变化 | 新增支持模型 |
|------|----------|----------|-------------|
| v0.1 | 2024-01 | RadixAttention, 结构化输出 DSL | Llama 2/3, Mistral |
| v0.2 | 2024-07 | FlashInfer 集成, Llama3 加速 | Llama 3/3.1 |
| v0.3.x | 2024-09 | 混合 chunk prefill, 多模态, 7x MLA | DeepSeek V2(首个 MLA), LLaVA-OneVision |
| v0.4.0 | 2024-12 | 零开销批调度器, 缓存感知负载均衡 | — |
| v0.4.2 | 2025-01-27 | DeepSeek V3/R1 Day-0, DeepEP/DeepGEMM, MTP, PD 分离 | DeepSeek V3, R1 |
| v0.4.5 | 2025-04-07 | Llama 4 完整支持, EAGLE3, 1M 上下文 | Llama 4 Scout/Maverick |
| v0.4.6 | 2025-04-27 | FA3 默认, DeepGEMM FP8, Qwen3 Day-0 | Qwen3 (dense + MoE) |
| v0.4.9 | 2025-07-06 | GLM-4.5 Day-0(推荐≥0.4.9.post6) | GLM-4.5, GLM-4.5-Air |
| v0.4.10 | 2025-07-31 | GLM-4.5 正式版, xAI Grok | — |
| v0.5.1 | 2025-08-23 | RDMA 批量传输, Qwen-1M 上下文 | — |
| v0.5.6 | 2025-12-03 | Gateway 组件, 生产稳定候选 | — |
| v0.5.8 | 2026-01-23 | GLM-4.7 Flash Day-0, EPD 弹性编码器 | GLM-4.7 Flash |
| v0.5.9 | 2026-02-24 | LoRA 加载重叠(-78% TTFT), TRT-LLM NSA kernel, 原生 Anthropic API | Kimi-K2.5, GLM-5, Qwen3.5 |
| v0.5.10 | 2026-04-06 | Piecewise CUDA graph 默认, Elastic EP, 原生 MLX 后端 | GLM-5(原生), Nemotron-3-Super |
| v0.5.12 | 2026-05-16 | DeepSeek V4 Day-0, HiSparse, NVFP4 MoE, MegaMoE | DeepSeek V4, Gemma 4 MTP |
| v0.5.13 | 2026-06-13 | Spec V2 默认, Nemotron DP attention | Nemotron 3 Ultra |
| v0.5.15 | 2026-07-10 | DCP for MLA, FlashKDA | Qwen3.6 NVFP4 |
| v0.5.16 | 2026-07-25 | DSpark 推测解码(383.7 tok/s) | Inkling(975B MoE) |
| v0.5.17 | 2026-08-08 | Kimi K3 Day-0, Rust 前端, DWDP MoE | Kimi K3, MiniMax-H3 |

### 3.3 NVIDIA TensorRT-LLM

> NVIDIA GPU 专用 LLM 推理优化库。截至 2026-08 已至 **v1.3.0rc24**（2025-09 毕业 1.0.0 稳定版）。v1.2 起移除 C++ TRT 后端，**PyTorch 后端为唯一运行时**。

| 版本 | 发布时间 | 关键变化 | 新增支持模型 |
|------|----------|----------|-------------|
| 0.10.0 | 2024-04 | TRT 10.0.1, executor API, paged KV, 权重剥离 | DBRX, Qwen2, LLaMA 3, Phi-3-Mini |
| 0.12.0 | 2024-07 | MoE LoRA, FP8 OOTB MoE, TP+EP, ReDrafter | LLaMA 3.1, GLM4, Qwen2 |
| 0.13.0 | 2024-09 | Lookahead 解码, FP8 FMHA Ada | Gemma 2, LLaMA 3.1 |
| 0.15.0 | 2024-11 | EAGLE/Medusa 推测解码, trtllm-serve | Llama 3.2/3.2-Vision, Deepseek-v2 |
| 0.16.0 | 2024-11 | XGrammar, W4A8 Ada, FP8 Llama-3.2 VLM | Qwen2-VL |
| 0.17.0 | 2025-01 | Blackwell B200, NVFP4 GEMM, 实验 PyTorch 工作流 | NVFP4 Llama/Mixtral |
| 0.19.0 | 2025-05 | C++ 运行时开源, FP8 MLA, FlashMLA, DeepEP, EAGLE-3 | DeepSeek V3/R1, Gemma3, Qwen2.5-VL |
| 0.20.0 | 2025-06 | XQA 开源, LoRA, DeepSeek-R1 W4A8 Hopper | Qwen3, Eagle-3 for LLaMA4 |
| 0.21.0 | 2025-08 | 大规模 EP MoE, DeepSeek FP8 cubins, NIXL | Gemma3 VLM |
| 1.0.0 | 2025-09 | **PyTorch 后端稳定/默认**, MXFP8-MXFP4, DeepEP FP4, EPLB+MTP | Mistral3.1 VLM, Qwen3 MoE(TRT) |
| 1.1.0 | 2025-12 | KV-cache Connector API, MLA 复用+卸载, B300/GB300 | GPT-OSS, Hunyuan-MoE |
| 1.2.0 | 2026-02 | **TRT 后端移除**, PyTorch 唯一, DGX Spark | GPT-OSS 验证, Qwen3-Next beta |
| 1.3.0rc21 | 2026-07-15 | DeepSeek V4, Qwen3.5-VL, Gemma 4 12B, Qwen3.6 NVFP4 | DeepSeek V4, Gemma 4, MiniMax M3 |
| 1.3.0rc24 | 2026-08-12 | MEGAMOE_CUTEDSL, Marlin NVFP4 Ada, Kimi K3 分离式 | Kimi K3, MiniCPM-V 4.6, FLUX.2 |

### 3.4 llama.cpp (ggml / GGUF)

> C/C++ 推理库，使用递增 build 号（bNNNN）而非日历版本。约 8-12 builds/天。截至 2026-08-13 已至 **b10405**。

| 版本(build) | 发布时间 | 关键变化 | 新增支持模型 |
|-------------|----------|----------|-------------|
| b3300 | 2023-12 | 首个 MoE 支持 | Mixtral 8x7B (PR #4406) |
| b3600 | 2024-08-17 | Mamba-2 SSM, Metal SSM kernel | Mamba-2, Mamba-Codestral-7B |
| b4400 | 2024-12-31 | MoE matmul 加固, DeepSeek-V3 集成中 | DeepSeek-V3 basic(PR #11049, 2025-01) |
| b4877 | 2025-03-12 | Gemma 3 day-zero | Gemma 3 (1B/4B/12B/27B) |
| b5050 | 2025-04 初 | Llama 4 MoE+多模态, MoE 层卸载 | Llama 4 Scout/Maverick |
| b5100-era | 2025-04-09 | Qwen3/Qwen3MoE, DeepSeek V2/V3 **MLA 完整实现** | Qwen3 (PR #12828), DeepSeek MLA (PR #12801) |
| b6100 | 2025-08-06 | 原生 MXFP4, GLM-4.5 MoE, --n-cpu-moe | GLM-4.5/Air (PR #14939), gpt-oss(MXFP4) |
| b9658 | 2026-06-15 | HIP graphs 默认, 推测解码 checkpoint, Anthropic API | LFM2-Audio |
| b10405 | 2026-08-13 | HIP 贪心修复, MoE imatrix 优化, pocket-tts | EXAONE 4.5, Qwen 解析加固 |

### 3.5 Ollama

> 本地 LLM 运行时（Go），封装 GGUF runner（v0.30.0 起直接集成 llama.cpp 引擎 + MLX 引擎）。提供 OpenAI 兼容 API 与 CLI/TUI。截至 2026-08 已至 **v0.32.11（pre-release）**。

| 版本 | 发布时间 | 关键变化 | 新增支持模型 |
|------|----------|----------|-------------|
| 0.3.0 | 2024-07-25 | **BREAKING** Modelfile 语法, 工具调用 | Llama 3.1 |
| 0.3.12 | 2024-09-28 | ARM Windows, 可恢复 pull | Llama 3.2(1B/3B), Qwen2.5-Coder |
| 0.4.0 | 2024-11-21 | **BREAKING** 内容寻址存储迁移 | — |
| 0.4.2 | 2024-11-23 | 首个多模态支持 | LLaVA, BakLLaVA |
| 0.5.1 | 2024-12-07 | — | Llama 3.3 70B |
| 0.5.5 | 2025-01-14 | — | DeepSeek-V3 671B |
| 0.5.7 | 2025-01-17 | — | DeepSeek-R1 |
| 0.5.9 | 2025-02-14 | — | Phi-4 |
| 0.6.0 | 2025-03-12 | — | Gemma 3 |
| 0.6.6 | 2025-04-08 | — | GLM-4, IBM Granite 3.3 |
| 0.6.7 | 2025-05-02 | — | Llama 4, Qwen3 |
| 0.7.0 | 2025-05-16 | 重写多模态引擎 | Qwen3/Qwen2.5-VL 稳定 |
| 0.8.0 | 2025-05-28 | 流式工具调用 | — |
| 0.9.0 | 2025-05-31 | **Thinking Mode**(/set think, API think:true) | DeepSeek-R1-0528 |
| 0.9.3 | 2025-06-17 | — | Gemma 3n |
| 0.10.0 | 2025-08 | 新 macOS/Win 应用, 多 GPU +10-30% | — |
| 0.11.0 | 2025-08-06 | MXFP4, 代理函数调用 | gpt-oss 20B/120B |
| 0.12.x | 2025-09 至 2025-11 | 云模型, Web Search API | — |
| 0.13.0 | 2025-11-19 | Bench 工具, /v1/responses API | DeepSeek-OCR, Ministral-3, Mistral-Large-3 |
| 0.14.0 | 2026-01-14 | 实验 Agent CLI, Anthropic /v1/messages, REQUIRES 指令 | GLM-4.7-Flash, Z-Image-Turbo |
| 0.15.0 | 2026-01-28 | ollama launch(Claude Code/Codex 零配置) | — |
| 0.24.0 | 2026-05-14 | Codex App 集成 | kimi-k2.6, GLM-5.1, Gemma4:31b, Qwen3.6 |
| 0.30.0 | 2026-05-13 | **MAJOR**: 直接 llama.cpp 引擎, GGUF-from-HF, ⚠️mllama 暂不支持 | Gemma 4 12B(0.30.3) |
| 0.31.1 | 2026-06-30 | Gemma 4 MTP 加速~90%(Apple Silicon) | — |
| 0.32.0 | 2026-07-11 | 交互式代理体验 | Laguna 2.1, Muse Glimmer, Nemotron 3.5 |

> **关键注意事项**：v0.30.0 架构切换后，Llama 3.2 Vision（mllama 类型）**暂时不可靠**，正在通过 PR 重新添加。

### 3.6 HuggingFace TGI (Text Generation Inference)

> HuggingFace 生产级推理服务器，支持连续批处理、TP、量化、多硬件后端。**2025-12-19 v3.3.7 为最终版本，2026-03-21 仓库归档，进入维护模式**。HuggingFace 建议新部署迁移至 vLLM/SGLang/llama.cpp/MLX。

| 版本 | 发布时间 | 关键变化 | 新增支持模型 |
|------|----------|----------|-------------|
| v2.0.0 | 2024-04-12 | Apache 2.0 回归, CUDA graphs, 首个 FP8 | Llava-Next, Command R+ |
| v2.1.0 | 2024-06-28 | 多 LoRA, Marlin GPTQ, 调度器 v3 | Gemma2 |
| v2.2.0 | 2024-07-23 | Flash decoding, FP8 扩展, AWQ→GPTQ-Marlin | Llama 3.1(405B), DeepSeek V2 |
| v2.3.0 | 2024-09-20 | 前缀缓存默认, FlashInfer 后端 | Qwen2 tied-embedding |
| v2.3.1 | 2024-10-03 | FP8/MoE 性能改进 | Mllama(Llama 3.2 Vision) |
| v2.4.0 | 2024-10-25 | 实验 prefill chunking, FP8 KV cache, MoE Marlin | — |
| v3.0.0 | 2024-12-09 | Chunked prefill, FP8 KV cache(compressed-tensors) | — |
| v3.0.2 | 2025-01-24 | transformers 后端, Flashinfer 0.2 | Cohere2, Qwen2-VL |
| v3.1.0 | 2025-01-31 | FP8 for MoE | DeepSeek R1(AMD+Nvidia) |
| v3.2.2 | 2025-04-06 | Torch 2.6, Gaudi 修复 | **Llama 4**, Qwen3 |
| v3.3.2 | 2025-05-30 | Gaudi OOM 修复 | Qwen3, Llama 4 Scout/Maverick 修复 |
| v3.3.7 | 2025-12-19 | **最终版本**, 维护模式 | — |
| 2026 归档 | 2026-03-21 | 仓库归档, 无新功能 | 无(Llama 4/Gemma 4/DeepSeek V4 不主动支持) |

### 3.7 LMDeploy (OpenMMLab / Shanghai AI Lab)

> 高效 LLM 部署工具，含 **TurboMind（C++ 高性能引擎）** 与 PyTorch 引擎双后端。支持 W4A16、KV INT8/INT4、FP8、推测解码、PD 分离、MoE。截至 2026-08 已至 **v0.15.0**。

| 版本 | 发布时间 | 关键变化 | 新增支持模型 |
|------|----------|----------|-------------|
| v0.6.3 | 2024-11-16 | TurboMind MoE 专家并行, column-major MoE kernel | Qwen2-MoE, Mixtral MoE AWQ |
| v0.6.4 | 2024-12-09 | KV-int8 on Ascend | DeepSeek-V2, Qwen2.5 function_call |
| v0.7.0 | 2025-01-15 | DeepSeek V3 FP8(PyTorch), MoE FP8, Cambricon 后端 | DeepSeek V3, InternLM3 |
| v0.7.0.post2 | 2025-01-27 | DeepSeek-R1 chat template | DeepSeek-R1 |
| v0.7.2 | 2025-03-19 | Flash MLA, PyTorch 多节点, 工具推理解析 | Qwen2.5-VL, Gemma3 |
| v0.8.0 | 2025-05-04 | Torch DP, DeepEP/all2all EP, MLA 优化, Qwen3 FP8 | Qwen3/Qwen3-MoE, Llama4, Phi4-mini |
| v0.9.0 | 2025-06-19 | PD 分离(DistServe), EPLB, FP8 MoE TurboMind | Qwen3 /think & /no_think |
| v0.10.0 | 2025-09-09 | TurboMind 卸载, MXFP4 GEMM, Ray MP | GPT-OSS, GLM-4-0414, GLM-4.5 |
| v0.10.1 | 2025-09-26 | ROCm/AMD, FP8xBF16 GEMM | GLM-4.5, InternVL3.5-Flash |
| v0.11.0 | 2025-12-04 | 推测解码, 上下文并行, MoE BF16 EP | Qwen3-VL |
| v0.12.0 | 2026-02-04 | Gloo comm, llm-compressor AWQ TurboMind | Intern-S1-Pro |
| v0.12.1 | 2026-02-13 | Ascend EP | GLM-4.7-Flash |
| v0.12.2 | 2026-03-18 | FP8 在线量化, MLA KV-cache 优化 | GLM5(754B), Qwen3.5 |
| v0.13.0 | 2026-05-12 | TurboQuant(KV-cache), Anthropic 端点 | Qwen3.5 Ascend, InternS2 Preview |
| v0.14.0 | 2026-06-24 | FP8 KV-cache, OpenAI Responses 端点, FA3 SM80+ | Qwen3 Omni, Mixtral 恢复 TurboMind |
| v0.15.0 | 2026-07-31 | 长上下文+MTP 前缀缓存, GDR, 无状态推理 | **DeepSeek V4**, FP8 MoE Qwen3.5 |

---

## 四、模型 × 框架 适配对应关系总表

> **说明**：单元格内容为该框架**首次支持该模型的最低版本**。✅=原生支持；⚠️=部分支持/有已知问题/需特定配置/仅特定引擎；❌=不支持/待适配；待确认=数据不足。版本冲突时取**较晚/较高版本**并在脚注说明。

### 4.1 Meta Llama 族

| 模型 | vLLM | SGLang | TRT-LLM | llama.cpp | Ollama | TGI | LMDeploy |
|------|------|--------|---------|-----------|--------|-----|----------|
| Llama 3.1 (8B/70B/405B) | ✅ v0.6.0 | ✅ v0.2 | ✅ 0.12.0 | ✅ 早期 | ✅ 0.3.0 | ✅ v2.2.0 | ✅ v0.6.0+ |
| Llama 3.2 (1B/3B text) | ✅ v0.6.0 | ✅ v0.4.x | ✅ 0.15.0 | ✅ 早期 | ✅ 0.3.12 | ✅ v2.3.0+ | ✅ v0.6.0+ |
| Llama 3.2 Vision (11B/90B) | ✅ v0.6.0+ | ✅ v0.4.x | ✅ 0.15.0(FP8 0.16.0) | ⚠️ 待确认¹ | ⚠️ 0.4.2(**0.30.x+ 损坏**)² | ✅ v2.3.1 | ❌ TurboMind不支持³ |
| Llama 3.3 70B | ✅ v0.6.0 | ✅ v0.4.x | ✅ 0.17.0+ | ✅ 早期 | ✅ 0.5.1 | ✅ v2.4.0+ | ✅ v0.6.0+ |
| Llama 4 Scout | ✅ v0.8.3(量化 v0.12.0)⁴ | ✅ v0.4.5 | ✅ 0.20.0 | ✅ ~b5050/b5423 | ✅ 0.6.7 | ✅ v3.3.2 | ⚠️ v0.8.0(PyTorch only)⁵ |
| Llama 4 Maverick | ✅ v0.8.3(量化 v0.12.0) | ✅ v0.4.5 | ✅ 0.20.0 | ✅ ~b5050/b5423 | ✅ 0.6.7 | ✅ v3.3.2 | ⚠️ v0.8.0(PyTorch only) |

### 4.2 Alibaba Qwen 族

| 模型 | vLLM | SGLang | TRT-LLM | llama.cpp | Ollama | TGI | LMDeploy |
|------|------|--------|---------|-----------|--------|-----|----------|
| Qwen2 / Qwen2.5 (dense) | ✅ v0.6.0 | ✅ v0.4.x | ✅ 0.10.0(2.5: 0.15.0) | ✅ 早期 | ✅ 0.3.12 | ✅ v2.0.2(2.5: v2.3.0) | ✅ v0.6.0+ |
| Qwen2.5-MoE | ✅ v0.6.0 | ✅ v0.4.x | ✅ 0.10.0 | ✅ 早期 | ✅ 0.3.12 | ✅ v2.3.0+ | ✅ v0.6.3 |
| QwQ-32B | ✅ v0.8.0 | ✅ v0.4.6+ | ✅ 0.20.0 | ✅ b5100+ | ✅ 0.6.x | ✅ v3.2.2 | ✅ v0.8.0 |
| Qwen3 (dense + MoE) | ✅ v0.8.5 | ✅ v0.4.6 | ✅ v1.0.0(dense)/v1.0.0(MoE) | ✅ b5100(PR #12828) | ✅ 0.6.7(稳定 0.7.0) | ✅ v3.3.2 | ✅ v0.8.0(FP8 v0.10.0) |
| Qwen3-Next (80B-A3B) | ✅ v0.13.x+ | ⚠️ 较新 main | ⚠️ v1.2.0(beta) | ✅ commit 10240+ | ⚠️ 经 llama.cpp 后端 | ❌ 待适配 | 待确认 |
| Qwen3-VL | ✅ v0.9.x+ | ✅ v0.5.x | ⚠️ v1.2.0(beta) | ✅ 2025-09 后原生 | ✅ 0.7.0+ | ❌ 待适配 | ✅ v0.11.0 |

### 4.3 DeepSeek 族

| 模型 | vLLM | SGLang | TRT-LLM | llama.cpp | Ollama | TGI | LMDeploy |
|------|------|--------|---------|-----------|--------|-----|----------|
| DeepSeek-V2 / V2.5 | ✅ v0.6.0 | ✅ v0.3.0(首个 MLA) | ✅ 0.15.0 | ✅ ~b4500(MLA b5150) | 待确认 | ✅ v2.2.0 | ✅ v0.6.4 |
| DeepSeek-V3 | ✅ v0.6.6(生产 v0.8.5) | ✅ v0.4.2(Day-0) | ✅ deepseek_v3 分支 | ✅ ~b4500(MLA b5150) | ✅ 0.5.5 | ✅ v3.1.0 | ✅ v0.7.0 |
| DeepSeek-R1 | ✅ v0.7.0(推荐 v0.8.5+) | ✅ v0.4.2(Day-0) | ✅ 0.19.0 | ✅ ~b4500(MLA b5150; MTP 2026-05) | ✅ 0.5.7(思考 0.9.0) | ✅ v3.1.0 | ✅ v0.7.0.post2 |
| DeepSeek-V3.1 | ✅ v0.9.1 | ✅ v0.5.x | ⚠️ 待确认 | 待确认 | 待确认 | ❌ | ✅ (late 2025) |
| DeepSeek-V3.2 (DSA) | ✅ v0.11.2 | ✅ v0.5.x | ⚠️ 第三方确认 | ⚠️ 待确认⁶ | 待确认 | ❌ | ✅ (late 2025) |
| DeepSeek-V4 | ✅ v0.26.0 | ✅ v0.5.12(Day-0) | ✅ 1.3.0rc21 | 待确认 | 待确认 | ❌ | ✅ v0.15.0 |

### 4.4 Mistral AI 族

| 模型 | vLLM | SGLang | TRT-LLM | llama.cpp | Ollama | TGI | LMDeploy |
|------|------|--------|---------|-----------|--------|-----|----------|
| Mistral 7B | ✅ v0.6.0 | ✅ v0.4.0+ | ✅ 0.5.0-era | ✅ 最早(GGUF v2) | ✅ 0.1.x | ✅ v2.0.0 | ✅ v0.2.0+ |
| Mixtral 8x7B/8x22B | ✅ v0.6.0(FP8) | ✅ v0.4.0+ | ✅ 0.5.0-era(FP8 0.10.0) | ✅ ~b3300(PR #4406) | ✅ 0.5.x | ✅ v1.1.0 | ✅ v0.6.0+(AWQ v0.6.3) |
| Mistral Large 2 (123B) | ✅ v0.6.0 | ✅ v0.4.x | ✅ via Llama impl | ✅ GGUF | ✅ 0.5.x | ✅ v2.4.0+ | ✅ v0.6.0+ |
| Pixtral 12B | ✅ v0.6.2 | ✅ v0.4.0+ | ⚠️ 未列入模型库 | ✅ b3400+ | ⚠️ 无官方库页 | ❌ 文本聚焦 | ❌ 未列入 |
| Mistral Small 3.1 (24B) | ✅ v0.8.0 | ✅ v0.4.x | ✅ 0.20.0 | ✅ GGUF | ✅ 0.6.x | ✅ v2.3.0+ | ✅ v0.6.0+ |
| Magistral Small | ✅ v0.8.5+ | ✅ v0.4.x | ⚠️ 未列入 | ✅ GGUF | ✅ 0.6.x | ✅ v2.3.0+ | ❌ 未列入 |
| Mistral Large 3 (675B) | ✅ v0.26.0(Transformers) | ✅ v0.4.x(MLA) | ⚠️ 待确认 | ✅ GGUF | ✅ 0.13.1 | ❌ 维护模式 | ❌ 未列入 |

### 4.5 Zhipu GLM 族

| 模型 | vLLM | SGLang | TRT-LLM | llama.cpp | Ollama | TGI | LMDeploy |
|------|------|--------|---------|-----------|--------|-----|----------|
| ChatGLM / GLM-4-9B | ✅ v0.6.0 | ✅ recent main | ✅ 0.12.0 | ✅ 早期 GGUF | ✅ 0.6.6 | ⚠️ partial(archived) | ✅ v0.6.0+ |
| GLM-4-32B-0414 | ✅ v0.6.0(Glm4ForCausalLM) | ✅ recent main | ✅ 0.12.0+ | ✅ GGUF | ✅ 0.6.6 | ⚠️ partial | ✅ v0.10.0 |
| GLM-4.5 / GLM-4.5-Air | ✅ v0.10.0(Glm4Moe) | ✅ v0.4.9.post6 | ⚠️ v1.3.0rc1(Air) | ✅ ~b6050-b6100(PR #14939) | ✅ via HF GGUF(0.6.6+) | ❌ MoE 不支持 | ✅ v0.10.1 |
| GLM-4.6 | ✅ v0.11.0+(AWQ v0.10.0) | ✅ v0.4.9.post6+ | ⚠️ 待确认 | ✅ b6100+ | ✅ via HF GGUF | ❌ | ✅ v0.10.1+ |
| GLM-4.7-Flash | ✅ v0.10.0+(同 MoE 架构) | ✅ v0.5.8(Day-0) | ⚠️ 待确认 | ✅ b6100+ | ✅ 0.14.3 | ❌ | ✅ v0.12.1 |
| GLM-5 / GLM-5.2 | ✅ v0.14.x–v0.18.x | ✅ v0.5.9/v0.5.14 | ⚠️ 待确认 | 待确认 | ✅ 0.24.x | ❌ | ✅ v0.12.2 |

### 4.6 Google Gemma、Microsoft Phi 及其他

| 模型 | vLLM | SGLang | TRT-LLM | llama.cpp | Ollama | TGI | LMDeploy |
|------|------|--------|---------|-----------|--------|-----|----------|
| Gemma 2 | ✅ v0.6.0 | ✅ v0.4.x | ✅ 0.13.0 | ✅ b3280+ | ✅ 0.1.x | ✅ v2.1.0 | ✅ v0.4.x |
| Gemma 3 | ✅ v0.8.0 | ✅ v0.5.3+ | ✅ 0.19.0(VLM 0.21.0) | ✅ b4877(day-zero) | ✅ 0.6.0 | ✅ v3.2.0 | ✅ v0.7.2 |
| Gemma 3n | ✅ v0.10+(无 LoRA/PP) | ⚠️ 较新 build | ⚠️ 未明确 | ✅ b4000+ | ✅ 0.9.3 | ⚠️ 未明确 | ❌ 未列入 |
| Gemma 4 | ✅ v0.19.0(Day-0) | ✅ v0.5.12(MTP) | ✅ 1.3.0rc21 | 待确认 | ✅ 0.30.3 | ❌ 维护模式 | 待确认 |
| Phi-3 / 3.5 | ✅ v0.6.0(Phi3ForCausalLM) | ✅ v0.4.x | ✅ 0.10.0–0.14.0 | ✅ 早期 | ✅ 0.1.x–0.3.x | ✅ v2.0.2 | ✅ v0.4.x |
| Phi-4 (14B) | ✅ v0.8.5+/v0.9.0 | ✅ v0.5.3+ | ✅ 0.19.0 | ✅ b4000+ | ✅ 0.5.9 | ✅ v3.0+ | ⚠️ v0.6+(仅 mini) |
| gpt-oss (20B/120B) | ✅ v0.11.0 | ✅ v0.5.9+ | ✅ 1.1.0 | ✅ b6100(MXFP4) | ✅ 0.11.0 | ❌ | ✅ v0.10.0 |
| Yi / Baichuan2 | ✅ v0.6.0(Llama 兼容) | ✅ v0.4.x | ✅ via Llama builder | ✅ GGUF | ⚠️ 手动 Modelfile | ✅ 早期 | ✅ v0.6.0+ |
| MiniCPM | ✅ v0.6.0(原生) | ✅ v0.4.x | ✅ via builder | ✅ GGUF | ⚠️ 手动 Modelfile | ✅ 早期 | ✅ v0.6.0+ |


1. **Llama 3.2 Vision (mllama) 在 llama.cpp**：数据集未明确列出 mllama build 号；因 Ollama 0.4.2 即通过 llama.cpp 支持该类型，推断早期 build 已支持，但 v0.30.0 架构切换后继承 Ollama 的损坏问题。
2. **Llama 3.2 Vision 在 Ollama v0.30.x+**：v0.30.0 切换至直接 llama.cpp 引擎后，legacy "mllama" 类型**暂时不可靠**，正在通过 PR 重新添加。**避免在 v0.30.x+ 上运行此模型**。
3. **Llama 3.2 Vision 在 LMDeploy**：TurboMind 引擎的 Llama 支持止于 3.2 (1B/3B 文本)；v0.11.0 起明确不再提供 mllama 支持。
4. **Llama 4 在 vLLM 量化**：v0.8.3 提供 Day-0 基础支持（V1 引擎 only）；FP16/W8A8-FP8 量化需 v0.12.0；MXFP4 W4A16 需 v0.14.0（Blackwell SM100）；NVFP4 需 v0.12.0（Blackwell only）。计算能力 <8.0 会回退 V0 引擎导致 Llama 4 支持损坏。取较高版本 v0.12.0 为量化推荐最低版本。
5. **Llama 4 在 LMDeploy**：仅 PyTorch 引擎支持（分类为 MLLM），FP16/BF16/KV INT8/INT4 已验证；W8A8/W4A16 未验证。TurboMind 引擎不支持 Llama 4（止于 Llama 3.2）。
6. **DeepSeek-V3.2 DSA 在 llama.cpp**：DSA（DeepSeek Sparse Attention）支持状态未在数据集中明确确认，需核对 llama.cpp release notes。

---

## 五、关键适配问题与注意事项

### 5.1 MoE（混合专家）支持

MoE 是 2024—2026 年最重要的架构趋势。**Llama 4**（16/128 专家）、**Qwen3 MoE**（128 专家 Top-8）、**DeepSeek V3/R1**（256 专家）、**GLM-4.5/4.6**（128 专家）、**Mistral Large 3**（675B/41B 激活）、**Mixtral** 均采用 MoE。

- **vLLM**：v0.6.0 起原生 MoE（MixtralForCausalLM）；v0.8.0 引入 EP/DP；v0.8.5 Qwen3 MoE；FP8 CUTLASS grouped GEMM。
- **SGLang**：DeepEP all-to-all 通信后端（Normal + Low-Latency 模式），EPLB 专家负载均衡，Elastic EP（部分 GPU 故障容错），DeepGEMM FP8 专家 GEMM。**MoE 优化最为成熟**。
- **TRT-LLM**：v0.12.0 FP8 OOTB MoE + TP+EP；v1.0.0 DeepEP FP4 all2all；v1.3.0rc 系列 MEGAMOE_CUTEDSL、MARLIN MoE。
- **llama.cpp**：`--n-cpu-moe` / `-ot '.ffn_.*_exps.=CPU'` 可将专家层卸载至 CPU，使大 MoE 在有限 GPU 上运行（如 Scout 单卡 24GB）。
- **LMDeploy**：v0.6.3 column-major MoE kernel；v0.8.0 DeepEP/all2all EP；v0.9.0 FP8 MoE TurboMind；v0.11.0 MoE BF16 EP。

### 5.2 MLA（多头潜在注意力）

**DeepSeek V2 首创** MLA，将 KV cache 压缩为潜在向量（~93% 减少），后续 V3/R1/V3.1/V3.2/V4 及 GLM-4.5/4.6、Mistral Large 3、Kimi K 系列均采用。

- **vLLM**：v0.7.0 原生 MLA 快速路径；v0.8.0 FlashMLA + MLA+chunked prefill；高 TP 会因 head padding 降低 DSA 性能。
- **SGLang**：**DeepSeek 官方推荐**，MLA 优化最显式——data-parallel attention、FP8 W8A8、FP8 KV cache、FlashMLA/FA3/FlashInfer/TRTLLM-MLA 多后端。
- **TRT-LLM**：v0.19.0 FP8 MLA on Hopper/Blackwell + FlashMLA SM90；v1.1.0 MLA 复用+host offloading。
- **llama.cpp**：PR #12801（2025-04-15）完整实现 MLA graph。
- **TGI**：**不支持** DeepSeek MLA 架构，DeepSeek 官方未将其列入推荐框架。

### 5.3 FP8 / INT4 / NVFP4 / MXFP4 量化与精度

| 量化格式 | 说明 | 框架支持要点 |
|----------|------|-------------|
| **FP8 (E4M3)** | Hopper+ 原生；DeepSeek V3/R1 官方 FP8 权重 | vLLM v0.7.1; SGLang v0.4.2 原生; TRT-LLM v0.19.0; LMDeploy v0.7.0(PyTorch) |
| **INT4 AWQ/GPTQ** | Ampere+；weight-only | TRT-LLM v0.10.0+; LMDeploy TurboMind W4A16; llama.cpp K-quant/I-quant |
| **NVFP4** | Blackwell 专用；4-bit 浮点 | TRT-LLM v0.17.0; vLLM v0.12.0(Blackwell); SGLang v0.5.12 |
| **MXFP4** | OCP Microscaling FP4；gpt-oss 原生 | llama.cpp b6100(GGML_TYPE_MXFP4); Ollama 0.11.0; TRT-LLM v1.0.0+ |

**精度注意事项**：
- llama.cpp build **b5237+** 的 flash-attention 变更可能导致 **1-3 bit 精度损失**，输出质量下降时使用 `--no-flash-attn`。
- DeepSeek-V3 FP8 最小 VRAM 805GB（8×H100/H200/B200）；NVFP4 变体 403GB（4×B200）。
- vLLM 对 Llama 4 的 MXFP4 **W4A4（激活量化）尚不支持**，仅 W4A16 weights-only。

### 5.4 长上下文（128K / 1M / 10M）

| 上下文规模 | 代表模型 | 框架要点 |
|-----------|----------|----------|
| 128K | Llama 3.x, Qwen2.5/3, DeepSeek V3, GLM-4.5, Gemma 3 | 各框架主流支持；chunked prefill / prefix caching |
| 160K–164K | DeepSeek-R1-0528, DeepSeek-V3(Ascend) | 需 MLA KV cache 管理 |
| 256K–1M | Qwen3-VL, Qwen3-Next, Llama 4 Maverick | SGLang chunked pipeline parallelism 近线性扩展; vLLM DP+EP |
| 10M | Llama 4 Scout | SGLang 8×H100 跑 1M; 8×H200 可跑 2.5M; Ollama Modelfile num_ctx |

- **KV cache 管理**：vLLM v0.20+ 2-bit KV cache（4× 容量）；SGLang HiSparse CPU KV offload + SSD offload via Mooncake；TRT-LLM KVCacheManagerV2。
- **Ollama**：num_ctx 通过 Modelfile PARAMETER 或 API options 设置；flash attention v0.11.6/v0.13.4 默认开启。

### 5.5 多模态（VL）

- **vLLM**：v0.6.0+ Mllama；v0.8.0 Gemma 3 多模态；v0.19.0 Gemma 4 原生音频+视觉。
- **SGLang**：v0.3.0 LLaVA-OneVision；v0.5.x EPD 弹性编码器扩展；Qwen3-VL/Step3-VL/Ernie4.5-VL/MiniCPM-V/Pixtral/Voxtral。
- **Ollama**：v0.4.2 首个多模态(LLaVA)；**v0.7.0 重写多模态引擎**（Qwen3/Qwen2 arch）；**⚠️ v0.30.0 后 Llama 3.2 Vision (mllama) 暂时损坏**。
- **llama.cpp**：b5423 Llama 4 vision；Gemma 3/3n via `llama-mtmd-cli`；Qwen3-VL 原生 GGUF（LLM.gguf + mmproj.gguf）。
- **TGI**：v2.3.1 Mllama；文本聚焦，Pixtral 不支持。

### 5.6 思考模式（Thinking Mode）

Qwen3 的**混合思考模式**（`enable_thinking` / `/think` / `/no_think`）与 DeepSeek-R1/GLM 的推理模式是 2025 年核心特性。

- **vLLM**：`--enable-reasoning --reasoning-parser deepseek_r1`（Qwen3）；`--reasoning-parser glm45`（GLM-4.5）。
- **SGLang**：`--reasoning-parser qwen3` / `glm45`（glm47 for 4.7）；Thinking Budget via `--enable-custom-logit-processor`（GLM）。
- **Ollama**：v0.9.0 **原生 Thinking Mode**（`/set think`、`/set nothink`、API `think:true`），返回结构化 thinking 字段。支持 DeepSeek-R1、Qwen3、GLM。
- **LMDeploy**：v0.9.0 Qwen3 `/think` & `/no_think`；v0.9.1 工具调用+推理内容解析。
- **TGI**：无专门 thinking mode 支持。

### 5.7 版本升级踩坑建议

1. **先升框架再换模型**：新模型发布后框架通常在 1-7 天内 Day-0 支持，但量化/稳定性可能需数周迭代。生产环境建议等待**框架的第二个 patch 版本**。
2. **vLLM V0→V1 引擎切换**：v0.8.0 起 V1 默认；部分旧特性（如 N>1 sampling）在 V0 上不稳定，V1 上已修复。Llama 4 在 V0 引擎上**支持损坏**（`use_irope` 不支持）。
3. **TRT-LLM v1.2 后端迁移**：v1.2 移除 C++ TRT 后端，`LLM(backend='tensorrt')` 直接报错。旧引擎构建流程不再适用。
4. **Ollama v0.30.0 架构切换**：直接集成 llama.cpp 引擎，但 **Llama 3.2 Vision 暂时损坏**；`nomic-embed-text` 小写输入行为变更。如需 Llama 3.2 Vision，**留在 v0.23.x 或更早**。
5. **TGI 已归档**：2026-03-21 仓库归档，无新功能。Llama 4 / Gemma 4 / DeepSeek V4 不主动支持。**新项目不要选 TGI**。
6. **LMDeploy 双引擎差异**：TurboMind（C++ 高性能）与 PyTorch 引擎支持模型不同。DeepSeek V3/R1 **仅在 PyTorch 引擎**运行（无 TurboMind W4A16/KV-int8）；Llama 4 仅 PyTorch 引擎；GLM-4.7-Flash 仅 FP16/BF16（无量化）。
7. **llama.cpp build 号映射**：使用递增 build 号而非日期，约 8-12 builds/天。注意 GitHub 页脚年份误标问题，以 PR 合并时间戳为准。

---

## 六、选型建议

| 场景 | 推荐框架 | 理由 |
|------|----------|------|
| **生产低延迟 / 高吞吐** | TRT-LLM / SGLang | TRT-LLM 在 NVIDIA GPU 上延迟最优（in-flight batching, FP8/NVFP4 kernel）；SGLang MoE/MLA 优化最成熟，DeepSeek 官方推荐 |
| **易用性 / 快速原型** | Ollama / vLLM | Ollama 一行命令拉取运行，CLI/TUI 友好；vLLM OpenAI 兼容 API，生态最广 |
| **边缘 / 消费级 GPU / CPU** | llama.cpp | GGUF 量化（Q2_K–Q8_0），CPU+GPU 混合，MoE 层卸载，最低硬件门槛 |
| **HuggingFace 生态 / 已有 TGI 部署** | vLLM / SGLang（迁移） | TGI 已归档；HuggingFace 官方建议迁移至 vLLM/SGLang |
| **国产模型（DeepSeek/Qwen/GLM）** | SGLang / LMDeploy | SGLang DeepSeek 官方推荐，MLA/MoE 优化最显式；LMDeploy 支持国产 Ascend/Cambricon 加速器，TurboMind W4A16 量化成熟 |
| **多模态 / VL** | vLLM / SGLang | 两者 VL 覆盖最广（Qwen3-VL, Gemma 3/4, Llama 4, Pixtral）；Ollama 次之但 v0.30.x 有 mllama 损坏问题 |
| **Apple Silicon** | Ollama / llama.cpp / SGLang(MLX) | Ollama MLX 引擎；llama.cpp Metal kernel；SGLang v0.5.10 原生 MLX 后端 |

**一句话总结**：截至 2026-08，**SGLang v0.5.17 + vLLM v0.27.1** 是国产 MoE/MLA 模型（DeepSeek/Qwen3/GLM）的两大首选生产框架；**TRT-LLM v1.3.0rc** 是 NVIDIA 硬件上延迟最优方案；**Ollama v0.32.x / llama.cpp b10405** 是本地与边缘部署首选；**TGI 已不建议新项目采用**。

---

## 附录：数据来源

> 截至 2026 年 8 月，版本快速迭代，使用前以官方 release 为准。

### 模型族来源

- Meta Llama: https://github.com/meta-llama/llama-models/blob/main/models/llama4/MODEL_CARD.md · https://ai.meta.com/blog/llama-4-multimodal-intelligence · https://ollama.com/library/llama4
- Alibaba Qwen: https://qwen.ai/blog · https://qwen.readthedocs.io/en/latest/deployment/vllm.html · https://github.com/vllm-project/vllm/releases/tag/v0.8.5 · https://huggingface.co/Qwen/Qwen3-VL-4B-Instruct-GGUF
- DeepSeek: https://huggingface.co/deepseek-ai · https://docs.sglang.ai/basic_usage/deepseek_v32.html · https://arxiv.org/html/2512.02556v1 · https://api-docs.deepseek.com/news/news260424
- Mistral AI: https://mistral.ai/news/magistral · https://mistral.ai/news/devstral-2-vibe-cli · https://huggingface.co/mistralai/Pixtral-12B-2409 · https://huggingface.co/blog/mixtral
- GLM (Zhipu/Z.ai): https://z.ai/blog/glm-4.6 · https://huggingface.co/docs/transformers/model_doc/glm4_moe · https://github.com/THUDM/GLM-4/blob/main/README_zh.md · https://blog.vllm.com.cn/2025/08/19/glm45-vllm.html
- Gemma/Phi/其他: https://blog.google/technology/developers/gemma-3/ · https://huggingface.co/blog/gemma2 · https://huggingface.co/blog/gemma3n · https://arxiv.org/abs/2412.08905 · https://github.com/OpenBMB/MiniCPM

### 框架来源

- vLLM: https://github.com/vllm-project/vllm/releases · https://docs.vllm.ai/en/latest/models/supported_models.html · https://pypi.org/project/vllm/
- SGLang: https://github.com/sgl-project/sglang/releases · https://docs.sglang.io · https://sglang.org/zh/supported_models/generative_models · https://lmsys.org/blog/2025-07-31-glm4-5
- TensorRT-LLM: https://nvidia.github.io/TensorRT-LLM/release-notes.html · https://github.com/NVIDIA/TensorRT-LLM/releases · https://nvidia.github.io/TensorRT-LLM/reference/support-matrix.html
- llama.cpp: https://github.com/ggml-org/llama.cpp/releases · https://github.com/ggml-org/llama.cpp/pull/12828 · https://github.com/ggml-org/llama.cpp/pull/12801 · https://github.com/ggml-org/llama.cpp/pull/14939 · https://github.com/ggml-org/llama.cpp/discussions/15095
- Ollama: https://ollama.com/library · https://github.com/ollama/ollama/releases · https://ollama.com/library/llama4 · https://ollama.com/library/deepseek-r1
- TGI: https://huggingface.co/docs/text-generation-inference/en/supported_models · https://github.com/huggingface/text-generation-inference/releases
- LMDeploy: https://github.com/InternLM/lmdeploy/releases · https://lmdeploy.readthedocs.io/en/latest/supported_models/supported_models.html · https://pypi.org/project/lmdeploy/

### 交叉验证与第三方来源

- https://docs.redhat.com/en/documentation/red_hat_ai_inference_server · https://newreleases.io · https://chocolatey.org/packages/Ollama · https://www.53ai.com · https://cloud.tencent.com.cn/developer · https://blog.csdn.net · https://huggingface.co/unsloth · https://www.unsloth.ai/docs · https://deepwiki.com · https://www.pbone.net
