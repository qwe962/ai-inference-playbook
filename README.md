# AI Inference Playbook

一个面向实际工程的 AI 推理部署与优化知识库。

这个项目不以堆放零散笔记为目标，而是沉淀可复用、可验证、可维护的部署方法、优化实验、故障案例与模型实践。

## 内容范围

- SGLang 和 vLLM 服务部署
- Docker、CUDA 与 NVIDIA GPU 环境
- NVIDIA H20 多卡模型部署
- Tensor Parallel、显存估算与 KV Cache
- API Key、健康检查与 OpenAI 兼容接口
- MiniMax、DeepSeek、Qwen 部署案例
- CUDA、OOM、端口与容器问题排查
- 推理优化方法、实验与效果分析

## 知识库结构

| 路径 | 内容 |
| --- | --- |
| [`docs/inference-optimization/`](docs/inference-optimization/) | 推理优化方法、测量方式与实验结论，项目的核心方向 |
| [`docs/serving/`](docs/serving/) | SGLang、vLLM 服务启动与配置 |
| [`docs/gpu-and-containers/`](docs/gpu-and-containers/) | Docker、CUDA、NVIDIA GPU 环境 |
| [`docs/multi-gpu/`](docs/multi-gpu/) | H20 多卡部署与并行策略 |
| [`docs/memory-and-kv-cache/`](docs/memory-and-kv-cache/) | 显存估算、KV Cache 与容量规划 |
| [`docs/api-and-operations/`](docs/api-and-operations/) | 鉴权、健康检查、兼容接口与服务运维 |
| [`docs/model-recipes/`](docs/model-recipes/) | MiniMax、DeepSeek、Qwen 等模型部署案例 |
| [`docs/model-framework-compatibility/`](docs/model-framework-compatibility/) | 主流模型、推理框架版本与功能适配关系 |
| [`docs/troubleshooting/`](docs/troubleshooting/) | CUDA、OOM、端口、容器等问题排查 |
| [`templates/`](templates/) | 新增部署、优化与故障记录时使用的模板 |
| [`assets/`](assets/) | 文档引用的图片、图表和其他静态资源 |

更完整的导航见 [`docs/README.md`](docs/README.md)。

## 内容标准

每篇实践文档应尽量做到：

1. 写明硬件、驱动、CUDA、框架、镜像与模型版本。
2. 区分计划、静态检查和已经完成的运行验证。
3. 提供可复现的命令、关键配置与验收方式。
4. 对优化项保留基线、变量、指标和结论，避免只记录参数。
5. 删除 API Key、访问令牌、内网地址等敏感信息后再提交。

## 使用方式

新增资料前，先选择最接近的主题目录，并优先从 [`templates/`](templates/) 复制对应模板。目录找不到合适位置时，再新增分类，避免同一主题散落在多个地方。

## 参与贡献

提交内容前请阅读 [`CONTRIBUTING.md`](CONTRIBUTING.md)。

