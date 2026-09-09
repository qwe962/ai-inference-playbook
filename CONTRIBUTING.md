# 贡献指南

## 基本原则

- 一篇文档解决一个明确问题。
- 结论必须能追溯到环境、命令、日志或指标。
- 版本相关结论必须写明适用版本和验证日期。
- 不把“预计可行”“命令可运行”和“端到端验证通过”混为一谈。
- 不提交模型权重、数据集、构建产物、日志归档或任何凭据。

## 新增内容

1. 在 `docs/` 中选择对应主题。
2. 从 `templates/` 复制最接近的模板。
3. 使用小写英文文件名和连字符，例如 `h20-tp4-deployment.md`。
4. 将新文档加入所属目录的 `README.md` 索引。
5. 检查命令、链接、图片路径和敏感信息。

## 文档状态

文档开头建议标明以下状态之一：

- `Draft`：正在整理，尚未完成验证。
- `Verified`：已在文档记录的环境中完成验证。
- `Outdated`：关键依赖或结论已经过时，保留作历史参考。

## 提交建议

提交信息保持简洁，并说明变更类型，例如：

```text
docs: add H20 TP4 deployment record
docs: document CUDA OOM diagnosis
chore: update knowledge base navigation
```
