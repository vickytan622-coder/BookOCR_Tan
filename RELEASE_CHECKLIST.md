# BookOCR 发布前检查

本项目已在 Apple Silicon Mac 上用一份 148 页中文扫描书完成端到端实测。对外发布前，请完成以下不可自动替用户决定的项目。

## 需要仓库所有者决定

- [ ] 选择开源许可证（MIT、Apache-2.0、GPL-3.0 等）；
- [ ] 创建或指定 GitHub 仓库，并确认公开范围；
- [ ] 决定是否附带示例 PDF（默认不附带受版权保护的书籍与 OCR 输出）。

## 发布内容

- [ ] 保留：`src/`、`tests/`、`README.md`、`pyproject.toml`、`CLAUDE.md` 与通用安装/启动脚本；
- [ ] 不提交：`.venv/`、`.cache/`、`output/`、`tmp/`、BookOCR 的本机 Keychain 配置；
- [ ] 不提交：带特定书名、任务 ID 的本机验收脚本；
- [ ] 在一台干净的 Apple Silicon Mac 上运行 `安装 BookOCR.command` 与 1 页试跑；
- [ ] 在干净机器上验证：服务已在跑时再次双击 `启动 BookOCR.command` 会直接打开浏览器；
- [ ] 验证 LLM 校对遇到网络波动时能自动进入 `waiting_network` 倒计时并重试；
- [ ] 验证暂停/继续、服务停止按钮正常工作。

## 已验证的闭环

- PDF 页面级断点 OCR，按 50 页调度，不物理切割原 PDF；
- MLX 本地服务可选加速；失败页可用 CPU 回退；
- 每页 Markdown、原始整书 Markdown、质量检查报告；
- 任何 OpenAI Chat Completions 兼容 LLM 的可恢复校对；
- Base URL/模型名持久化，API Key 保存到 macOS 钥匙串；
- 本机仅监听 `127.0.0.1`，不上传 PDF。
