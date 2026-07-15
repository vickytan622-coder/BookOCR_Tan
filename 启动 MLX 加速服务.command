#!/bin/zsh
set -e

cd "$(dirname "$0")"

if [[ ! -x .venv/bin/python ]]; then
  echo "请先运行‘安装 BookOCR.command’。"
  read -k 1 "?按任意键退出"
  exit 1
fi

# 仅监听本机。复用 PaddleOCR 已下载的本地模型，避免 Hugging Face SSL 下载失败。
.venv/bin/python -m mlx_vlm.server \
  --host 127.0.0.1 \
  --port 8118 \
  --model "$PWD/.cache/paddlex/official_models/PaddleOCR-VL-1.6" \
  --trust-remote-code

status=$?
echo
echo "MLX 服务已退出，退出码：$status"
echo "请复制上方最后 30 行文字发给 Codex。"
read -k 1 "?按任意键关闭窗口"
exit $status
