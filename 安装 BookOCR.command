#!/bin/zsh
set -e

cd "$(dirname "$0")"

if [[ "$(uname -m)" != "arm64" ]]; then
  echo "当前试用版仅支持 Apple Silicon（M 系列）Mac。"
  read -k 1 "?按任意键退出"
  exit 1
fi

python_bin=""
for candidate in python3.12 python3.11 python3; do
  if command -v "$candidate" >/dev/null 2>&1 && "$candidate" -c 'import sys; raise SystemExit(0 if (3, 11) <= sys.version_info < (3, 13) else 1)'; then
    python_bin="$(command -v "$candidate")"
    break
  fi
done

if [[ -z "$python_bin" ]]; then
  echo "需要 Python 3.11 或 3.12。当前试用版不支持 Python 3.13+。"
  read -k 1 "?按任意键退出"
  exit 1
fi

echo "使用 Python：$($python_bin --version)"

if [[ ! -d .venv ]]; then
  "$python_bin" -m venv .venv
fi

.venv/bin/python -m pip install --upgrade pip
.venv/bin/python -m pip install '.[ocr,web]'
.venv/bin/python -m pip install mlx-vlm
.venv/bin/bookocr doctor
echo "安装完成。首次 OCR 会下载模型文件，请保持网络连接。"
read -k 1 "?按任意键退出"
