#!/bin/zsh
set -e

cd "$(dirname "$0")"

if [[ ! -x .venv/bin/python ]]; then
  echo "尚未完成安装。请先运行‘安装 BookOCR.command’。"
  read -k 1 "?按任意键退出"
  exit 1
fi

# 如果 8765 已被 BookOCR 占用，直接打开浏览器即可，不要强制用户先关窗口。
if lsof -nP -iTCP:8765 -sTCP:LISTEN >/dev/null 2>&1; then
  BOOKOCR_PID=$(lsof -nP -iTCP:8765 -sTCP:LISTEN | awk 'NR>1 {print $2; exit}')
  if ps -p "$BOOKOCR_PID" -o args= | grep -q "bookocr.cli"; then
    echo "BookOCR 服务已在运行，正在打开浏览器…"
    open "http://127.0.0.1:8765"
    exit 0
  fi
  echo "端口 8765 被其他程序占用，请先关闭占用端口的程序。"
  read -k 1 "?按任意键退出"
  exit 1
fi

# 始终从当前文件夹的 src 启动，避免启动器本身没改而误用旧安装包。
# 不在 Python 内调用 webbrowser.open，而是等服务起来后用 macOS open
# 命令显式打开默认浏览器，比在 Terminal 里依赖 Python 的浏览器模块更可靠。
env PYTHONPATH="$PWD/src" .venv/bin/python -m bookocr.cli serve --no-browser &
SERVER_PID=$!

echo "正在启动 BookOCR 本地服务…"
for i in {1..30}; do
  if lsof -nP -iTCP:8765 -sTCP:LISTEN >/dev/null 2>&1; then
    break
  fi
  sleep 0.2
done

if ! lsof -nP -iTCP:8765 -sTCP:LISTEN >/dev/null 2>&1; then
  echo "服务未能在 6 秒内启动，请查看上方错误信息。"
  kill $SERVER_PID 2>/dev/null || true
  wait $SERVER_PID 2>/dev/null || true
  read -k 1 "?按任意键退出"
  exit 1
fi

open "http://127.0.0.1:8765"
echo "已打开浏览器：http://127.0.0.1:8765"
echo "请保持本窗口打开；关闭本窗口会停止服务。"
echo "也可在网页右上角点击「停止 BookOCR 服务」来关闭。"

wait $SERVER_PID
