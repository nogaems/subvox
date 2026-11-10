#!/bin/sh
# Обёртка для Linux/macOS: корзина смоук-видео.
# Сам скрипт — contrib/fetch_smoke.py, платформенно нейтральный.
# Windows: powershell -ExecutionPolicy Bypass -File contrib/fetch-smoke.ps1
set -eu
here=$(cd "$(dirname "$0")" && pwd)
exec uv run --project "$here/.." python "$here/fetch_smoke.py" "$@"
