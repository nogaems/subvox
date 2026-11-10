<#
Обёртка для Windows: корзина смоук-видео.
Сам скрипт — contrib/fetch_smoke.py, платформенно нейтральный.
Linux/macOS: contrib/fetch-smoke.sh
#>
$ErrorActionPreference = "Stop"
$here = Split-Path -Parent $MyInvocation.MyCommand.Path
uv run --project (Join-Path $here "..") python (Join-Path $here "fetch_smoke.py") @args
