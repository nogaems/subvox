<#
Установка subvox под Windows: ffmpeg, uv с Python 3.12, сам subvox.

СТАТУС: черновик-попытка, НА WINDOWS НЕ ЗАПУСКАЛСЯ — написан по документации
winget/uv. Что сломается — поправить при первой проверке на живой машине.

Запуск (из каталога с файлом или скачав его):
    powershell -ExecutionPolicy Bypass -File install.ps1
    powershell -ExecutionPolicy Bypass -File install.ps1 -FromSource C:\path\to\subvox

Всё per-user; прав администратора скрипт сам не требует — winget сам
запросит UAC там, где установщик захочет машинную установку.

GPU опционален: для CUDA нужен CUDA Toolkit 12 (https://developer.nvidia.com/
cuda-downloads); без него subvox честно работает на CPU.
#>

param(
    [string]$FromSource = "",
    [string]$Repo = "nogaems/subvox"
)

$ErrorActionPreference = "Stop"

function Say($msg)  { Write-Host "==> $msg" -ForegroundColor Cyan }
function Fail($msg)  { Write-Host "ошибка: $msg" -ForegroundColor Red; exit 1 }

# --- 1. ffmpeg -----------------------------------------------------------

if (Get-Command ffmpeg -ErrorAction SilentlyContinue) {
    Say "ffmpeg есть"
} elseif (Get-Command winget -ErrorAction SilentlyContinue) {
    Say "ffmpeg не найден, ставлю через winget"
    winget install --id=Gyan.FFmpeg -e --accept-source-agreements --accept-package-agreements
    if ($LASTEXITCODE -ne 0) {
        Fail "winget не смог поставить ffmpeg. Поставьте вручную: https://www.gyan.dev/ffmpeg/builds/ и убедитесь, что он в PATH"
    }
    # winget обновил PATH в реестре, но текущая сессия этого не видит.
    # Для установки subvox ffmpeg не нужен — он нужен в рантайме.
    Say "ffmpeg поставлен; если subvox его не увидит — переоткройте терминал"
} else {
    Fail "ffmpeg не найден, а winget недоступен (Windows 10 1709+ должен иметь). Поставьте ffmpeg вручную: https://www.gyan.dev/ffmpeg/builds/"
}

# --- 2. uv ----------------------------------------------------------------

# winget мог поставить uv только что — в этой сессии его ещё нет в PATH,
# поэтому ищем и по известным местам.
$uv = $null
if (Get-Command uv -ErrorAction SilentlyContinue) {
    $uv = (Get-Command uv).Source
} else {
    foreach ($p in @(
        "$env:USERPROFILE\.local\bin\uv.exe",
        "$env:LOCALAPPDATA\Programs\uv\uv.exe",
        "$env:LOCALAPPDATA\Microsoft\WinGet\Links\uv.exe"
    )) {
        if (Test-Path $p) { $uv = $p; break }
    }
}

if (-not $uv) {
    if (Get-Command winget -ErrorAction SilentlyContinue) {
        Say "uv не найден, ставлю через winget"
        winget install --id=astral-sh.uv -e --accept-source-agreements --accept-package-agreements
        if ($LASTEXITCODE -ne 0) {
            Fail "winget не смог поставить uv. Альтернатива: powershell -c ""irm https://astral.sh/uv/install.ps1 | iex"""
        }
    } else {
        Say "winget нет, ставлю uv официальным установщиком"
        & powershell -NoProfile -ExecutionPolicy Bypass -Command "irm https://astral.sh/uv/install.ps1 | iex"
    }
    foreach ($p in @(
        "$env:USERPROFILE\.local\bin\uv.exe",
        "$env:LOCALAPPDATA\Programs\uv\uv.exe",
        "$env:LOCALAPPDATA\Microsoft\WinGet\Links\uv.exe"
    )) {
        if (Test-Path $p) { $uv = $p; break }
    }
    if (-not $uv) {
        Fail "uv установлен, но не найден ни в PATH, ни в известных местах. Переоткройте терминал и запустите скрипт снова"
    }
}

Say "uv: $uv"

# --- 3. subvox ------------------------------------------------------------

if ($FromSource -ne "") {
    if (-not (Test-Path $FromSource)) { Fail "-FromSource: $FromSource не существует" }
    Say "uv tool install $FromSource"
    & $uv tool install --python 3.12 $FromSource
} else {
    $url = "https://github.com/$Repo/releases/latest/download/subvox.whl"
    $wheel = Join-Path $env:TEMP "subvox.whl"
    Say "скачиваю $url"
    try {
        Invoke-WebRequest -Uri $url -OutFile $wheel
    } catch {
        Fail "не удалось скачать wheel ($url). Релизы ещё не опубликованы? Укажите -Repo nogaems/subvox или -FromSource <каталог>"
    }
    Say "uv tool install subvox"
    & $uv tool install --python 3.12 $wheel
}
if ($LASTEXITCODE -ne 0) { Fail "uv tool install не удался — текст ошибки выше" }

Say "готово. Модели (~4.5 ГБ) скачаются сами при первом запуске, в $env:LOCALAPPDATA\subvox"
Say "если команда subvox не находится — переоткройте терминал"
