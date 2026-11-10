#!/bin/sh
# Установка subvox на Linux/macOS: ffmpeg, uv с Python 3.12, сам subvox.
#
# СТАТУС: черновик-попытка. Путь --from-source проверен на Linux;
# путь через GitHub Releases (SUBVOX_REPO) не проверялся — репозиторий
# ещё не опубликован. Что сломается — правится руками по командам ниже,
# они все показаны по ходу.
#
# Запуск:
#   ./install.sh                            # из GitHub Releases (SUBVOX_REPO)
#   SUBVOX_REPO=owner/subvox ./install.sh   # то же, с указанием репозитория
#   ./install.sh --from-source ~/subvox     # из локальной копии (разработка)
#   ./install.sh --gpu                      # + CUDA-библиотеки в окружении
#
# Всё per-user: системные пакеты — только ffmpeg, и только если его нет.
set -eu

repo="${SUBVOX_REPO:-owner/subvox}"
wheel_url="https://github.com/${repo}/releases/latest/download/subvox.whl"
from_source=""
gpu=""

while [ "$#" -gt 0 ]; do
    case "$1" in
        --from-source)
            [ "$#" -ge 2 ] || { echo "--from-source требует путь" >&2; exit 1; }
            from_source="$2"; shift 2 ;;
        --gpu) gpu="1"; shift ;;
        -h|--help)
            sed -n '2,16p' "$0"; exit 0 ;;
        *)
            echo "неизвестный аргумент: $1" >&2; exit 1 ;;
    esac
done

say() { printf '\033[1;34m==>\033[0m %s\n' "$*"; }
fail() { printf '\033[1;31mошибка:\033[0m %s\n' "$*" >&2; exit 1; }

# --- 1. ffmpeg -----------------------------------------------------------

if command -v ffmpeg >/dev/null 2>&1; then
    say "ffmpeg есть: $(ffmpeg -version 2>/dev/null | head -1)"
else
    say "ffmpeg не найден, ставлю системным пакетным менеджером"
    if command -v brew >/dev/null 2>&1; then
        brew install ffmpeg
    elif command -v apt-get >/dev/null 2>&1; then
        sudo apt-get install -y ffmpeg
    elif command -v dnf >/dev/null 2>&1; then
        sudo dnf install -y ffmpeg
    elif command -v pacman >/dev/null 2>&1; then
        sudo pacman -S --noconfirm ffmpeg
    elif command -v zypper >/dev/null 2>&1; then
        sudo zypper --non-interactive install ffmpeg
    else
        fail "не нашёл пакетный менеджер. Поставьте ffmpeg сами и запустите
скрипт снова: apt/dnf/pacman/zypper/brew — или статическая сборка
https://johnvansickle.com/ffmpeg/"
    fi
fi

# --- 2. uv с Python 3.12 --------------------------------------------------

if ! command -v uv >/dev/null 2>&1; then
    say "uv не найден, ставлю официальный установщик (~/.local/bin)"
    curl -LsSf https://astral.sh/uv/install.sh | sh
    # Установщик кладёт uv в ~/.local/bin — он может ещё не быть в PATH
    # этого терминала, добавляю только для этого запуска.
    PATH="$HOME/.local/bin:$PATH"
    export PATH
    command -v uv >/dev/null 2>&1 \
        || fail "uv установлен, но не в PATH — переоткройте терминал и запусти скрипт снова"
fi

# --- 3. subvox ------------------------------------------------------------

uv_install() {
    # Python 3.12 uv скачает сам, если его нет; версия зафиксирована —
    # у ctranslate2 нет колёс под более новые.
    say "uv tool install $1"
    uv tool install --python 3.12 "$1"
}

if [ -n "$from_source" ]; then
    [ -d "$from_source" ] || fail "--from-source: $from_source — не каталог"
    spec="$from_source"
    [ -n "$gpu" ] && spec="$spec[gpu]"
    uv_install "$spec"
else
    say "скачиваю $wheel_url"
    tmp="$(mktemp -d)"
    trap 'rm -rf "$tmp"' EXIT
    curl -LsSf -o "$tmp/subvox.whl" "$wheel_url" || fail "не удалось скачать wheel.
Релизы ещё не опубликованы? Укажите репозиторий: SUBVOX_REPO=owner/subvox ./install.sh
Или поставьте из исходника: ./install.sh --from-source <каталог>"
    spec="$tmp/subvox.whl"
    [ -n "$gpu" ] && spec="$spec[gpu]"
    uv_install "$spec"
fi

say "готово: subvox $(uv tool list 2>/dev/null | sed -n 's/^subvox v\([^ ]*\).*/\1/p')"
say "модели (~4.5 ГБ) скачаются сами при первом запуске, в ~/.local/share/subvox"
say "если команда subvox не находится — переоткройте терминал"
[ -n "$gpu" ] || say "GPU опционален; без него subvox честно работает на CPU"
