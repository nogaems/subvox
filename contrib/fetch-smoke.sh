#!/bin/sh
# Скачивает корзину смоук-видео в paths.smoke_dir() — стандартное
# системное per-user место.
#
# Ссылки живут отдельно, в манифесте contrib/smoke-videos.md: там
# единственный источник правды о том, что качать; этот скрипт лишь
# исполняет его. Смена ссылки — правка манифеста, скрипт не трогается.
#
# Идемпотентно: существующие слоты пропускаются, ничего не
# перезаписывается. Повторный запуск докачивает только недостающее.
#
# Linux/macOS. На Windows — та же yt-dlp-команда вручную,
# шаблон в contrib/smoke-videos.md.
#
# Потолок 480p: пайплайн ест только аудиодорожку, видео нужно лишь
# для проверки синхронности глазами.
set -eu

here=$(cd "$(dirname "$0")" && pwd)
project=$(cd "$here/.." && pwd)
manifest="$here/smoke-videos.md"

[ -f "$manifest" ] || { echo "нет манифеста: $manifest" >&2; exit 1; }

dir=$(uv run --project "$project" python -c 'from subvox import paths; print(paths.smoke_dir())')
mkdir -p "$dir"
echo "корзина: $dir"
echo "манифест: $manifest"

# Строки манифеста вида `| слот | https://... | ...` — вытаскиваем
# первые две ячейки. Ссылка обязана начинаться с https://youtu.be/.
entries=$(grep -E '^\| *`[a-z0-9-]+` *\| *https://youtu\.be/' "$manifest" \
    | awk -F'|' '{ gsub(/`| /, "", $2); gsub(/ /, "", $3); print $2 "|" $3 }')

count=$(printf '%s\n' "$entries" | grep -c .)
echo "слотов в манифесте: $count"

failed=0
missing=0
for entry in $entries; do
    slot=${entry%%|*}
    url=${entry#*|}
    if ls "$dir/$slot".* >/dev/null 2>&1; then
        echo "  есть:   $slot"
        continue
    fi
    missing=$((missing + 1))
    printf '  качаю:  %s\n' "$slot"
    if ! uvx yt-dlp -f "bv*[height<=480]+ba/b[height<=480]/b" \
        --merge-output-format mkv --no-overwrites \
        -o "$dir/$slot.%(ext)s" "$url" >/dev/null; then
        echo "  ОШИБКА: $slot ($url)"
        failed=$((failed + 1))
    fi
done

echo
if [ "$failed" -gt 0 ]; then
    echo "не скачалось: $failed из $missing недостающих — см. ошибки выше"
    exit 1
fi
echo "готово: корзина целиком:"
ls -lh "$dir"
