#!/bin/bash
# Download anime clips from official YouTube channels using yt-dlp.
# All sources are official studio/distributor uploads (verified 2026).
#
# Strategy: pull the N most recent uploads from each channel, prioritizing
# trailers, PVs, and TOHO's non-credit OP/ED collections (best dense-motion data).
#
# Output: /storage/SSD3/yptsai/data/youtube_anime/
#   videos/   (deleted after frame extraction)
#   frames/   (final training data)

VENV=/storage/SSD2/users/yptsai/program/venv
REPO=/storage/SSD2/users/yptsai/program/anime_interp
OUT_VIDEO=/storage/SSD3/yptsai/data/youtube_anime/videos
OUT_FRAMES=/storage/SSD3/yptsai/data/youtube_anime/frames

FPS=12              # extraction fps (anime broadcast = 24fps; 12 = every 2nd frame)
MIN_FRAMES=33       # discard clips shorter than this
MAX_DIM=768         # resize long edge

PER_CHANNEL_LIMIT=30   # max videos per channel (set higher for more data)
MIN_DURATION=15        # skip clips shorter than 15s
MAX_DURATION=1600      # ~26 min — accommodates full anime episodes (AniOne)
                       # while still skipping multi-hour livestreams

source "$VENV/bin/activate"
mkdir -p "$OUT_VIDEO" "$OUT_FRAMES"

# ── Official anime YouTube channels ────────────────────────────────────────
# All verified official channels. Channel IDs prefixed with UU give the
# uploads-only playlist (UC<id> → UU<id>).
CHANNELS=(
    # name|channel-or-playlist-url
    "TOHO_animation|https://www.youtube.com/@tohoanimation/videos"
    "MAPPA|https://www.youtube.com/@MAPPACHANNEL/videos"
    "ufotable|https://www.youtube.com/@ufotable_inc/videos"
    "KyoAni|https://www.youtube.com/channel/UCpGY2vcoKXf7K6tFzsbSv7w/videos"
    "WIT_STUDIO|https://www.youtube.com/channel/UCivtAzCENYI1jb6Clxydvdw/videos"
    "Production_IG|https://www.youtube.com/channel/UCmpf0bVV5fTemOo2WVxjLow/videos"
    "Madhouse|https://www.youtube.com/user/MadhousePlus/videos"
    "Aniplex_JP|https://www.youtube.com/@aniplex/videos"
    "Aniplex_USA|https://www.youtube.com/playlist?list=UUDb0peSmF5rLX7BvuTcJfCw"
    "Crunchyroll_Collection|https://www.youtube.com/@CrunchyrollCollection/videos"
    "KADOKAWAanime|https://www.youtube.com/channel/UCY5fcqgSrQItPAX_Z5Frmwg/videos"
    "PonyCanyon_Anime|https://www.youtube.com/channel/UCb-ekPowbBlQhyt7ZXPiu5Q/videos"
    "Bandai_Namco_Filmworks|https://www.youtube.com/channel/UCQ5URCSs1f5Cz9rh-cDGxNQ/videos"
    "Toei_Animation|https://www.youtube.com/channel/UCQYYekTKCb1y12sas08T6gQ/videos"
    # AniOne — Medialink, full anime episodes (~24 min) accessible from Asia IPs.
    # Higher per-channel cap because episodes are full length (not short PVs).
    "AniOne|https://www.youtube.com/c/AniOneAnime/videos"
)

# ─────────────────────────────────────────────────────────────────────────────

echo "=== Anime YouTube downloader ==="
echo "  Channels: ${#CHANNELS[@]}"
echo "  Per-channel limit: $PER_CHANNEL_LIMIT videos"
echo "  Duration filter: ${MIN_DURATION}s – ${MAX_DURATION}s"
echo "  Output: $OUT_FRAMES"
echo ""

extract_and_cleanup() {
    local video_file="$1"
    local name="$(basename "$video_file" .mp4)"
    local frame_dir="$OUT_FRAMES/$name"

    if [ -d "$frame_dir" ] && [ "$(ls "$frame_dir"/*.png 2>/dev/null | wc -l)" -ge "$MIN_FRAMES" ]; then
        rm -f "$video_file"
        return
    fi

    python "$REPO/data/preprocess.py" \
        --input "$video_file" --output "$frame_dir" \
        --fps $FPS --max_dim $MAX_DIM 2>/dev/null

    local count=$(ls "$frame_dir"/*.png 2>/dev/null | wc -l)
    if [ "$count" -lt "$MIN_FRAMES" ]; then
        rm -rf "$frame_dir"
        echo "    [skip] $name: $count < $MIN_FRAMES frames"
    else
        echo "    [ok]   $name: $count frames"
    fi
    rm -f "$video_file"
}

for channel_entry in "${CHANNELS[@]}"; do
    channel_name="${channel_entry%%|*}"
    channel_url="${channel_entry##*|}"

    echo ""
    echo "── $channel_name ─────────────────────────────────────────────"

    channel_dir="$OUT_VIDEO/$channel_name"
    mkdir -p "$channel_dir"

    # Download with duration filter and per-channel cap
    yt-dlp \
        --format "bestvideo[ext=mp4][height<=1080]+bestaudio[ext=m4a]/best[ext=mp4][height<=1080]/best" \
        --merge-output-format mp4 \
        --output "$channel_dir/%(id)s.%(ext)s" \
        --playlist-end "$PER_CHANNEL_LIMIT" \
        --match-filter "duration>=${MIN_DURATION} & duration<=${MAX_DURATION}" \
        --ignore-errors \
        --no-warnings \
        --quiet \
        --no-mtime \
        --download-archive "$channel_dir/.downloaded.txt" \
        "$channel_url" 2>&1 | grep -v "^$" | tail -30

    # Extract frames from each downloaded video, then delete the video
    for video_file in "$channel_dir"/*.mp4; do
        [ -f "$video_file" ] || continue
        # Prefix output dir with channel name to avoid collisions
        name="${channel_name}_$(basename "$video_file" .mp4)"
        frame_dir="$OUT_FRAMES/$name"
        if [ -d "$frame_dir" ] && [ "$(ls "$frame_dir"/*.png 2>/dev/null | wc -l)" -ge "$MIN_FRAMES" ]; then
            rm -f "$video_file"
            continue
        fi

        python "$REPO/data/preprocess.py" \
            --input "$video_file" --output "$frame_dir" \
            --fps $FPS --max_dim $MAX_DIM 2>/dev/null

        count=$(ls "$frame_dir"/*.png 2>/dev/null | wc -l)
        if [ "$count" -lt "$MIN_FRAMES" ]; then
            rm -rf "$frame_dir"
        else
            echo "    [ok] $name: $count frames"
        fi
        rm -f "$video_file"
    done

    rmdir "$channel_dir" 2>/dev/null
done

# ── Summary ────────────────────────────────────────────────────────────────
echo ""
echo "═══ Summary ═══"
python "$REPO/data/preprocess.py" --verify "$OUT_FRAMES" --min_frames $MIN_FRAMES

echo ""
echo "Frames directory: $OUT_FRAMES"
echo "Use this as DATA_ROOT for Stage 1 LoRA training."
