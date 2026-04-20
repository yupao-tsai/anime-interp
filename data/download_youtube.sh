#!/bin/bash
# Download anime clips from official YouTube sources using yt-dlp.
# These are official/legal uploads: trailers, PV, studio-released content.
#
# Output: /storage/SSD3/yptsai/data/youtube_anime/
# After download, frames are extracted automatically.
# Target: ≥17 consecutive frames per clip for Stage 1 LoRA training.

VENV=/storage/SSD2/users/yptsai/program/venv
REPO=/storage/SSD2/users/yptsai/program/anime_interp
OUT_VIDEO=/storage/SSD3/yptsai/data/youtube_anime/videos
OUT_FRAMES=/storage/SSD3/yptsai/data/youtube_anime/frames

FPS=12          # extraction fps (anime is typically 8-24fps; 12 gives ~2 frames/sec at 24fps source)
MIN_FRAMES=33   # minimum frames per clip to keep
MAX_DIM=768     # resize long edge to this

source "$VENV/bin/activate"
mkdir -p "$OUT_VIDEO" "$OUT_FRAMES"

# ── Official anime YouTube sources ─────────────────────────────────────────
# Format: "URL" "clip_name"
# Add more URLs here. These should be official studio/platform uploads.
# Avoid: fan uploads, pirated content, full episodes from unofficial sources.

URLS=(
    # --- Crunchyroll Official (公式) ---
    # Add specific video URLs here, e.g.:
    # "https://www.youtube.com/watch?v=XXXXX" "crunchyroll_clip1"

    # --- Studio Ghibli Official ---
    # "https://www.youtube.com/watch?v=XXXXX" "ghibli_clip1"

    # --- Official anime trailers (PV) ---
    # PVs and trailers are released by studios for promotion.
    # They contain high-quality cel animation, ideal for training.
    # Example (replace with actual URLs):
    # "https://www.youtube.com/watch?v=XXXXX" "anime_pv_1"
)

# ── Playlist download (uncomment and set playlist URL) ─────────────────────
# Useful for downloading all PVs from a channel, e.g.:
# PLAYLIST_URL="https://www.youtube.com/@StudioGhibliOfficial/videos"
PLAYLIST_URL=""

# ─────────────────────────────────────────────────────────────────────────────

download_and_extract() {
    local url="$1"
    local name="$2"
    local video_file="$OUT_VIDEO/${name}.mp4"
    local frame_dir="$OUT_FRAMES/$name"

    echo ""
    echo "=== $name ==="

    # Skip if frames already extracted
    if [ -d "$frame_dir" ] && [ "$(ls "$frame_dir"/*.png 2>/dev/null | wc -l)" -ge "$MIN_FRAMES" ]; then
        echo "  Already extracted ($(ls "$frame_dir"/*.png | wc -l) frames), skipping."
        return
    fi

    # Download video
    if [ ! -f "$video_file" ]; then
        echo "  Downloading..."
        yt-dlp \
            --format "bestvideo[ext=mp4][height<=1080]+bestaudio[ext=m4a]/best[ext=mp4]/best" \
            --merge-output-format mp4 \
            --output "$video_file" \
            --no-playlist \
            --quiet \
            "$url" || { echo "  [SKIP] Download failed"; return; }
    fi

    # Extract frames
    echo "  Extracting frames at ${FPS}fps..."
    python "$REPO/data/preprocess.py" --input "$video_file" --output "$frame_dir" --fps $FPS --max_dim $MAX_DIM

    count=$(ls "$frame_dir"/*.png 2>/dev/null | wc -l)
    echo "  → $count frames"

    if [ "$count" -lt "$MIN_FRAMES" ]; then
        echo "  [SKIP] Too few frames ($count < $MIN_FRAMES), removing."
        rm -rf "$frame_dir"
    fi

    # Delete video after extraction to save space
    rm -f "$video_file"
}

# ── Process individual URLs ────────────────────────────────────────────────
if [ "${#URLS[@]}" -gt 0 ]; then
    for i in $(seq 0 2 $((${#URLS[@]} - 1))); do
        url="${URLS[$i]}"
        name="${URLS[$((i+1))]}"
        download_and_extract "$url" "$name"
    done
fi

# ── Process playlist ───────────────────────────────────────────────────────
if [ -n "$PLAYLIST_URL" ]; then
    echo ""
    echo "=== Downloading playlist ==="
    yt-dlp \
        --format "bestvideo[ext=mp4][height<=1080]+bestaudio[ext=m4a]/best[ext=mp4]/best" \
        --merge-output-format mp4 \
        --output "$OUT_VIDEO/%(title)s_%(id)s.%(ext)s" \
        --quiet \
        --ignore-errors \
        "$PLAYLIST_URL"

    echo "Extracting frames from all downloaded videos..."
    for video in "$OUT_VIDEO"/*.mp4; do
        [ -f "$video" ] || continue
        name=$(basename "$video" .mp4)
        frame_dir="$OUT_FRAMES/$name"
        echo "  $name..."
        python "$REPO/data/preprocess.py" --input "$video" --output "$frame_dir" --fps $FPS --max_dim $MAX_DIM
        count=$(ls "$frame_dir"/*.png 2>/dev/null | wc -l)
        if [ "$count" -lt "$MIN_FRAMES" ]; then
            rm -rf "$frame_dir"
        fi
        rm -f "$video"
    done
fi

# ── Summary ────────────────────────────────────────────────────────────────
echo ""
echo "=== Summary ==="
python "$REPO/data/preprocess.py" --verify "$OUT_FRAMES" --min_frames $MIN_FRAMES
echo ""
echo "Frames directory: $OUT_FRAMES"
echo "Use this as DATA_ROOT for Stage 1 LoRA training."
