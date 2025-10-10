from rp import *
import re

# Paths to the video folders
revideo_folder = "../ReVideo"
gwtf_folder = "../GWTF"
output_folder = "video_pairs_revideo_gwtf"

# Create output folder if needed
import os
os.makedirs(output_folder, exist_ok=True)

# Get all videos
revideo_videos = get_all_files(revideo_folder, file_extension_filter="mp4")
gwtf_videos = get_all_files(gwtf_folder, file_extension_filter="mp4")

# Parse seed from filename
def get_seed(filename):
    match = re.search(r'\[Seed (\d+)\]', filename)
    return match.group(1) if match else None

# Create pairs by matching seeds
def get_pairs():
    pairs = []
    for revideo_path in revideo_videos:
        seed = get_seed(revideo_path)
        if not seed:
            continue

        # Find matching GWTF video
        for gwtf_path in gwtf_videos:
            if f'[Seed {seed}]' in gwtf_path:
                pairs.append((seed, revideo_path, gwtf_path))
                break

    return pairs

pairs = get_pairs()
print(f"Found {len(pairs)} matching pairs")

def process(pair):
    try:
        seed, revideo_path, gwtf_path = pair

        # Create output filename
        out_name = f"Seed{seed}_ReVideo_vs_GWTF.mp4"
        out_path = path_join(output_folder, out_name)

        if file_exists(out_path):
            print(f"SKIPPING {out_path}")
            return

        print(f"Processing Seed {seed}")

        # Load videos
        revideo_video = load_video(revideo_path, use_cache=False, show_progress=False)
        gwtf_video = load_video(gwtf_path, use_cache=False, show_progress=False)

        # Resize to match study requirements: 49 frames and (480, 720)
        revideo_video = resize_list(revideo_video, 49)
        revideo_video = resize_images(revideo_video, size=(480, 720))

        gwtf_video = resize_list(gwtf_video, 49)
        gwtf_video = resize_images(gwtf_video, size=(480, 720))

        # Extract input frame (first frame) from GWTF for context
        input_frame = gwtf_video[0]
        input_video = [input_frame] * len(gwtf_video)  # Repeat first frame
        input_video = labeled_images(input_video, 'Input Video', font='Futura', size=20)

        # Label the videos
        revideo_labeled = labeled_images(revideo_video, 'Option A (ReVideo)', font='Futura', size=20)
        gwtf_labeled = labeled_images(gwtf_video, 'Option B (GWTF)', font='Futura', size=20)

        # Concatenate horizontally
        cat_vid = horizontally_concatenated_videos(input_video, revideo_labeled, gwtf_labeled)

        # Save
        save_video_mp4(cat_vid, out_path, framerate=15, backend='ffmpeg', show_progress=False)
        print(f"Saved {out_path}")

        return cat_vid
    except Exception:
        print_stack_trace()

# Process all pairs
load_files(process, shuffled(pairs), show_progress=True, num_threads=8)
