from rp import *
import re
import os

# Paths
ati_outputs_folder = "/Users/ryan/CleanCode/Projects/Google2025_Paper/ati_outputs"
edits_folder = "/Users/ryan/CleanCode/Projects/Google2025_Paper/inferblobs_edit_results"
revideo_folder = "../ReVideo"
gwtf_folder = "../GWTF"
output_folder = "video_pairs_all_methods"

os.makedirs(output_folder, exist_ok=True)

# Extract base scene name (remove seed, variations, extensions)
def get_base_name(filename):
    # Remove file extension
    name = filename.replace('.mp4', '')
    # Remove prefixes
    name = re.sub(r'^NoDots_result_', '', name)
    name = re.sub(r'^first_frame_then_tracks---', '', name)
    # Remove seed pattern
    name = re.sub(r'\[Seed \d+\]\s*', '', name)
    # Remove technical suffixes from GWTF
    name = re.sub(r'_<.*>$', '', name)
    # Normalize copy variations
    name = re.sub(r'_copy\d*$', '', name)
    return name.strip()

# Get all files
edit_folders = get_subfolders(edits_folder)
ati_videos = get_all_files(ati_outputs_folder, file_extension_filter="mp4")
revideo_videos = get_all_files(revideo_folder, file_extension_filter="mp4")
gwtf_videos = get_all_files(gwtf_folder, file_extension_filter="mp4")

# Build mappings by base name
scenes = {}

for path in edit_folders:
    base = get_base_name(get_file_name(path))
    if base not in scenes:
        scenes[base] = {'ours': [], 'ati': [], 'revideo': None, 'gwtf': None}
    scenes[base]['ours'].append(path)

for path in ati_videos:
    base = get_base_name(get_file_name(path))
    if base not in scenes:
        scenes[base] = {'ours': [], 'ati': [], 'revideo': None, 'gwtf': None}
    scenes[base]['ati'].append(path)

for path in revideo_videos:
    base = get_base_name(get_file_name(path))
    if base not in scenes:
        scenes[base] = {'ours': [], 'ati': [], 'revideo': None, 'gwtf': None}
    scenes[base]['revideo'] = path

for path in gwtf_videos:
    base = get_base_name(get_file_name(path))
    if base not in scenes:
        scenes[base] = {'ours': [], 'ati': [], 'revideo': None, 'gwtf': None}
    scenes[base]['gwtf'] = path

# Filter to scenes that have all 4 methods
complete_scenes = {k: v for k, v in scenes.items()
                   if v['ours'] and v['ati'] and v['revideo'] and v['gwtf']}

print(f"Found {len(complete_scenes)} complete scenes with all 4 methods")

# Create pairs (pick first from ours/ati if multiple)
pairs = []
for scene_name, methods in complete_scenes.items():
    pair = (
        scene_name,
        methods['ours'][0],  # Pick first if multiple
        methods['ati'][0],   # Pick first if multiple
        methods['revideo'],
        methods['gwtf']
    )
    pairs.append(pair)

print(f"Creating {len(pairs)} comparison videos")

def process(pair):
    try:
        scene_name, ours_folder, ati_path, revideo_path, gwtf_path = pair

        out_name = f"{scene_name}.mp4"
        out_path = path_join(output_folder, out_name)

        if file_exists(out_path):
            print(f"SKIPPING {out_path}")
            return

        print(f"Processing {scene_name}")

        # Load videos
        counter_video = load_video(path_join(ours_folder, "counter_video.mp4"), use_cache=False, show_progress=False)
        ours_video = load_video(path_join(ours_folder, "output_video.mp4"), use_cache=False, show_progress=False)
        tracks_video = load_video(path_join(ours_folder, "counter_tracking_frames.mp4"), use_cache=False, show_progress=False)
        counter_tracks_video = load_video(path_join(ours_folder, "tracking_frames.mp4"), use_cache=False, show_progress=False)

        ati_video = load_video(ati_path, use_cache=False, show_progress=False)
        revideo_video = load_video(revideo_path, use_cache=False, show_progress=False)
        gwtf_video = load_video(gwtf_path, use_cache=False, show_progress=False)

        # Resize all to 49 frames and (480, 720)
        ati_video = resize_list(ati_video, 49)
        ati_video = resize_images(ati_video, size=(480, 720))

        revideo_video = resize_list(revideo_video, 49)
        revideo_video = resize_images(revideo_video, size=(480, 720))

        gwtf_video = resize_list(gwtf_video, 49)
        gwtf_video = resize_images(gwtf_video, size=(480, 720))

        # Add tracks overlay
        def add_tracks(video, track_video):
            video = as_float_images(video)
            track_video = as_float_images(track_video)
            alpha = track_video.max(3, keepdims=True) * 2
            alpha = np.clip(alpha, 0, 1)
            return alpha * track_video + (1 - alpha) * video

        ours_with_tracks = add_tracks(ours_video, counter_tracks_video)
        ati_with_tracks = add_tracks(ati_video, counter_tracks_video)
        input_with_tracks = add_tracks(counter_video, tracks_video)

        # Label all videos
        input_labeled = labeled_images(input_with_tracks, 'Input', font='Futura', size=20)
        ours_labeled = labeled_images(ours_with_tracks, 'Ours', font='Futura', size=20)
        ati_labeled = labeled_images(ati_with_tracks, 'ATI', font='Futura', size=20)
        revideo_labeled = labeled_images(revideo_video, 'ReVideo', font='Futura', size=20)
        gwtf_labeled = labeled_images(gwtf_video, 'GWTF', font='Futura', size=20)

        # Concatenate all 5 horizontally
        cat_vid = horizontally_concatenated_videos(
            input_labeled, ours_labeled, ati_labeled, revideo_labeled, gwtf_labeled
        )

        # Save
        save_video_mp4(cat_vid, out_path, framerate=15, backend='ffmpeg', show_progress=False)
        print(f"Saved {out_path}")

        return cat_vid
    except Exception:
        print_stack_trace()

# Process all pairs
load_files(process, shuffled(pairs), show_progress=True, num_threads=8)

print(f"\nGenerated {len(pairs)} videos in {output_folder}/")
