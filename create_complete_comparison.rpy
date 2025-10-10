from rp import *
import re
import os

# Paths
existing_pairs_folder = "video_pairs_20"
revideo_folder = "../ReVideo"
gwtf_folder = "../GWTF"
output_folder = "video_pairs_complete"

os.makedirs(output_folder, exist_ok=True)

# Extract base scene name
def get_base_name(name):
    name = re.sub(r'\[Seed \d+\]\s*', '', name)
    name = re.sub(r'_copy\d*$', '', name)
    name = name.strip()
    return name

# Get existing pairs
existing_pairs = get_all_files(existing_pairs_folder, file_extension_filter="mp4")

# Get ReVideo and GWTF videos
# IMPORTANT: Use ONLY NoDots_result_ files from ReVideo (not regular result_ files)
all_revideo_videos = get_all_files(revideo_folder, file_extension_filter="mp4")
revideo_videos = [v for v in all_revideo_videos if 'NoDots_result_' in v]
gwtf_videos = get_all_files(gwtf_folder, file_extension_filter="mp4")

# Build ReVideo mapping by base name
revideo_map = {}
for path in revideo_videos:
    filename = get_file_name(path)
    # Extract scene name after seed
    match = re.search(r'\[Seed \d+\]\s*(.+)\.mp4$', filename)
    if match:
        scene = match.group(1)
        base = get_base_name(scene)
        revideo_map[base] = path

# Build GWTF mapping by base name
gwtf_map = {}
for path in gwtf_videos:
    filename = get_file_name(path)
    # Extract scene name after seed
    match = re.search(r'\[Seed \d+\]\s*(.+?)_<', filename)
    if not match:
        match = re.search(r'\[Seed \d+\]\s*(.+)\.mp4$', filename)
    if match:
        scene = match.group(1)
        base = get_base_name(scene)
        gwtf_map[base] = path

print(f"ReVideo scenes: {len(revideo_map)}")
print(f"GWTF scenes: {len(gwtf_map)}")

# Process existing pairs and add ReVideo/GWTF
pairs = []
for pair_path in existing_pairs:
    filename = get_file_name(pair_path)
    # Extract scene name from format: {scene}.mp4ATI_{num}--[Seed {seed}] {scene}.mp4
    match = re.match(r'(.+?)\.mp4ATI_\d+--\[Seed \d+\] (.+)\.mp4$', filename)
    if match:
        scene_name = match.group(2)
        base_name = get_base_name(scene_name)

        # Find matching ReVideo and GWTF
        revideo_path = revideo_map.get(base_name)
        gwtf_path = gwtf_map.get(base_name)

        if revideo_path and gwtf_path:
            pairs.append((pair_path, revideo_path, gwtf_path, scene_name))
            print(f"Match: {base_name}")

print(f"\nFound {len(pairs)} complete matches with all methods")

def process(pair):
    try:
        existing_pair_path, revideo_path, gwtf_path, scene_name = pair

        # Use the full existing pair filename to preserve uniqueness
        out_name = get_file_name(existing_pair_path)
        out_path = path_join(output_folder, out_name)

        if file_exists(out_path):
            print(f"SKIPPING {out_path}")
            return

        print(f"Processing {scene_name}")

        # Load existing 3-way comparison (Input + Ours + ATI)
        existing_video = load_video(existing_pair_path, use_cache=False, show_progress=False)

        # Load ReVideo and GWTF
        revideo_video = load_video(revideo_path, use_cache=False, show_progress=False)
        gwtf_video = load_video(gwtf_path, use_cache=False, show_progress=False)

        # Resize ReVideo and GWTF to match (49 frames, 480x720)
        revideo_video = resize_list(revideo_video, 49)
        revideo_video = resize_images(revideo_video, size=(480, 720))

        gwtf_video = resize_list(gwtf_video, 49)
        gwtf_video = resize_images(gwtf_video, size=(480, 720))

        # Label ReVideo and GWTF
        revideo_labeled = labeled_images(revideo_video, 'ReVideo', font='Futura', size=20)
        gwtf_labeled = labeled_images(gwtf_video, 'GWTF', font='Futura', size=20)

        # Concatenate: existing (Input+Ours+ATI) + ReVideo + GWTF
        cat_vid = horizontally_concatenated_videos(existing_video, revideo_labeled, gwtf_labeled)

        # Save
        save_video_mp4(cat_vid, out_path, framerate=15, backend='ffmpeg', show_progress=False)
        print(f"Saved {out_path}")

        return cat_vid
    except Exception:
        print_stack_trace()

# Process all pairs
load_files(process, shuffled(pairs), show_progress=True, num_threads=8)

print(f"\nGenerated {len(pairs)} complete comparison videos")
