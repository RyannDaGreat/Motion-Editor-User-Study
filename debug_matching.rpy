from rp import *
import re

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

# Get sample files
ati_outputs_folder = "/Users/ryan/CleanCode/Projects/Google2025_Paper/ati_outputs"
edits_folder = "/Users/ryan/CleanCode/Projects/Google2025_Paper/inferblobs_edit_results"
revideo_folder = "../ReVideo"
gwtf_folder = "../GWTF"

edit_folders = get_subfolders(edits_folder)[:5]
ati_videos = get_all_files(ati_outputs_folder, file_extension_filter="mp4")[:5]
revideo_videos = get_all_files(revideo_folder, file_extension_filter="mp4")[:5]
gwtf_videos = get_all_files(gwtf_folder, file_extension_filter="mp4")[:5]

print("OURS (edit folders):")
for path in edit_folders:
    print(f"  {get_file_name(path)} -> {get_base_name(get_file_name(path))}")

print("\nATI:")
for path in ati_videos:
    print(f"  {get_file_name(path)} -> {get_base_name(get_file_name(path))}")

print("\nReVideo:")
for path in revideo_videos:
    print(f"  {get_file_name(path)} -> {get_base_name(get_file_name(path))}")

print("\nGWTF:")
for path in gwtf_videos:
    print(f"  {get_file_name(path)} -> {get_base_name(get_file_name(path))}")
