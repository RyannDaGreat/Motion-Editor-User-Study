chosen_names = [
    "[Seed 2] Kittycat Fish",
    "[Seed 875] Hot Air Baloons_ Slow camera, make baloons rise",
    "[Seed 950] City Biker",
    "[Seed 1131] Kids Racing",
    "[Seed 1201] Move the car faster forward_copy3",
    "[Seed 1514] Hot Air Baloons_ Swap all three and make them rise",
    "[Seed 1515] Blacks Freeze Camera_copy",
    "[Seed 1515] Blacks Freeze Camera_copy2",
    "[Seed 1579] Motorcycle Chase_ The motorcycle chases the car_copy1",
    "[Seed 2222] Blacks Freeze Camera",
    "[Seed 4360] Hot Air Baloons_ Slow camera, make baloons rise",
    "[Seed 4370] [Failure] Stop Sign Lady_copy6",
    "[Seed 4409] Cheerleader Two_copy",
    "[Seed 4764] Candle Grab StopCam",
    "[Seed 5065] Bichon + Corgi _ Bichon Stays Behind",
    "[Seed 5072] Truck Before Cab_copy1",
    "[Seed 5176] Judge_ Walk In From Right + Zoom_copy1",
    "[Seed 5176] Judge_ Walk Out_copy1",
    "[Seed 5280] Candle Grab StopCam",
    "[Seed 5440] Penguins Walk Together",
    "[Seed 5666] Candle Grab StopCam",
    "[Seed 5819] Cheerleader",
    "[Seed 6227] Boat_ Move Test",
    "[Seed 6303] Sora Basketball_ The ball goes into the hoop",
    "[Seed 6933] [Failure] Stop Sign Lady",
    "[Seed 7945] Bichon + Corgi _ Bichon Stay Behind",
    "[Seed 8184] Judge_ Walk Out",
    "[Seed 8464] Blacks Freeze Camera",
    "[Seed 8848] Shakycam",
    "[Seed 8917] Move the car faster forward",
    "[Seed 9221] Knight Chases Windmill [Slower]",
    "[Seed 9471] Shakycam",
    "[Seed 9567] City Biker",
    "[Seed 9593] Move the car faster forward",
    "[Seed 9651] Reverse Windmills",
    "[Seed 9995] Bichon + Corgi _ Corgi Stay Behind",
    "[Seed 9995] Blacks Swan Go Faster.mp4",
]
ati_outputs_folder = "/Users/ryan/CleanCode/Projects/Google2025_Paper/ati_outputs"
edits_folder = "/Users/ryan/CleanCode/Projects/Google2025_Paper/inferblobs_edit_results"

# Make pairs
def get_pairs():
    pairs = []
    edit_folders = get_subfolders(edits_folder)
    ati_videos = get_all_files(ati_outputs_folder, file_extension_filter="mp4")
    for name in chosen_names:
        ati_video_options = [x for x in ati_videos if name in x]
        edit_folder_options = [x for x in edit_folders if name in x]
        pairs += list(cartesian_product(edit_folder_options, ati_video_options))
    return pairs


pairs = get_pairs()


def process(pair):
    edit_folder, ati_video = pair

    video_cache = True
    ati_video = load_video(ati_video, use_cache=video_cache)
    counter_video = load_video(path_join(edit_folder, "counter_video.mp4"), use_cache=video_cache)
    output_video = load_video(path_join(edit_folder, "output_video.mp4"), use_cache=video_cache)
    tracks_video = load_video(path_join(edit_folder, "counter_tracking_frames.mp4"), use_cache=video_cache)
    counter_tracks_video = load_video(path_join(edit_folder, "tracking_frames.mp4"), use_cache=video_cache)

    ati_video = resize_list(ati_video, 49)
    ati_video = resize_images(ati_video,size= (480, 720))

    def add_tracks(video, track_video):
        video = as_float_images(video)
        track_video = as_float_images(track_video)
        alpha = track_video.max(3, keepdims=True) * 2
        alpha = np.clip(alpha, 0, 1)
        return alpha * track_video + (1 - alpha) * video

    out_output_video = add_tracks(output_video, counter_tracks_video)
    ati_output_video = add_tracks(ati_video, counter_tracks_video)
    out_counter_video = add_tracks(counter_video, tracks_video)

    return out_counter_video, out_output_video, ati_output_video

