import rp

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

chosen_seeds = [
    "[Seed 7945]",
    "[Seed 9995]",
    "[Seed 8464]",
    "[Seed 1515]",
    "[Seed 6227]",
    "[Seed 4764]",
    "[Seed 5819]",
    "[Seed 9567]",
    "[Seed 875]",
    "[Seed 5176]",
    "[Seed 8184]",
    "[Seed 5176]",
    "[Seed 2]",
    "[Seed 1579]",
    "[Seed 8917]",
    "[Seed 9651]",
    "[Seed 9471]",
    "[Seed 6303]",
    "[Seed 5072]",
    "[Seed 5072]",
    "[Seed 4370]",
]

chosen_names = [x for x in chosen_names if rp.contains_any(x,chosen_seeds)]

ati_glob     = "/Users/ryan/CleanCode/Projects/Google2025_Paper/ati_outputs/*mp4"
gwtf_glob    = "/Users/ryan/CleanCode/Projects/Google2025_Paper/GWTF/*.mp4"
revideo_glob = "/Users/ryan/CleanCode/Projects/Google2025_Paper/ReVideo/slowmo_NoDots_result_*.mp4"
edits_folder = "/Users/ryan/CleanCode/Projects/Google2025_Paper/inferblobs_edit_results"



# Make pairs
def get_pairs():
    pairs = []
    edit_folders = rp.get_subfolders(edits_folder)
    ati_videos     = rp.glob(ati_glob)
    gwtf_videos    = rp.glob(gwtf_glob)
    revideo_videos = rp.glob(revideo_glob)
    
    filtered_names = [
        x
        for x in chosen_names
        if all(
            any(x in name for name in y)
            for y in [ati_videos, gwtf_videos, revideo_videos]
        )
    ]

    rp.print_lines(filtered_names)

    for name in filtered_names:
        ati_video_options     = [x for x in ati_videos     if name in x]
        revideo_video_options = [x for x in revideo_videos if name in x]
        gwtf_video_options    = [x for x in gwtf_videos    if name in x]
        edit_folder_options   = [x for x in edit_folders   if name in x]
        new_pairs = list(
            rp.cartesian_product(
                edit_folder_options,
                ati_video_options,
                revideo_video_options,
                gwtf_video_options,
                edit_folder_options,
            )
        )
        for pair in new_pairs:
            pairs.append((name,pair))
    return pairs


pairs = get_pairs()


def process(pair):
    def rgbyte(x):
        return rp.as_rgb_images(rp.as_byte_images(x))

    def normalize_video(video):
        video = rp.resize_list(video, 49)
        video = rp.resize_images(video, size=(480, 720))
        return video

    def add_tracks(video, track_video):
        video = rp.as_float_images(video)
        track_video = rp.as_float_images(track_video)
        alpha = track_video.max(3, keepdims=True) * 2
        alpha = rp.np.clip(alpha, 0, 1)
        output = alpha * track_video + (1 - alpha) * video
        return rgbyte(output)

    try:
        name, (edit_folder_path, ati_video_path, revideo_video_path, gwtf_video_path, _) = pair

        out_name=rp.rp.get_file_name(ati_video_path)
        out_path = 'video_pairs/'+out_name

        if rp.file_exists(out_path):
            print("SKIPPING",out_path)
            return out_path

        counter_video        = rp.load_video_via_decord(rp.path_join(edit_folder_path, "counter_video.mp4"))
        output_video         = rp.load_video_via_decord(rp.path_join(edit_folder_path, "output_video.mp4"))
        tracks_video         = rp.load_video_via_decord(rp.path_join(edit_folder_path, "counter_tracking_frames.mp4"))
        counter_tracks_video = rp.load_video_via_decord(rp.path_join(edit_folder_path, "tracking_frames.mp4"))

        ati_video     = rp.load_video_via_decord(ati_video_path,     49)
        revideo_video = rp.load_video_via_decord(revideo_video_path, 49)
        gwtf_video    = rp.load_video_via_decord(gwtf_video_path,    49)

        ati_video     = normalize_video(ati_video)
        revideo_video = normalize_video(revideo_video)
        gwtf_video    = normalize_video(gwtf_video)

        out_output_video     = add_tracks(output_video,  counter_tracks_video)
        ati_output_video     = add_tracks(ati_video,     counter_tracks_video)
        revideo_output_video = add_tracks(revideo_video, counter_tracks_video)
        gwtf_output_video    = add_tracks(gwtf_video,    counter_tracks_video)
        out_counter_video    = add_tracks(counter_video, tracks_video        )

        font_size=30
        out_counter_video    = rgbyte(rp.labeled_images(out_counter_video,    'Input Video', font='Futura', size=font_size))
        out_output_video     = rgbyte(rp.labeled_images(out_output_video,     'Option A',    font='Futura', size=font_size))
        ati_output_video     = rgbyte(rp.labeled_images(ati_output_video,     'Option B',    font='Futura', size=font_size))
        revideo_output_video = rgbyte(rp.labeled_images(revideo_output_video, 'Option C',    font='Futura', size=font_size))
        gwtf_output_video    = rgbyte(rp.labeled_images(gwtf_output_video,    'Option D',    font='Futura', size=font_size))

        #TODO: Shuffle them
        cat_vid=rgbyte(rp.horizontally_concatenated_videos(out_counter_video,out_output_video,ati_output_video,revideo_output_video,gwtf_output_video))
        
        rp.save_video_mp4(cat_vid,out_path,framerate=15,backend='ffmpeg',show_progress=False)

        return out_path
    except Exception:
        rp.print_stack_trace()
        
    return output

files = rp.load_files(process, rp.shuffled(pairs), show_progress=True, num_threads=15, strict=False)
files = rp.get_relative_paths(files)

#Choose Files for Survey
files = rp.unique(files,key=lambda x:rp.get_file_name(x)[len('ATI_0074--'):])

html = r"""<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Video User Study</title>
    <style>
        body {
            font-family: Arial, sans-serif;
            max-width: 90%;
            margin: 0 auto;
            padding: 20px;
            background-color: #f5f5f5;
        }
        
        .video-container {
            background-color: white;
            margin-bottom: 30px;
            padding: 20px;
            border-radius: 8px;
            box-shadow: 0 2px 4px rgba(0,0,0,0.1);
        }
        
        .video-title {
            font-size: 18px;
            font-weight: bold;
            margin-bottom: 15px;
            text-align: center;
        }
        
        video {
            width: 100%;
            /* max-width: 600px; */
            height: auto;
            display: block;
            margin: 0 auto 20px auto;
        }
        
        .questions {
            margin-bottom: 20px;
        }
        
        .question {
            background-color: #f9f9f9;
            padding: 15px;
            margin-bottom: 15px;
            border-radius: 5px;
            border-left: 4px solid #007bff;
        }
        
        .question-title {
            font-weight: bold;
            margin-bottom: 10px;
            color: #333;
        }
        
        .options {
            display: flex;
            justify-content: center;
            gap: 30px;
        }
        
        .option {
            font-size: 16px;
            display: flex;
            align-items: center;
        }
        
        input[type="radio"] {
            margin-right: 8px;
            transform: scale(1.2);
        }
        
        .results-container {
            background-color: white;
            padding: 20px;
            border-radius: 8px;
            box-shadow: 0 2px 4px rgba(0,0,0,0.1);
            margin-top: 30px;
        }
        
        .results-title {
            font-size: 18px;
            font-weight: bold;
            margin-bottom: 15px;
        }
        
        #results {
            width: 100%;
            height: 150px;
            font-family: 'Courier New', monospace;
            font-size: 14px;
            padding: 10px;
            border: 1px solid #ccc;
            border-radius: 4px;
            background-color: #f9f9f9;
            resize: vertical;
        }
    </style>
</head>
<body>
    <h1 style="text-align: center; margin-bottom: 30px;">Video User Study</h1>
    
    <div style="background-color: white; padding: 20px; margin-bottom: 20px; border-radius: 8px; box-shadow: 0 2px 4px rgba(0,0,0,0.1);">
        <p><strong>Instructions:</strong></p>
        <p>You will be shown videos comparing five different motion editing methods side by side: Input (with original motion), Ours, ATI, ReVideo, and GWTF. For each video, please answer the three questions below by selecting your preference.</p>
        <p><strong>Note:</strong> All methods receive the same input video and desired motion trajectories. Your task is to evaluate which method produces better results.</p>
        <p><strong>How to evaluate:</strong> The video edits are indicated by the change in position of the colored dots (shown on Input, Ours, and ATI). Look at matching dots from input to output to determine the intended edits and how well each method preserves content while achieving the desired motion changes.</p>
        <p><strong>When finished:</strong> Click the email link at the bottom of the page and it will prepare an email with your results to send.</p>
    </div>
    
    <div id="video-containers"></div>

    <div class="results-container">
        <div class="results-title">Results (JSON Format):</div>
        <textarea id="results" readonly placeholder="Results will appear here as you make selections..."></textarea>
        <p style="margin-top: 15px; font-weight: bold; color: #333;">
            Once completed, please click here to send results: <a href="#" id="emailLink">rburgert@cs.stonybrook.edu</a>
        </p>
    </div>

    <script>
        const videos = VIDEO_STR;
        
        let results = {};
        
        function updateResults() {
            const resultsTextarea = document.getElementById('results');
            const jsonData = JSON.stringify(results, null, 2);
            resultsTextarea.value = jsonData;
            
            // Update email link with JSON data
            const emailLink = document.getElementById('emailLink');
            const subject = 'Video User Study Results';
            const body = `Here are my user study results:\n\n${jsonData}`;
            emailLink.href = `mailto:rburgert@cs.stonybrook.edu?subject=${encodeURIComponent(subject)}&body=${encodeURIComponent(body)}`;
        }
        
        function createVideoContainer(videoPath, index) {
            const container = document.createElement('div');
            container.className = 'video-container';
            
            const title = document.createElement('div');
            title.className = 'video-title';
            title.textContent = `Video Pair ${index}`;
            
            const video = document.createElement('video');
            video.src = videoPath;
            video.controls = true;
            video.autoplay = true;
            video.muted = true; // Required for autoplay in most browsers
            video.loop = true;
            
            const questionsContainer = document.createElement('div');
            questionsContainer.className = 'questions';
            
            const questions = [
                "Q1: Which method best preserves the input video's content?",
                "Q2: Which method best reflects the desired motion?",
                "Q3: Which method produces the overall best edit?"
            ];
            
            questions.forEach((questionText, qIndex) => {
                const questionDiv = document.createElement('div');
                questionDiv.className = 'question';
                
                const questionTitle = document.createElement('div');
                questionTitle.className = 'question-title';
                questionTitle.textContent = questionText;
                
                const options = document.createElement('div');
                options.className = 'options';
                
                const methods = ['Ours', 'ATI', 'ReVideo', 'GWTF'];
                methods.forEach((method, mIndex) => {
                    const optionLabel = document.createElement('label');
                    optionLabel.className = 'option';
                    const radio = document.createElement('input');
                    radio.type = 'radio';
                    radio.name = `video_${index}_q${qIndex}`;
                    radio.value = method;
                    radio.addEventListener('change', () => {
                        if (radio.checked) {
                            if (!results[index]) results[index] = {};
                            results[index][`q${qIndex + 1}`] = method;
                            updateResults();
                        }
                    });
                    optionLabel.appendChild(radio);
                    optionLabel.appendChild(document.createTextNode(method));
                    options.appendChild(optionLabel);
                });
                
                questionDiv.appendChild(questionTitle);
                questionDiv.appendChild(options);
                questionsContainer.appendChild(questionDiv);
            });
            
            container.appendChild(title);
            container.appendChild(video);
            container.appendChild(questionsContainer);
            
            return container;
        }
        
        function initializeStudy() {
            const videoContainers = document.getElementById('video-containers');
            
            videos.forEach((videoPath, index) => {
                const container = createVideoContainer(videoPath, index);
                videoContainers.appendChild(container);
            });
            
            updateResults();
        }
        
        // Initialize the study when the page loads
        document.addEventListener('DOMContentLoaded', initializeStudy);
    </script>
</body>
</html>
"""

html = html.replace("VIDEO_STR",repr(files))
rp.save_text_file(html, 'index.html')
