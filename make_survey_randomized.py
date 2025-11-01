import rp
import random
import sys

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

#This was after manual filtering
chosen_ati_files=[
    "ATI_0065--[Seed 4764] Candle Grab StopCam.mp4",  
    "ATI_0040--[Seed 5072] Truck Before Cab_copy1.mp4",  
    "ATI_0045--[Seed 5176] Judge_ Walk In From Right + Zoom_copy1.mp4",  
    "ATI_0019--[Seed 5819] Cheerleader.mp4",  
    "ATI_0004--[Seed 6227] Boat_ Move Test.mp4",  
    "ATI_0076--[Seed 6303] Sora Basketball_ The ball goes into the hoop.mp4",  
    #"ATI_0039--[Seed 6303] Sora Basketball_ The ball goes into the hoop_copy2.mp4",  
    #"ATI_0052--[Seed 6303] Sora Basketball_ The ball goes into the hoop_copy3.mp4",  
    "ATI_0094--[Seed 7945] Bichon + Corgi _ Bichon Stay Behind.mp4",  
    #"ATI_0050--[Seed 7945] Bichon + Corgi _ Bichon Stay Behind_copy.mp4",  
    #"ATI_0025--[Seed 7945] Bichon + Corgi _ Bichon Stay Behind_copy3.mp4",  
    #"ATI_0099--[Seed 7945] Bichon + Corgi _ Bichon Stay Behind_copy4.mp4",  
    "ATI_0021--[Seed 8184] Judge_ Walk Out.mp4",  
    "ATI_0000--[Seed 8464] Blacks Freeze Camera.mp4",  
    "ATI_0089--[Seed 875] Hot Air Baloons_ Slow camera, make baloons rise.mp4",  
    "ATI_0099--[Seed 8917] Move the car faster forward.mp4",  
    "ATI_0094--[Seed 9471] Shakycam.mp4",  
    "ATI_0065--[Seed 9567] City Biker.mp4",  
    "ATI_0064--[Seed 9651] Reverse Windmills.mp4",  
    "ATI_0053--[Seed 9995] Bichon + Corgi _ Corgi Stay Behind.mp4",  
    #"ATI_0093--[Seed 9995] Bichon + Corgi _ Corgi Stay Behind_copy.mp4",  
    #"ATI_0064--[Seed 9995] Bichon + Corgi _ Corgi Stay Behind_copy3.mp4",  
    #"ATI_0081--[Seed 9995] Bichon + Corgi _ Corgi Stay Behind_copy4.mp4",  
]


#Choose Files for Survey
chosen_ati_files=rp.path_join('video_pairs',chosen_ati_files)


chosen_names = [x for x in chosen_names if rp.rp.contains_any(x,chosen_seeds)]

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

        ati_folder_options = [x for x in ati_video_options if rp.contains_any(x,chosen_ati_files)]

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
            break
        
    return pairs

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

        random.seed(rp.get_sha256_hash(out_name.encode(), format='int'))
        permutation = rp.random_permutation(NUM_CHOICES)
        permutation_string = "".join(map(str, permutation)) #Like "0231" or "2130" etc
        out_path = "video_pairs/" + permutation_string + "><" + out_name

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

        #TODO: Shuffle them
        choices = (
            out_output_video,
            ati_output_video,
            revideo_output_video,
            gwtf_output_video,
        )
        assert len(choices) == NUM_CHOICES
        choices = rp.gather(choices, permutation)

        videos = [out_counter_video,*choices]
        videos = rp.labeled_videos(
            videos,
            ["Input Video", "Option A", "Option B", "Option C", "Option D"],
            font="Futura",
            size=30,
        )
        cat_vid=rgbyte(rp.horizontally_concatenated_videos(videos))
        
        rp.save_video_mp4(cat_vid,out_path,framerate=15,backend='ffmpeg',show_progress=False)

        return out_path
    except Exception:
        rp.print_verbose_stack_trace()
        
    return output

pairs = get_pairs()
NUM_CHOICES=len(pairs[0][1])-1

###########################################################
################# PROCESSING THE VIDEOS ###################
###########################################################

# Check for --skip-videos or --html-only argument
skip_videos = '--skip-videos' in sys.argv or '--html-only' in sys.argv

if skip_videos:
    print("Skipping video generation, using existing videos...")
    files = rp.glob('video_pairs/*.mp4')
    files = rp.get_relative_paths(files)
else:
    files = rp.load_files(process, rp.shuffled(pairs), show_progress=True, num_threads=5, strict=False)
    files = rp.get_relative_paths(files)


########################################################
################# MAKING THE WEBSITE ###################
########################################################


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
            padding: 20px 0 0 0;
            border-radius: 8px;
            box-shadow: 0 2px 4px rgba(0,0,0,0.1);
            overflow: hidden;
        }

        .video-title {
            font-size: 18px;
            font-weight: bold;
            margin-bottom: 15px;
            text-align: center;
            padding: 0 20px;
        }

        video {
            width: calc(100% - 40px);
            height: auto;
            display: block;
            margin: 0 20px;
        }

        .questions {
            margin: 0;
            padding: 0 0 20px 0;
            position: relative;
        }

        .question {
            background-color: #f9f9f9;
            padding: 15px 0 0 0;
            margin: 0 20px 15px 20px;
            border-radius: 5px;
            position: relative;
            overflow: hidden;
        }

        .question::before {
            content: '';
            position: absolute;
            left: 0;
            top: 0;
            bottom: 0;
            width: 4px;
            background-color: #007bff;
            border-radius: 5px 0 0 5px;
        }

        .question-title {
            font-weight: bold;
            margin-bottom: 10px;
            padding: 0 15px;
            color: #333;
        }

        .options {
            display: flex;
            width: 100%;
            margin: 10px 0 15px 0;
        }

        .option {
            flex: 1;
            height: 60px;
            display: flex;
            align-items: center;
            justify-content: center;
            font-size: 20px;
            font-weight: bold;
            cursor: pointer;
            border: 3px solid #ddd;
            background-color: #f9f9f9;
            transition: all 0.2s ease;
            user-select: none;
            position: relative;
        }

        .option::before {
            content: '';
            position: absolute;
            bottom: 100%;
            left: 0;
            right: 0;
            height: 0;
            max-height: 100vh;
            background: linear-gradient(to top, rgba(33, 150, 243, 0.3), rgba(33, 150, 243, 0));
            transition: height 0.3s ease;
            pointer-events: none;
            z-index: -1;
        }

        .option:hover {
            background-color: #e3f2fd;
            border-color: #2196F3;
        }

        .option:hover::before {
            height: 400px;
        }

        .option.selected {
            background-color: #4CAF50;
            border-color: #4CAF50;
            color: white;
        }

        .option input[type="radio"] {
            display: none;
        }

        .spacer {
            flex: 1;
        }

        .video-info {
            background-color: #f0f8ff;
            padding: 15px;
            margin: 0 20px 0 20px;
            border-radius: 5px;
            border-left: 4px solid #4CAF50;
        }

        .video-info-item {
            margin-bottom: 10px;
        }

        .video-info-label {
            font-weight: bold;
            color: #333;
            margin-right: 8px;
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
            title.textContent = `Video Pair ${index + 1}`;
            
            const video = document.createElement('video');
            video.src = videoPath;
            video.controls = true;
            video.autoplay = true;
            video.muted = true; // Required for autoplay in most browsers
            video.loop = true;

            // Add Prompt and Intent section
            const videoInfo = document.createElement('div');
            videoInfo.className = 'video-info';

            const promptItem = document.createElement('div');
            promptItem.className = 'video-info-item';
            const promptLabel = document.createElement('span');
            promptLabel.className = 'video-info-label';
            promptLabel.textContent = 'Prompt:';
            const promptText = document.createElement('span');
            promptText.textContent = '[Prompt goes here]';
            promptItem.appendChild(promptLabel);
            promptItem.appendChild(promptText);

            const intentItem = document.createElement('div');
            intentItem.className = 'video-info-item';
            const intentLabel = document.createElement('span');
            intentLabel.className = 'video-info-label';
            intentLabel.textContent = 'Intent:';
            const intentText = document.createElement('span');
            intentText.textContent = '[Intent goes here]';
            intentItem.appendChild(intentLabel);
            intentItem.appendChild(intentText);

            videoInfo.appendChild(promptItem);
            videoInfo.appendChild(intentItem);

            const questionsContainer = document.createElement('div');
            questionsContainer.className = 'questions';
            
            const questions = [
                "Q1: Which method best preserves the input video's content?",
                "Q2: Which method best reflects the desired motion?",
                "Q3: Which method produces the overall best edit?"
            ];

            const originalMethods = ['Ours', 'ATI', 'ReVideo', 'GWTF'];
            const permutation_string = videoPath.split('/')[1].split('><')[0];
            const permutation = permutation_string.split('').map(Number);
            
            questions.forEach((questionText, qIndex) => {
                const questionDiv = document.createElement('div');
                questionDiv.className = 'question';

                const questionTitle = document.createElement('div');
                questionTitle.className = 'question-title';
                questionTitle.textContent = questionText;

                const options = document.createElement('div');
                options.className = 'options';

                // Add spacer for Input section (first 20% of video)
                const spacer = document.createElement('div');
                spacer.className = 'spacer';
                options.appendChild(spacer);

                const optionLabels = ['A', 'B', 'C', 'D'];
                optionLabels.forEach((label, mIndex) => {
                    const optionDiv = document.createElement('div');
                    optionDiv.className = 'option';
                    optionDiv.textContent = label;

                    const radio = document.createElement('input');
                    radio.type = 'radio';
                    radio.name = `video_${index}_q${qIndex}`;

                    const permuted_method_index = permutation[mIndex];
                    const original_method = originalMethods[permuted_method_index];
                    radio.value = original_method;

                    optionDiv.appendChild(radio);

                    optionDiv.addEventListener('click', () => {
                        // Remove selected class from all options in this question
                        options.querySelectorAll('.option').forEach(opt => {
                            opt.classList.remove('selected');
                        });

                        // Add selected class to this option
                        optionDiv.classList.add('selected');

                        // Check the radio button
                        radio.checked = true;

                        if (!results[videoPath]) results[videoPath] = {};
                        results[videoPath][`q${qIndex + 1}`] = radio.value;
                        updateResults();
                    });

                    options.appendChild(optionDiv);
                });

                questionDiv.appendChild(questionTitle);
                questionDiv.appendChild(options);
                questionsContainer.appendChild(questionDiv);
            });
            
            container.appendChild(title);
            container.appendChild(video);
            container.appendChild(videoInfo);
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
