import os
import json
import requests
import cv2
from moviepy.video.io.ffmpeg_tools import ffmpeg_extract_subclip

# Directory to save the downloaded videos
os.makedirs('msasl_videos/train', exist_ok=True)
os.makedirs('msasl_videos/val', exist_ok=True)
os.makedirs('msasl_videos/test', exist_ok=True)

# Load the JSON files
with open(r'C:\Users\nishi\Videos\MS-ASL\MS-ASL\MSASL_train.json', 'r') as f:
    train_data = json.load(f)

with open(r'C:\Users\nishi\Videos\MS-ASL\MS-ASL\MSASL_val.json', 'r') as f:
    val_data = json.load(f)

with open(r'C:\Users\nishi\Videos\MS-ASL\MS-ASL\MSASL_test.json', 'r') as f:
    test_data = json.load(f)

def download_video(url, save_path):
    """
    Download a video from a URL and save it locally.
    """
    try:
        response = requests.get(url, stream=True)
        if response.status_code == 200:
            with open(save_path, 'wb') as f:
                for chunk in response.iter_content(1024):
                    f.write(chunk)
            print(f"Downloaded {save_path}")
        else:
            print(f"Failed to download {url}")
    except Exception as e:
        print(f"Error downloading {url}: {e}")

def extract_and_save_clip(data, output_dir):
    """
    Download and save video clips based on the start and end times from the JSON data.
    """
    for item in data:
        video_url = item['url']
        start_time = item['start_time']
        end_time = item['end_time']
        label = item['label']
        signer_id = item['signer_id']
        
        # Create a unique file name for each clip
        video_name = f"{label}_{signer_id}.mp4"
        video_path = os.path.join(output_dir, video_name)
        
        # Download the full video first
        download_video(video_url, video_path)
        
        # Extract the specific clip using start_time and end_time
        clip_path = os.path.join(output_dir, f"clip_{label}_{signer_id}.mp4")
        ffmpeg_extract_subclip(video_path, start_time, end_time, targetname=clip_path)
        
        # Optionally delete the full video after extracting the clip
        os.remove(video_path)

# Download and extract clips for train, val, and test sets
extract_and_save_clip(train_data, 'msasl_videos/train')
extract_and_save_clip(val_data, 'msasl_videos/val')
extract_and_save_clip(test_data, 'msasl_videos/test')
