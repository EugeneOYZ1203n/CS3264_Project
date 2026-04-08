import json
import os
import requests
import argparse

def download_subset(json_path, output_dir, max_videos_per_word=2):
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    with open(json_path, 'r') as f:
        data = json.load(f)

    download_count = 0

    for entry in data:
        gloss = entry['gloss'].lower()
        
        word_dir = os.path.join(output_dir, gloss)
        if not os.path.exists(word_dir):
            os.makedirs(word_dir)

        # Get up to 2 instances
        instances = entry['instances'][:max_videos_per_word]
        
        for i, inst in enumerate(instances):
            video_id = inst['video_id']
            url = inst['url']
            file_ext = ".mp4" # Default extension
            
            # Note: Many WLASL links are direct URLs to various sites
            # Some may require youtube-dl/yt-dlp for YouTube links
            print(f"Downloading {gloss} ({i+1}/{max_videos_per_word}): {video_id}")
            
            try:
                # Basic download logic for direct links
                # If the link is YouTube, you'd need to use 'yt-dlp' library here
                response = requests.get(url, timeout=10)
                if response.status_code == 200:
                    file_path = os.path.join(word_dir, f"{video_id}{file_ext}")
                    with open(file_path, 'wb') as vid_file:
                        vid_file.write(response.content)
                    download_count += 1
            except Exception as e:
                print(f"Failed to download {video_id}: {e}")

    print(f"Finished! Total videos downloaded: {download_count}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Download WLASL subset")
    parser.add_argument("--json", type=str, default="WLASL_v0.3.json", help="Path to WLASL JSON file")
    parser.add_argument("--out", type=str, required=True, help="Output directory")
    
    args = parser.parse_args()
    download_subset(args.json, args.out)