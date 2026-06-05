import os
import yt_dlp

def download_video(url, temp_dir):
    """
    Downloads a short-form video from the given URL.
    Restricts the duration to <= 60 seconds.
    Returns the absolute path to the downloaded video file.
    """
    out_tmpl = os.path.join(temp_dir, '%(id)s.%(ext)s')
    
    ydl_opts = {
        'format': 'bestvideo[ext=mp4]+bestaudio[ext=m4a]/best[ext=mp4]/best',
        'outtmpl': out_tmpl,
        'noplaylist': True,
        'quiet': True,
        'no_warnings': True,
        # 'match_filter': yt_dlp.utils.match_filter_func(
        #     lambda info, incomplete: 'Video is too long' if info.get('duration', 0) > 60 else None
        # ),
    }

    try:
        with yt_dlp.YoutubeDL(ydl_opts) as ydl:
            info = ydl.extract_info(url, download=False)
            duration = info.get('duration')
            if duration and duration > 60:
                raise Exception("Video is too long (over 60 seconds). Only short-form content is supported.")
            
            ydl.download([url])
            
            # Find the downloaded file
            video_id = info.get('id')
            ext = info.get('ext', 'mp4')
            filename = f"{video_id}.{ext}"
            filepath = os.path.join(temp_dir, filename)
            
            if not os.path.exists(filepath):
                # Fallback if outtmpl didn't exactly match (e.g. mkv fallback)
                files = os.listdir(temp_dir)
                for f in files:
                    if f.startswith(video_id):
                        return os.path.join(temp_dir, f)
                raise Exception("Downloaded file not found.")
                
            return filepath
            
    except Exception as e:
        raise Exception(f"Failed to download video: {str(e)}")
