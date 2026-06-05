import cv2
import base64

def extract_frames(video_path, num_frames=10):
    """
    Extracts evenly spaced frames from a video.
    Returns a list of frames (as BGR NumPy arrays for OpenCV) 
    and a list of base64 encoded strings for the frontend.
    """
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise Exception("Failed to open video file.")
        
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    if total_frames <= 0:
        raise Exception("Video has no frames.")
        
    interval = max(1, total_frames // num_frames)
    
    frames_bgr = []
    frames_base64 = []
    
    for i in range(num_frames):
        frame_idx = min(i * interval, total_frames - 1)
        cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
        ret, frame = cap.read()
        
        if ret:
            # Resize frame if it's too large to save processing time and base64 size
            height, width = frame.shape[:2]
            max_dim = 640
            if max(height, width) > max_dim:
                scale = max_dim / max(height, width)
                frame = cv2.resize(frame, (int(width * scale), int(height * scale)))
                
            frames_bgr.append(frame)
            
            # Encode to base64
            _, buffer = cv2.imencode('.jpg', frame, [cv2.IMWRITE_JPEG_QUALITY, 80])
            b64_str = base64.b64encode(buffer).decode('utf-8')
            frames_base64.append(f"data:image/jpeg;base64,{b64_str}")
            
    cap.release()
    
    if len(frames_bgr) == 0:
        raise Exception("Failed to extract any frames from the video.")
        
    return frames_bgr, frames_base64
