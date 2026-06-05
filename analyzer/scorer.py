import os
from .downloader import download_video
from .extractor import extract_frames
from .face_analysis import analyze_faces
from .clip_classifier import get_clip_scores
from .heuristics import analyze_skin_texture, detect_hand_anomalies

def analyze_video(url, temp_dir):
    video_path = None
    try:
        # 1. Download
        video_path = download_video(url, temp_dir)
        
        # 2. Extract Frames
        frames_bgr, frames_base64 = extract_frames(video_path, num_frames=10)
        
        # 3. Analyze Faces
        has_face, face_data = analyze_faces(frames_bgr)
        if not has_face:
            raise Exception("No person detected \u2014 this tool works best with videos featuring a visible human face.")
            
        # 4. CLIP Classification
        clip_scores = get_clip_scores(frames_bgr)
        
        # 5. Heuristics
        skin_scores = [analyze_skin_texture(frame, fd) for frame, fd in zip(frames_bgr, face_data)]
        hand_flags_list = detect_hand_anomalies(frames_bgr)
        
        # 6. Aggregate Scoring
        frame_scores = []
        total_ai_weight = 0
        red_flags_set = set()
        reassuring_signs_set = set()
        
        for i in range(len(frames_bgr)):
            # Combine CLIP score and Skin Texture score
            # Weight: 60% CLIP, 40% Skin Texture
            c_score = clip_scores[i]
            s_score = skin_scores[i]
            
            combined = (c_score * 0.6) + (s_score * 0.4)
            frame_scores.append(round(combined, 1))
            total_ai_weight += combined
            
            # Record flags
            if c_score > 70:
                red_flags_set.add(f"High AI probability detected by CLIP in frame {i+1}")
            if s_score > 80:
                red_flags_set.add(f"Unnaturally smooth skin texture in frame {i+1}")
            
            for flag in hand_flags_list[i]:
                red_flags_set.add(f"{flag} in frame {i+1}")
                
            if combined < 30:
                reassuring_signs_set.add(f"Natural features detected in frame {i+1}")
                
        # Calculate overall confidence
        avg_score = total_ai_weight / len(frames_bgr)
        
        # Boost confidence if many red flags
        if len(red_flags_set) > 3:
            avg_score = min(100, avg_score + 10)
            
        confidence = round(avg_score, 1)
        
        # Determine verdict
        if confidence > 85:
            verdict = "AI Generated"
        elif confidence > 65:
            verdict = "Likely AI"
        elif confidence > 40:
            verdict = "Uncertain"
        elif confidence > 20:
            verdict = "Likely Real"
        else:
            verdict = "Real"
            
        # Generate summary
        if verdict in ["AI Generated", "Likely AI"]:
            summary = f"Analysis indicates a high likelihood ({confidence}%) that this video contains AI-generated elements. Multiple anomalies were detected across the frames."
        elif verdict == "Uncertain":
            summary = f"The analysis is inconclusive ({confidence}% confidence). Some frames show potential AI artifacts, while others look natural. Proceed with caution."
        else:
            summary = f"The video appears mostly natural ({100 - confidence}% confidence of being real). Very few AI artifacts were detected."
            
        return {
            "verdict": verdict,
            "confidence": confidence,
            "red_flags": list(red_flags_set)[:5], # limit to top 5
            "reassuring_signs": list(reassuring_signs_set)[:3],
            "frame_scores": frame_scores,
            "summary": summary,
            "thumbnails": frames_base64
        }
        
    finally:
        # Cleanup
        if video_path and os.path.exists(video_path):
            try:
                os.remove(video_path)
            except Exception as e:
                print(f"Warning: Failed to delete temp file {video_path}: {e}")
