import cv2
import numpy as np
import mediapipe as mp

mp_hands = mp.solutions.hands

def analyze_skin_texture(frame_bgr, face_landmarks):
    """
    Analyzes skin texture variance in the cheek region.
    Unnaturally smooth skin (low variance) is a common AI artifact.
    Returns a score (0-100) where higher means more likely AI (too smooth).
    """
    if not face_landmarks:
        return 0
        
    height, width, _ = frame_bgr.shape
    
    # Cheek landmarks (approximate indices for left and right cheeks)
    left_cheek_idx = [116, 117, 118, 119, 120, 100, 101, 50]
    right_cheek_idx = [345, 346, 347, 348, 349, 329, 330, 280]
    
    pts = []
    for idx in left_cheek_idx + right_cheek_idx:
        lm = face_landmarks.landmark[idx]
        pts.append([int(lm.x * width), int(lm.y * height)])
        
    pts = np.array(pts, dtype=np.int32)
    
    # Create mask for cheek region
    mask = np.zeros((height, width), dtype=np.uint8)
    if len(pts) > 0:
        cv2.fillConvexPoly(mask, cv2.convexHull(pts), 255)
        
    # Calculate Laplacian variance (focus on high frequency details / texture)
    gray = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2GRAY)
    laplacian = cv2.Laplacian(gray, cv2.CV_64F)
    
    # Only consider pixels within the cheek mask
    masked_laplacian = laplacian[mask == 255]
    if len(masked_laplacian) == 0:
        return 0
        
    variance = np.var(masked_laplacian)
    
    # Normalize variance to a 0-100 score (heuristic thresholds)
    # Natural skin usually has variance > 50-100 depending on lighting/resolution
    # Extremely low variance (< 10) is suspicious
    ai_score = max(0, min(100, 100 - (variance / 2.0)))
    
    return float(ai_score)

def detect_hand_anomalies(frames_bgr):
    """
    Uses Mediapipe Hands to check for finger count anomalies or fused fingers.
    Returns a list of flags per frame.
    """
    flags = []
    with mp_hands.Hands(
        static_image_mode=True,
        max_num_hands=2,
        min_detection_confidence=0.5
    ) as hands:
        for frame in frames_bgr:
            image_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            results = hands.process(image_rgb)
            
            frame_flags = []
            if results.multi_hand_landmarks:
                for hand_landmarks in results.multi_hand_landmarks:
                    # Basic heuristic: Check if distance between consecutive fingertips is unusually small (fused)
                    # or if hand proportions are highly skewed.
                    # Fingertip indices: 4 (Thumb), 8 (Index), 12 (Middle), 16 (Ring), 20 (Pinky)
                    tips = [4, 8, 12, 16, 20]
                    height, width, _ = frame.shape
                    
                    points = [(hand_landmarks.landmark[i].x * width, hand_landmarks.landmark[i].y * height) for i in tips]
                    
                    # Simplistic check for overlapping/fused fingertips
                    fused_count = 0
                    for i in range(len(points)):
                        for j in range(i + 1, len(points)):
                            dist = np.sqrt((points[i][0] - points[j][0])**2 + (points[i][1] - points[j][1])**2)
                            if dist < (width * 0.01): # Extremely close fingertips
                                fused_count += 1
                                
                    if fused_count > 0:
                        frame_flags.append("Possible fused fingers detected")
                        
            flags.append(frame_flags)
            
    return flags
