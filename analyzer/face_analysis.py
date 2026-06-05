import cv2
import mediapipe as mp

mp_face_mesh = mp.solutions.face_mesh

def analyze_faces(frames_bgr):
    """
    Runs Mediapipe FaceMesh on the frames to detect faces.
    Returns:
    - has_face: True if a face is detected in the majority of frames.
    - face_data: List of landmarks for each frame (None if no face).
    """
    face_data = []
    frames_with_face = 0
    
    with mp_face_mesh.FaceMesh(
        static_image_mode=True,
        max_num_faces=1,
        refine_landmarks=True,
        min_detection_confidence=0.5
    ) as face_mesh:
        for frame in frames_bgr:
            # Convert the BGR image to RGB before processing.
            image_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            results = face_mesh.process(image_rgb)
            
            if results.multi_face_landmarks:
                frames_with_face += 1
                face_data.append(results.multi_face_landmarks[0])
            else:
                face_data.append(None)
                
    has_face = frames_with_face >= (len(frames_bgr) / 2)
    return has_face, face_data
