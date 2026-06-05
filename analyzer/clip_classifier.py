from transformers import CLIPProcessor, CLIPModel
from PIL import Image
import cv2
import torch

# Initialize model globally to avoid reloading on every request
model_id = "openai/clip-vit-base-patch32"
device = "cuda" if torch.cuda.is_available() else "cpu"

try:
    model = CLIPModel.from_pretrained(model_id).to(device)
    processor = CLIPProcessor.from_pretrained(model_id)
except Exception as e:
    print(f"Warning: Failed to load CLIP model. {e}")
    model = None
    processor = None

def get_clip_scores(frames_bgr):
    """
    Uses CLIP to zero-shot classify each frame.
    Prompts: "a real human face", "an AI generated face", "a deepfake video", "CGI human"
    Returns a list of AI-probability scores (0-100) per frame.
    """
    if model is None or processor is None:
        return [0] * len(frames_bgr)
        
    prompts = ["a real photograph of a human face", "an AI generated face", "a deepfake video", "a CGI rendered human"]
    
    scores = []
    
    for frame in frames_bgr:
        # Convert BGR to RGB PIL Image
        image_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        pil_image = Image.fromarray(image_rgb)
        
        inputs = processor(text=prompts, images=pil_image, return_tensors="pt", padding=True).to(device)
        
        with torch.no_grad():
            outputs = model(**inputs)
            logits_per_image = outputs.logits_per_image # this is the image-text similarity score
            probs = logits_per_image.softmax(dim=1).cpu().numpy()[0]
            
            # Combine probabilities for AI-related prompts (index 1, 2, 3)
            ai_prob = (probs[1] + probs[2] + probs[3]) * 100
            scores.append(float(ai_prob))
            
    return scores
