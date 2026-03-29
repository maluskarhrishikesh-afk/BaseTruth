import logging
import math
from typing import Any, Optional
import base64

import cv2
import numpy as np

from basetruth.vision.face import get_face_analyzer

log = logging.getLogger(__name__)

class VideoKYCProcessor:
    """Real-time TCP/WebSocket processor for active Video KYC."""
    
    def __init__(self) -> None:
        self.reference_embedding: Optional[np.ndarray] = None
        self.match_threshold: float = 0.40
        self.face_app = get_face_analyzer()
        
        # State tracking
        self.current_score: float = 0.0
        self.is_match: bool = False
        self.liveness_state: str = "Center"
        self.head_turn_passed: bool = False

    def set_reference_embedding(self, embedding: np.ndarray) -> None:
        """Injects the ArcFace embedding of the ID Document user."""
        self.reference_embedding = embedding
        
    def process_base64_frame(self, b64_str: str) -> str:
        """Process an incoming base64 jpeg string and return an annotated base64 jpeg."""
        # Clean prefix if it exists (e.g. data:image/jpeg;base64,...)
        if "," in b64_str:
            b64_str = b64_str.split(",", 1)[1]
            
        # Decode base64 to OpenCV image
        img_data = base64.b64decode(b64_str)
        nparr = np.frombuffer(img_data, np.uint8)
        img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        
        if img is None:
            return ""
            
        try:
            # Detect faces using RetinaFace
            faces = self.face_app.get(img)
            
            if faces:
                # Find the largest bounding box (primary subject)
                primary_face = max(faces, key=lambda f: (f.bbox[2]-f.bbox[0]) * (f.bbox[3]-f.bbox[1]))
                box = primary_face.bbox.astype(int)
                
                color = (0, 0, 255) # Red denotes no-match or no reference
                
                # Draw facial landmarks
                if primary_face.kps is not None:
                    # Update Liveness (head orientation)
                    self._update_liveness(primary_face.kps)
                    
                    kps = primary_face.kps.astype(int)
                    for p in kps:
                        cv2.circle(img, (p[0], p[1]), 3, (255, 0, 0), cv2.FILLED)
                
                # Perform continuous identity matching if we have an ID
                text = "Live Stream Active"
                if self.reference_embedding is not None:
                    emb = primary_face.normed_embedding
                    sim = float(np.dot(emb, self.reference_embedding))
                    
                    self.current_score = min(max((sim - (-0.5)) / (1.0 - (-0.5)) * 100, 0), 100)
                    self.is_match = sim >= self.match_threshold
                    
                    if self.is_match:
                        color = (0, 255, 0) # Green for match
                        text = f"VERIFIED: {self.current_score:.1f}%"
                    else:
                        text = f"FRAUD: {self.current_score:.1f}%"
                
                # Draw bounding box
                cv2.rectangle(img, (box[0], box[1]), (box[2], box[3]), color, 3)
                
                # Render UI Text onto the frame
                cv2.putText(img, text, (box[0], box[1] - 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)
                
                live_text = f"Live: {self.liveness_state} {'(PASS)' if self.head_turn_passed else ''}"
                cv2.putText(img, live_text, (box[0], box[1] - 7), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 2)
                
            else:
                cv2.putText(img, "No face detected in video feed.", (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)
                
        except Exception as e:
            log.exception("Error in video KYC frame processing: %s", e)
            
        # Re-encode to base64 jpeg
        _, buffer = cv2.imencode('.jpg', img, [int(cv2.IMWRITE_JPEG_QUALITY), 60])
        out_b64 = base64.b64encode(buffer).decode('utf-8')
        return "data:image/jpeg;base64," + out_b64

    def _update_liveness(self, landmarks: np.ndarray) -> None:
        """Detect side-to-side head movement based on 2D facial point ratios.
        landmarks: [left_eye, right_eye, nose, left_mouth, right_mouth]
        """
        left_eye_x = landmarks[0][0]
        right_eye_x = landmarks[1][0]
        nose_x = landmarks[2][0]
        
        # Horizontal distances from eyes to nose
        dist_left = abs(nose_x - left_eye_x)
        dist_right = abs(right_eye_x - nose_x)
        
        # Protect against division by zero 
        if dist_right < 1:
            dist_right = 1.0
            
        ratio = dist_left / dist_right
        
        if ratio > 1.6:
            self.liveness_state = "Turned Right"
            self.head_turn_passed = True
        elif ratio < 0.6:
            self.liveness_state = "Turned Left"
            self.head_turn_passed = True
        else:
            self.liveness_state = "Center"
