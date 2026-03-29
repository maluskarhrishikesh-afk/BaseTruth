# Identity Verification & Fraud Detection

BaseTruth incorporates a robust, **offline-first** computer vision pipeline specifically engineered for identity fraud detection. This allows the system to verify that the person on an ID document (Aadhaar, PAN, Passport) matches a provided live Selfie, mathematically scoring the likelihood of fraud.

## Core Technology Stack

The pipeline runs locally without making any external API calls, ensuring high privacy, reducing costs, and keeping latency strictly bounded by the edge hardware.

| Component | Responsibility | Purpose in Flow |
| --- | --- | --- |
| **OpenCV** (`cv2`) | Image Processing | Decodes raw byte streams from the UI, manages RGB/BGR color space conversions, heavily resizes images for memory safety, and draws visual forensic evidence (bounding boxes). |
| **RetinaFace** | Face Detection | Deep learning model that acts as the "eyes". It aggressively searches the document to locate the face, cropping it securely and mapping 5 key facial landmarks (eyes, nose, mouth) required to properly align the face angle. |
| **ArcFace** | Identity Recognition | Deep learning model that acts as the "brain". It takes the aligned face from RetinaFace and translates it into a 512-dimensional vector (an embedding). |
| **ONNX Runtime** | Inference Engine | Executes both RetinaFace and ArcFace locally on the CPU (or GPU if available) using the `buffalo_l` pre-trained model pack, bypassing heavy dependencies like PyTorch. |

## Workflow

BaseTruth offers two variants of Identity Verification:

### 1. Static Verification (ID vs Selfie)
1. **Ingestion**: The operator uploads an ID document and a Selfie via the `🧑‍💻 Identity` screen.
2. **Decoding**: OpenCV translates the inputs into mathematical arrays.
3. **Detection & Vectorization**: RetinaFace isolates the largest face. ArcFace converts it into an embedding vector.
4. **Comparison**: The system calculates the **Cosine Similarity** (threshold > 0.40).

### 2. Video KYC (Live Stream)
Designed specifically to prevent impersonation or photo-spoofing fraud.

1. **Reference Extraction**: The user provides the base truth ID document. The AI extracts the face embedding safely.
2. **WebRTC Stream Setup**: The browser intercepts the user's webcam at 15 FPS and streams packets down to the backend `VideoKYCProcessor`.
3. **Liveness Detection**: The system calculates the lateral projection angle between the nose and eyes (using the RetinaFace spatial ratio). By asking the user to turn their head `Left` or `Right`, the system validates physical 3D presence rather than a flat, static printed photo.
4. **Continuous Matching**: Alongside tracking movement, ArcFace verifies every single frame against the Reference ID document in real time, projecting `VERIFIED MATCH` or `FRAUD ALERT` directly over the user's face stream.

## Why Not External APIs?

Many competitive products rely on AWS Rekognition or Azure Face API. BaseTruth uses this local ONNX stack because:
* PII (Personally Identifiable Information) never leaves the server.
* Fraud operators can scan thousands of historical case documents per minute without incurring massive per-call cloud API bills.
* The system elegantly degrades; if the internet goes down, fraud checking continues seamlessly.

## Location in Codebase
The primary driver for this engine lives within `src/basetruth/vision/face.py` and is instantiated lazily to preserve strict, fast app loading times. It manages the `MPLCONFIGDIR` and `/tmp/cache` paths to securely handle the containerized environment.
