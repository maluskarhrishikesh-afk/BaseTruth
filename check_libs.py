import cv2, numpy as np, base64
from basetruth.vision.face import get_mediapipe_faces
from basetruth.kyc.liveness import extract_features, analyze_challenge

# Use sklearn china.jpg
img_path = r'.venv\Lib\site-packages\sklearn\datasets\images\china.jpg'
img = cv2.imread(img_path)
print('Image shape:', img.shape)

faces = get_mediapipe_faces(img)
print('Faces detected:', len(faces))
if faces:
    f = faces[0]
    feat = extract_features(f)
    print('EAR:', round(feat['ear'], 4), '(should be ~0.25-0.35 for open eye)')
    print('Nose rel X:', round(feat['nose_rel_x'], 4), '(should be ~0.4-0.6 for frontal)')
    print()
    
    history = [feat] * 25
    for ch in ['blink', 'turn_left', 'turn_right', 'nod']:
        r = analyze_challenge(history, ch)
        print(ch + ': passed=' + str(r['passed']) + ' feedback=' + r['feedback'])
