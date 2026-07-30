import cv2
import numpy as np
import onnxruntime as ort

session = ort.InferenceSession('model.onnx')
input_name = session.get_inputs()[0].name

with open('labels.txt', 'r') as f:
    labels = [line.strip() for line in f.readlines()]

def predict(image_bgr, use_rgb=True):
    if use_rgb:
        image = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
    else:
        image = image_bgr.copy()
        
    h, w = image.shape[:2]
    if h < w:
        new_h = 256
        new_w = int(w * (256 / h))
    else:
        new_w = 256
        new_h = int(h * (256 / w))
        
    image = cv2.resize(image, (new_w, new_h))
    start_y = (new_h - 224) // 2
    start_x = (new_w - 224) // 2
    image = image[start_y:start_y+224, start_x:start_x+224]
    
    image = image.astype(np.float32) / 255.0
    image = np.transpose(image, (2, 0, 1))
    image = np.expand_dims(image, axis=0)
    
    outputs = session.run(None, {input_name: image})
    pred_label_raw = outputs[0][0]
    if isinstance(pred_label_raw, (np.ndarray, list)):
        pred_label = pred_label_raw[0]
    else:
        pred_label = pred_label_raw
    probs = outputs[1][0]
    return pred_label, probs

# Create a dummy image
img = np.random.randint(0, 256, (1080, 1920, 3), dtype=np.uint8)

print("RGB PREDICTION:", predict(img, True))
print("BGR PREDICTION:", predict(img, False))
