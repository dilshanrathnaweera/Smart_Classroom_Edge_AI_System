import cv2
import numpy as np
import onnxruntime as ort

session = ort.InferenceSession('model.onnx')
input_name = session.get_inputs()[0].name

def test(img, scale=True):
    img = img.astype(np.float32)
    if scale:
        img = img / 255.0
    img = np.transpose(img, (2, 0, 1))
    img = np.expand_dims(img, axis=0)
    outputs = session.run(None, {input_name: img})
    return outputs[1][0]

zeros = np.zeros((224, 224, 3))
ones_255 = np.ones((224, 224, 3)) * 255
random_img = np.random.randint(0, 256, (224, 224, 3))

print("Zeros [0,1]:", test(zeros, scale=True))
print("Ones [0,1] (i.e. all ones):", test(np.ones((224,224,3)), scale=True))
print("Ones [0,255] (scaled):", test(ones_255, scale=True))
print("Ones [0,255] (unscaled):", test(ones_255, scale=False))
print("Random [0,1] (scaled):", test(random_img, scale=True))
print("Random [0,255] (unscaled):", test(random_img, scale=False))
