import cv2
import numpy as np
import onnxruntime as ort

session = ort.InferenceSession('model.onnx')
input_name = session.get_inputs()[0].name

def test(img):
    img = np.transpose(img, (2, 0, 1))
    img = np.expand_dims(img, axis=0).astype(np.float32)
    outputs = session.run(None, {input_name: img})
    return outputs[1][0]

zeros = np.zeros((224, 224, 3))
ones = np.ones((224, 224, 3))
random = np.random.rand(224, 224, 3)

print("Zeros:", test(zeros))
print("Ones:", test(ones))
print("Random:", test(random))
