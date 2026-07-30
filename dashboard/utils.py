import numpy as np
import math
import onnxruntime
import onnx
import tempfile
import os
from PIL import Image


# ──────────────────────────────────────────────────────────────────────────────
# Custom Vision YOLO Object Detection (adapted from Microsoft's reference code)
# ──────────────────────────────────────────────────────────────────────────────

class ObjectDetection:
    """Microsoft Custom Vision YOLO post-processing logic."""

    ANCHORS = np.array([[0.573, 0.677], [1.87, 2.06], [3.34, 5.47], [7.88, 3.53], [9.77, 9.17]])
    IOU_THRESHOLD = 0.45
    DEFAULT_INPUT_SIZE = 512 * 512

    def __init__(self, labels, prob_threshold=0.50, max_detections=50):
        assert len(labels) >= 1
        self.labels = labels
        self.prob_threshold = prob_threshold
        self.max_detections = max_detections

    def _logistic(self, x):
        return np.where(x > 0, 1 / (1 + np.exp(-x)), np.exp(x) / (1 + np.exp(x)))

    def _non_maximum_suppression(self, boxes, class_probs, max_detections):
        assert len(boxes) == len(class_probs)
        max_detections = min(max_detections, len(boxes))
        max_probs = np.amax(class_probs, axis=1)
        max_classes = np.argmax(class_probs, axis=1)
        areas = boxes[:, 2] * boxes[:, 3]
        selected_boxes, selected_classes, selected_probs = [], [], []

        while len(selected_boxes) < max_detections:
            i = np.argmax(max_probs)
            if max_probs[i] < self.prob_threshold:
                break
            selected_boxes.append(boxes[i])
            selected_classes.append(max_classes[i])
            selected_probs.append(max_probs[i])
            box = boxes[i]
            other_indices = np.concatenate((np.arange(i), np.arange(i + 1, len(boxes))))
            other_boxes = boxes[other_indices]
            x1 = np.maximum(box[0], other_boxes[:, 0])
            y1 = np.maximum(box[1], other_boxes[:, 1])
            x2 = np.minimum(box[0] + box[2], other_boxes[:, 0] + other_boxes[:, 2])
            y2 = np.minimum(box[1] + box[3], other_boxes[:, 1] + other_boxes[:, 3])
            w = np.maximum(0, x2 - x1)
            h = np.maximum(0, y2 - y1)
            overlap_area = w * h
            iou = overlap_area / (areas[i] + areas[other_indices] - overlap_area)
            overlapping_indices = other_indices[np.where(iou > self.IOU_THRESHOLD)[0]]
            overlapping_indices = np.append(overlapping_indices, i)
            class_probs[overlapping_indices, max_classes[i]] = 0
            max_probs[overlapping_indices] = np.amax(class_probs[overlapping_indices], axis=1)
            max_classes[overlapping_indices] = np.argmax(class_probs[overlapping_indices], axis=1)

        return selected_boxes, selected_classes, selected_probs

    def _extract_bb(self, prediction_output, anchors):
        assert len(prediction_output.shape) == 3
        num_anchor = anchors.shape[0]
        height, width, channels = prediction_output.shape
        assert channels % num_anchor == 0
        num_class = int(channels / num_anchor) - 5
        assert num_class == len(self.labels)
        outputs = prediction_output.reshape((height, width, num_anchor, -1))
        x = (self._logistic(outputs[..., 0]) + np.arange(width)[np.newaxis, :, np.newaxis]) / width
        y = (self._logistic(outputs[..., 1]) + np.arange(height)[:, np.newaxis, np.newaxis]) / height
        w = np.exp(outputs[..., 2]) * anchors[:, 0][np.newaxis, np.newaxis, :] / width
        h = np.exp(outputs[..., 3]) * anchors[:, 1][np.newaxis, np.newaxis, :] / height
        x = x - w / 2
        y = y - h / 2
        boxes = np.stack((x, y, w, h), axis=-1).reshape(-1, 4)
        objectness = self._logistic(outputs[..., 4])
        class_probs = outputs[..., 5:]
        class_probs = np.exp(class_probs - np.amax(class_probs, axis=3)[..., np.newaxis])
        class_probs = class_probs / np.sum(class_probs, axis=3)[..., np.newaxis] * objectness[..., np.newaxis]
        class_probs = class_probs.reshape(-1, num_class)
        return boxes, class_probs

    def preprocess(self, image):
        """Resize image to model input size (maintaining aspect ratio, aligned to 32)."""
        image = image.convert("RGB") if image.mode != "RGB" else image
        ratio = math.sqrt(self.DEFAULT_INPUT_SIZE / image.width / image.height)
        new_width = 32 * math.ceil(int(image.width * ratio) / 32)
        new_height = 32 * math.ceil(int(image.height * ratio) / 32)
        return image.resize((new_width, new_height))

    def postprocess(self, prediction_outputs):
        """Convert raw ONNX outputs to a clean list of detections."""
        boxes, class_probs = self._extract_bb(prediction_outputs, self.ANCHORS)
        max_probs = np.amax(class_probs, axis=1)
        index, = np.where(max_probs > self.prob_threshold)
        index = index[(-max_probs[index]).argsort()]
        selected_boxes, selected_classes, selected_probs = self._non_maximum_suppression(
            boxes[index], class_probs[index], self.max_detections
        )
        return [
            {
                'probability': round(float(selected_probs[i]), 4),
                'tagName': self.labels[selected_classes[i]],
                'boundingBox': {
                    'left': round(float(selected_boxes[i][0]), 4),
                    'top': round(float(selected_boxes[i][1]), 4),
                    'width': round(float(selected_boxes[i][2]), 4),
                    'height': round(float(selected_boxes[i][3]), 4),
                }
            }
            for i in range(len(selected_boxes))
        ]


# ──────────────────────────────────────────────────────────────────────────────
# ONNX Runtime wrapper
# ──────────────────────────────────────────────────────────────────────────────

class SmartClassroomModel(ObjectDetection):
    """Loads and runs the Custom Vision ONNX Object Detection model."""

    def __init__(self, model_path, labels_path):
        with open(labels_path, 'r') as f:
            labels = [l.strip() for l in f.readlines() if l.strip()]

        super().__init__(labels)

        # Patch dynamic dims so ONNX Runtime accepts variable input sizes
        model = onnx.load(model_path)
        with tempfile.TemporaryDirectory() as dirpath:
            temp = os.path.join(dirpath, "model_patched.onnx")
            model.graph.input[0].type.tensor_type.shape.dim[-1].dim_param = 'dim1'
            model.graph.input[0].type.tensor_type.shape.dim[-2].dim_param = 'dim2'
            onnx.save(model, temp)
            self.session = onnxruntime.InferenceSession(temp)

        self.input_name = self.session.get_inputs()[0].name
        self.is_fp16 = self.session.get_inputs()[0].type == 'tensor(float16)'

    def _run_inference(self, preprocessed_image):
        """Run ONNX inference and return raw output array."""
        inputs = np.array(preprocessed_image, dtype=np.float32)[np.newaxis, :, :, (2, 1, 0)]  # RGB -> BGR
        inputs = np.ascontiguousarray(np.rollaxis(inputs, 3, 1))
        if self.is_fp16:
            inputs = inputs.astype(np.float16)
        outputs = self.session.run(None, {self.input_name: inputs})
        return np.squeeze(outputs).transpose((1, 2, 0)).astype(np.float32)

    def _cross_class_nms(self, detections, iou_threshold=0.45):
        """
        Remove duplicate detections where two different-class boxes
        (e.g. 'student' + 'janitor') overlap the same physical person.
        Keeps the detection with higher confidence.
        """
        if len(detections) <= 1:
            return detections

        # Sort by probability descending
        detections = sorted(detections, key=lambda d: d['probability'], reverse=True)
        keep = []

        for i, det in enumerate(detections):
            bb_i = det['boundingBox']
            suppressed = False
            for kept_det in keep:
                bb_j = kept_det['boundingBox']
                # Compute intersection
                x1 = max(bb_i['left'], bb_j['left'])
                y1 = max(bb_i['top'], bb_j['top'])
                x2 = min(bb_i['left'] + bb_i['width'], bb_j['left'] + bb_j['width'])
                y2 = min(bb_i['top'] + bb_i['height'], bb_j['top'] + bb_j['height'])
                inter_w = max(0.0, x2 - x1)
                inter_h = max(0.0, y2 - y1)
                inter_area = inter_w * inter_h
                area_i = bb_i['width'] * bb_i['height']
                area_j = bb_j['width'] * bb_j['height']
                union_area = area_i + area_j - inter_area
                iou = inter_area / union_area if union_area > 0 else 0.0
                if iou > iou_threshold:
                    suppressed = True
                    break
            if not suppressed:
                keep.append(det)

        return keep

    def predict(self, frame_bgr, prob_threshold=None):
        """
        Run object detection on a BGR OpenCV frame.
        Applies cross-class NMS so the same person is never counted
        as both a student AND a janitor simultaneously.

        Args:
            frame_bgr: BGR OpenCV frame
            prob_threshold: optional override for detection confidence (0.0-1.0)

        Returns:
            detections: list of dicts with 'tagName', 'probability', 'boundingBox'
            student_count: int
            janitor_count: int
        """
        # Apply runtime threshold override if supplied
        if prob_threshold is not None:
            original_threshold = self.prob_threshold
            self.prob_threshold = prob_threshold
        from PIL import Image as PILImage
        import cv2
        # Convert BGR -> RGB PIL image
        frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
        pil_image = PILImage.fromarray(frame_rgb)

        preprocessed = self.preprocess(pil_image)
        raw_output = self._run_inference(preprocessed)
        detections = self.postprocess(raw_output)

        # Restore original threshold
        if prob_threshold is not None:
            self.prob_threshold = original_threshold

        # Remove overlapping cross-class duplicates (same person labeled as both student & janitor)
        detections = self._cross_class_nms(detections, iou_threshold=0.45)

        student_count = sum(1 for d in detections if d['tagName'].lower() == 'student')
        janitor_count = sum(1 for d in detections if d['tagName'].lower() in ('jeniter', 'janitor'))

        return detections, student_count, janitor_count


# ──────────────────────────────────────────────────────────────────────────────
# Occupancy & AC logic
# ──────────────────────────────────────────────────────────────────────────────

def get_occupancy_level(student_count, janitor_count):
    """
    Determines occupancy level based on detected people counts.

    Rules:
      - If there are ONLY janitors (0 students, >=1 janitor) → LOW (janitor mode)
      - If student_count < 3                                  → LOW
      - If 3 <= student_count <= 9                            → MEDIUM
      - If student_count > 9                                  → HIGH

    Returns:
        level (str): 'low', 'medium', or 'high'
        is_janitor_only (bool): True if the room only has janitors
    """
    total = student_count + janitor_count
    if total == 0:
        return 'low', False
    if student_count == 0 and janitor_count > 0:
        return 'low', True         # Janitor-only mode
    if student_count < 3:
        return 'low', False
    elif student_count <= 9:
        return 'medium', False
    else:
        return 'high', False


def get_ac_state(occupancy_level, is_janitor_only, medium_temp, high_temp):
    """
    Returns (ac_state, temperature_str) based on occupancy level.

    Janitor-only or LOW → AC OFF
    MEDIUM              → AC ON at medium_temp
    HIGH                → AC ON at high_temp
    """
    if is_janitor_only or occupancy_level == 'low':
        return 'OFF', '—'
    elif occupancy_level == 'medium':
        return 'ON', f'{medium_temp}°C'
    else:  # high
        return 'ON', f'{high_temp}°C'
