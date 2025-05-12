import cv2
import numpy as np
import os
import sys
import warnings
import logging
import argparse
from ultralytics import YOLO
from collections import deque
import signal

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

os.environ["NNPACK_ENABLED"] = "0"
os.environ["PYTORCH_NO_NNPACK"] = "1"
os.environ["QT_LOGGING_RULES"] = "qt5ct.debug=false"
warnings.filterwarnings("ignore", category=UserWarning, module="torch")

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from scripts.stereo_processing import StereoProcessor
from scripts.stereoconfig import stereoCamera

class KalmanFilter:
    def __init__(self):
        self.kf = cv2.KalmanFilter(6, 3)
        self.kf.transitionMatrix = np.array([
            [1, 0, 0, 1, 0, 0],
            [0, 1, 0, 0, 1, 0],
            [0, 0, 1, 0, 0, 1],
            [0, 0, 0, 1, 0, 0],
            [0, 0, 0, 0, 1, 0],
            [0, 0, 0, 0, 0, 1]
        ], dtype=np.float32)
        self.kf.measurementMatrix = np.array([
            [1, 0, 0, 0, 0, 0],
            [0, 1, 0, 0, 0, 0],
            [0, 0, 1, 0, 0, 0]
        ], dtype=np.float32)
        self.kf.processNoiseCov = np.eye(6, dtype=np.float32) * 0.01
        self.kf.measurementNoiseCov = np.eye(3, dtype=np.float32) * 1.0
        self.kf.errorCovPost = np.eye(6, dtype=np.float32)
        self.kf.statePost = np.zeros((6, 1), dtype=np.float32)

    def predict(self):
        return self.kf.predict()

    def correct(self, measurement):
        return self.kf.correct(measurement)

    @property
    def statePost(self):
        return self.kf.statePost

def enhance_image(image):
    lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    l = clahe.apply(l)
    lab = cv2.merge((l, a, b))
    enhanced = cv2.cvtColor(lab, cv2.COLOR_LAB2BGR)
    kernel = np.array([[0, -1, 0], [-1, 5, -1], [0, -1, 0]])
    sharpened = cv2.filter2D(enhanced, -1, kernel)
    return sharpened

def enhance_color_regions(image):
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    red_lower1 = np.array([0, 50, 50])
    red_upper1 = np.array([10, 255, 255])
    red_lower2 = np.array([170, 50, 50])
    red_upper2 = np.array([180, 255, 255])
    red_mask1 = cv2.inRange(hsv, red_lower1, red_upper1)
    red_mask2 = cv2.inRange(hsv, red_lower2, red_upper2)
    red_mask = red_mask1 | red_mask2
    blue_lower = np.array([100, 50, 50])
    blue_upper = np.array([140, 255, 255])
    blue_mask = cv2.inRange(hsv, blue_lower, blue_upper)
    combined_mask = red_mask | blue_mask
    enhanced = image.copy()
    enhanced[combined_mask > 0] = cv2.addWeighted(
        enhanced[combined_mask > 0], 0.7, enhanced[combined_mask > 0], 0.3, 1.5
    )
    return enhanced

def refine_bbox_center(image, bbox):
    x1, y1, x2, y2 = map(int, bbox)
    center_x = (x1 + x2) / 2
    center_y = (y1 + y2) / 2
    roi = image[max(0, y1-10):min(image.shape[0], y2+10), max(0, x1-10):min(image.shape[1], x2+10)]
    if roi.size == 0:
        return center_x, center_y
    gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
    corners = cv2.goodFeaturesToTrack(gray, maxCorners=1, qualityLevel=0.01, minDistance=10)
    if corners is not None:
        x, y = corners[0].ravel()
        return x + max(0, x1-10), y + max(0, y1-10)
    return center_x, center_y

def compute_iou(box1, box2):
    x1, y1, x2, y2 = box1
    x1_p, y1_p, x2_p, y2_p = box2
    xi1 = max(x1, x1_p)
    yi1 = max(y1, y1_p)
    xi2 = min(x2, x2_p)
    yi2 = min(y2, y2_p)
    inter_area = max(0, xi2 - xi1) * max(0, yi2 - yi1)
    box1_area = (x2 - x1) * (y2 - y1)
    box2_area = (x2_p - x1_p) * (y2_p - y1_p)
    union_area = box1_area + box2_area - inter_area
    return inter_area / union_area if union_area > 0 else 0

def resolve_overlaps(detections, height, width):
    adjusted_detections = []
    occupied_rects = []  # Tracks label rectangles
    bbox_rects = [(int(d['bbox'][0]), int(d['bbox'][1]), int(d['bbox'][2]), int(d['bbox'][3])) for d in detections if d['depth'] is not None]

    for det in detections:
        if det['depth'] is None:
            continue

        bbox = det['bbox']
        class_name = det['class_name']
        depth = det['depth']
        x1, y1, x2, y2 = map(int, bbox)
        font_scale = 0.6
        font_thickness = 2

        label_text = f"{class_name}: {int(depth)} mm"
        text_size, baseline = cv2.getTextSize(label_text, cv2.FONT_HERSHEY_SIMPLEX, font_scale, font_thickness)

        # Try multiple positions: top, bottom, left, right, diagonals
        position_attempts = [
            # (x_offset, y_offset, priority)
            (0, -15 - text_size[1], 1),  # Above
            (0, y2 - y1 + text_size[1] + 15, 2),  # Below
            (-text_size[0] - 10, (y2 - y1) // 2, 3),  # Left
            (x2 - x1 + 10, (y2 - y1) // 2, 4),  # Right
            (-text_size[0] - 10, -15 - text_size[1], 5),  # Top-left
            (x2 - x1 + 10, -15 - text_size[1], 6),  # Top-right
            (-text_size[0] - 10, y2 - y1 + text_size[1] + 15, 7),  # Bottom-left
            (x2 - x1 + 10, y2 - y1 + text_size[1] + 15, 8),  # Bottom-right
        ]

        best_position = None
        best_priority = float('inf')

        for x_offset, y_offset, priority in position_attempts:
            text_x = int(x1 + (x2 - x1 - text_size[0]) / 2 + x_offset)
            text_y = y1 + y_offset
            label_rect = [text_x, text_y - text_size[1] - 5, text_x + text_size[0] + 5, text_y + baseline + 5]

            # Check if label is within image bounds
            if not (0 <= label_rect[0] <= width and 0 <= label_rect[2] <= width and
                    0 <= label_rect[1] <= height and 0 <= label_rect[3] <= height):
                continue

            # Check for overlap with other labels and bounding boxes
            overlap = False
            for occ_rect in occupied_rects + bbox_rects:
                if compute_iou(label_rect, occ_rect) > 0.05:  # Stricter IoU threshold
                    overlap = True
                    break

            if not overlap and priority < best_priority:
                best_position = (text_x, text_y, label_rect)
                best_priority = priority

        if best_position is None:
            logger.debug(f"Skipping drawing for {class_name} at bbox {bbox} due to unresolvable overlap")
            continue

        text_x, text_y, label_rect = best_position
        det['text_x'] = text_x
        det['text_y'] = text_y
        det['label_rect'] = label_rect
        adjusted_detections.append(det)
        occupied_rects.append(label_rect)

    return adjusted_detections

def process_frame(frame, stereo_processor, model, width, height, frame_width, frame_height, use_tracking=False, depth_history=None, detection_history=None, kalman_filters=None, object_id=0):
    frame_height, frame_width = frame.shape[:2]
    left_img = frame[:, :frame_width//2]
    right_img = frame[:, frame_width//2:]

    left_img = cv2.resize(left_img, (width, height))
    right_img = cv2.resize(right_img, (width, height))
    left_img_enhanced = enhance_image(left_img)
    right_img_enhanced = enhance_image(right_img)
    left_img_enhanced = enhance_color_regions(left_img_enhanced)
    right_img_enhanced = enhance_color_regions(right_img_enhanced)

    try:
        # Predict with lower confidence for better recall
        results = model.predict(left_img_enhanced, conf=0.05, iou=0.6, classes=[0, 1, 2])
        result = results[0]
        # Log all detection confidences for debugging
        for box in result.boxes:
            cls_id = int(box.cls)
            class_name = {0: 'branch', 1: 'burl', 2: 'intersection'}.get(cls_id, 'unknown')
            conf = box.conf.cpu().numpy()[0]
            logger.debug(f"Raw detection: {class_name}, Confidence: {conf:.4f}")
    except Exception as e:
        logger.error(f"YOLOv8 prediction error: {str(e)}")
        return left_img, [], object_id

    # Validate class IDs
    valid_classes = {0: 'branch', 1: 'burl', 2: 'intersection'}
    for box in result.boxes:
        cls_id = int(box.cls)
        if cls_id not in valid_classes:
            logger.warning(f"Invalid class ID {cls_id} detected, expected {list(valid_classes.keys())}")
            return left_img, [], object_id

    max_retries = 3
    for attempt in range(max_retries):
        try:
            left_gray, right_gray = stereo_processor.preprocess(left_img, right_img)
            left_rect, right_rect = stereo_processor.rectifyImage(left_gray, right_gray)
            disparity = stereo_processor.stereoMatchSGBM(left_rect, right_rect)
            break
        except Exception as e:
            logger.warning(f"Stereo processing attempt {attempt + 1} failed: {str(e)}")
            if attempt == max_retries - 1:
                logger.error("Stereo processing failed after max retries")
                return left_img, [], object_id

    points_3d = stereo_processor.compute_3d_points(disparity)

    detections = []
    for box in result.boxes:
        cls_id = int(box.cls)
        class_name = valid_classes[cls_id]
        bbox = box.xyxy[0].cpu().numpy()
        conf = box.conf.cpu().numpy()[0]

        center_x, center_y = refine_bbox_center(left_img, bbox)
        depth = stereo_processor.get_depth_for_bbox(points_3d, bbox)
        obj_key = f"{class_name}_{object_id}"
        smoothed_depth = depth

        if depth is not None and 200 < depth < 4000:
            if use_tracking:
                if depth_history is None or detection_history is None or kalman_filters is None:
                    logger.error("Tracking parameters not provided for video processing")
                    return left_img, [], object_id

                if obj_key not in depth_history:
                    depth_history[obj_key] = deque(maxlen=10)
                    kalman_filters[obj_key] = KalmanFilter()
                    object_id += 1
                depth_history[obj_key].append(depth)
                depths = list(depth_history[obj_key])
                smoothed_depth = np.median(depths) if depths else depth

                kf = kalman_filters[obj_key]
                kf.predict()
                measurement = np.array([[center_x], [center_y], [smoothed_depth]], dtype=np.float32)
                kf.correct(measurement)
                state = kf.statePost
                center_x, center_y, smoothed_depth = state[0, 0], state[1, 0], state[2, 0]
            else:
                object_id += 1

            detections.append({
                'class_name': class_name,
                'confidence': conf,
                'depth': smoothed_depth,
                'bbox': bbox,
                'center_x': center_x,
                'center_y': center_y,
                'obj_id': obj_key
            })
            logger.info(f"Object: {class_name} (ID: {obj_key}), Confidence: {conf:.2f}, Depth: {smoothed_depth:.2f} mm, Center: ({center_x:.2f}, {center_y:.2f}), Bbox: {bbox}, Color: {'Blue' if class_name == 'branch' else 'Red' if class_name == 'burl' else 'Green'}")
        else:
            logger.warning(f"No valid depth for {class_name} at bbox {bbox}")
            object_id += 1

    if use_tracking and detection_history is not None:
        detection_history.append(detections)
        if len(detection_history) > 1:
            prev_detections = detection_history[-2]
            for det in detections:
                for prev_det in prev_detections:
                    if det['class_name'] == prev_det['class_name']:
                        iou = compute_iou(det['bbox'], prev_det['bbox'])
                        if iou > 0.7:
                            det['confidence'] = max(det['confidence'], prev_det['confidence'] * 0.95)

    detections = sorted(detections, key=lambda x: x['confidence'], reverse=True)
    detections = resolve_overlaps(detections, height, width)

    for det in detections:
        bbox = det['bbox']
        class_name = det['class_name']
        depth = det['depth']
        text_x = det['text_x']
        text_y = det['text_y']
        center_x = det['center_x']
        center_y = det['center_y']
        color = (255, 0, 0) if class_name == 'branch' else (0, 0, 255) if class_name == 'burl' else (0, 255, 0)
        x1, y1, x2, y2 = map(int, bbox)

        # Draw bounding box around the object
        cv2.rectangle(left_img, (x1, y1), (x2, y2), color, 1)

        # Draw label with background
        label_text = f"{class_name}: {int(depth)} mm"
        font_scale = 0.6
        font_thickness = 2
        text_size, baseline = cv2.getTextSize(label_text, cv2.FONT_HERSHEY_SIMPLEX, font_scale, font_thickness)
        cv2.rectangle(left_img, (text_x - 5, text_y - text_size[1] - 5),
                      (text_x + text_size[0] + 5, text_y + baseline + 5), (0, 0, 0), -1)
        cv2.putText(left_img, label_text, (text_x, text_y), cv2.FONT_HERSHEY_SIMPLEX,
                    font_scale, color, font_thickness)

        # Draw line from label to object center
        line_x = text_x + text_size[0] // 2
        line_y_start = text_y + baseline - 2 if text_y < y1 else text_y - text_size[1]
        line_y_end = int(center_y)
        line_x_end = int(center_x)
        cv2.line(left_img, (line_x, line_y_start), (line_x_end, line_y_end), color, 1)

    # Draw resolution text
    resolution_text = f"Resolution: {width}x{height}"
    font_scale = 0.6
    font_thickness = 2
    text_size, baseline = cv2.getTextSize(resolution_text, cv2.FONT_HERSHEY_SIMPLEX, font_scale, font_thickness)
    cv2.rectangle(left_img, (10, 40 - text_size[1] - 5),
                  (10 + text_size[0] + 10, 40 + baseline + 5), (0, 0, 0), -1)
    cv2.putText(left_img, resolution_text, (10, 40), cv2.FONT_HERSHEY_SIMPLEX,
                font_scale, (255, 255, 255), font_thickness)

    return left_img, detections, object_id

def cleanup(cap=None, out=None):
    """Release OpenCV resources and close windows."""
    if cap and cap.isOpened():
        cap.release()
    if out:
        out.release()
    cv2.destroyAllWindows()
    logger.info("Cleaned up OpenCV resources.")

def signal_handler(sig, frame):
    """Handle Ctrl+C gracefully."""
    logger.info("Received Ctrl+C, shutting down...")
    cleanup()
    sys.exit(0)

def main():
    # Set up signal handler for Ctrl+C
    signal.signal(signal.SIGINT, signal_handler)

    parser = argparse.ArgumentParser(description="Process stereo video or image for object detection and depth estimation.")
    parser.add_argument('--input', type=str, required=True, help='Path to input video or image')
    parser.add_argument('--model', type=str, default='superior5', choices=['baseline4', 'superior5'], help='Model to use: baseline4, superior5')
    args = parser.parse_args()

    input_path = args.input
    is_video = input_path.lower().endswith(('.avi', '.mp4', '.mov'))
    is_image = input_path.lower().endswith(('.jpg', '.jpeg', '.png'))

    if not (is_video or is_image):
        logger.error("Input must be a video (.avi, .mp4, .mov) or image (.jpg, .jpeg, .png)")
        cleanup()
        return

    if not os.path.exists(input_path):
        logger.error(f"Input file {input_path} does not exist.")
        cleanup()
        return

    model_path = os.path.join(os.path.dirname(__file__), f'../runs/detect/train_{args.model}/weights/best.pt')
    if not os.path.exists(model_path):
        logger.error(f"Model file {model_path} not found.")
        cleanup()
        return
    try:
        model = YOLO(model_path)
    except Exception as e:
        logger.error(f"Failed to load YOLO model: {str(e)}")
        cleanup()
        return

    config = stereoCamera()
    width, height = 640, 480
    try:
        stereo_processor = StereoProcessor(config, width, height)
    except Exception as e:
        logger.error(f"Error initializing StereoProcessor: {str(e)}")
        cleanup()
        return

    cv2.namedWindow('YOLOv8 + Stereo', cv2.WND_PROP_AUTOSIZE)
    cv2.resizeWindow('YOLOv8 + Stereo', 1280, 720)

    if is_video:
        cap = cv2.VideoCapture(input_path)
        if not cap.isOpened():
            logger.error(f"Could not open video {input_path}. Ensure the video format is supported by OpenCV (e.g., install FFmpeg with 'sudo apt-get install ffmpeg').")
            cleanup()
            return

        frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        output_path = os.path.join(os.path.dirname(__file__), '../output/processed_stereo.avi')
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        fourcc = cv2.VideoWriter_fourcc(*'XVID')
        out = cv2.VideoWriter(output_path, fourcc, 30.0, (width, height))

        depth_history = {}
        detection_history = deque(maxlen=3)
        kalman_filters = {}
        object_id = 0
        frame_count = 0

        try:
            while cap.isOpened():
                ret, frame = cap.read()
                if not ret:
                    logger.info("End of video or error reading frame.")
                    break

                left_img, detections, object_id = process_frame(
                    frame, stereo_processor, model, width, height, frame_width, frame_height,
                    use_tracking=True, depth_history=depth_history,
                    detection_history=detection_history, kalman_filters=kalman_filters,
                    object_id=object_id
                )

                valid_detections = [d for d in detections if d['depth'] is not None]
                logger.info(f"Frame {frame_count}: Detected {len(valid_detections)} objects: {[d['class_name'] for d in valid_detections]}")
                if valid_detections:
                    depths = [d['depth'] for d in valid_detections]
                    logger.info(f"Depth stats: Min={min(depths):.2f}mm, Max={max(depths):.2f}mm, Mean={np.mean(depths):.2f}mm")

                out.write(left_img)
                cv2.imshow('YOLOv8 + Stereo', left_img)
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    logger.info("User requested exit.")
                    break

                frame_count += 1
                if frame_count % 10 == 0:
                    logger.info(f"Processed {frame_count} frames")
        except KeyboardInterrupt:
            logger.info("Video processing interrupted by user.")
        finally:
            cleanup(cap, out)

    else:
        frame = cv2.imread(input_path)
        if frame is None:
            logger.error(f"Could not read image {input_path}.")
            cleanup()
            return

        frame_height, frame_width = frame.shape[:2]
        left_img, detections, _ = process_frame(
            frame, stereo_processor, model, width, height, frame_width, frame_height,
            use_tracking=False
        )

        valid_detections = [d for d in detections if d['depth'] is not None]
        logger.info(f"Image processed: Detected {len(valid_detections)} objects: {[d['class_name'] for d in valid_detections]}")
        if valid_detections:
            depths = [d['depth'] for d in valid_detections]
            logger.info(f"Depth stats: Min={min(depths):.2f}mm, Max={max(depths):.2f}mm, Mean={np.mean(depths):.2f}mm")

        output_dir = os.path.join(os.path.dirname(__file__), '../output')
        os.makedirs(output_dir, exist_ok=True)
        input_filename = os.path.splitext(os.path.basename(input_path))[0]
        output_path = os.path.join(output_dir, f"processed_{input_filename}.jpg")
        cv2.imwrite(output_path, left_img)
        logger.info(f"Saved processed image to {output_path}")

        try:
            cv2.imshow('YOLOv8 + Stereo', left_img)
            while True:
                key = cv2.waitKey(50) & 0xFF
                if key == ord('q') or cv2.getWindowProperty('YOLOv8 + Stereo', cv2.WND_PROP_VISIBLE) < 1:
                    break
        except KeyboardInterrupt:
            logger.info("Image display interrupted by user.")
        finally:
            cleanup()

if __name__ == "__main__":
    main()