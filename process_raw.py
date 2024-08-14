import numpy as np
import cv2

output_data = np.fromfile('output0.raw', dtype=np.float32)

# Reshape the output according to expected dimensions
output_data = output_data.reshape(1, 25200, 85)

def non_max_suppression(boxes, scores, iou_threshold=0.5):
    x1 = boxes[:, 0]
    y1 = boxes[:, 1]
    x2 = boxes[:, 2]
    y2 = boxes[:, 3]

    areas = (x2 - x1) * (y2 - y1)
    order = scores.argsort()[::-1]

    keep = []
    while order.size > 0:
        i = order[0]
        keep.append(i)

        xx1 = np.maximum(x1[i], x1[order[1:]])
        yy1 = np.maximum(y1[i], y1[order[1:]])
        xx2 = np.minimum(x2[i], x2[order[1:]])
        yy2 = np.minimum(y2[i], y2[order[1:]])

        w = np.maximum(0, xx2 - xx1)
        h = np.maximum(0, yy2 - yy1)
        inter = w * h

        iou = inter / (areas[i] + areas[order[1:]] - inter)
        inds = np.where(iou <= iou_threshold)[0]
        order = order[inds + 1]

    return keep

def draw_detections_on_image(image, boxes, scores, class_ids, class_names):
    for i, (box, score, class_id) in enumerate(zip(boxes, scores, class_ids)):
        x1, y1, x2, y2 = box.astype(int)

        cv2.rectangle(image, (x1, y1), (x2, y2), color=(0, 255, 0), thickness=2)

        label = f"{class_names[class_id]}: {score:.2f}"

        # Calculate label position
        (label_width, label_height), baseline = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 2)
        top_left = (x1, y1 - label_height if y1 - label_height > 0 else y1 + label_height)
        bottom_right = (x1 + label_width, y1)

        cv2.rectangle(image, top_left, bottom_right, color=(0, 255, 0), thickness=cv2.FILLED)

        cv2.putText(image, label, (x1, y1 if y1 - label_height > 0 else y1 + label_height),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, color=(0, 0, 0), thickness=2)

def print_yolo_detections(output_data, image, threshold=0.5, iou_threshold=0.5, min_size=10):
    boxes = output_data[..., :4]
    objectness_scores = output_data[..., 4]
    class_probs = output_data[..., 5:]

    valid_detections = objectness_scores > threshold
    boxes = boxes[valid_detections]
    scores = objectness_scores[valid_detections]
    class_ids = class_probs[valid_detections].argmax(axis=-1)

    img_w, img_h = image.shape[1], image.shape[0]

    # Convert center coordinates to top-left and bottom-right corners
    boxes[:, 0:2] = boxes[:, 0:2] - boxes[:, 2:4] / 2
    boxes[:, 2:4] = boxes[:, 0:2] + boxes[:, 2:4]

    # Clip the coordinates to ensure they are within the image bounds
    boxes[:, [0, 2]] = np.clip(boxes[:, [0, 2]], 0, img_w)
    boxes[:, [1, 3]] = np.clip(boxes[:, [1, 3]], 0, img_h)

    # Apply Non-Maximum Suppression (NMS)
    nms_indices = non_max_suppression(boxes, scores, iou_threshold)
    boxes = boxes[nms_indices]
    scores = scores[nms_indices]
    class_ids = class_ids[nms_indices]

    # Filter out small or invalid boxes
    filtered_boxes = []
    filtered_scores = []
    filtered_class_ids = []

    for i, (box, score, class_id) in enumerate(zip(boxes, scores, class_ids)):
        width = box[2] - box[0]
        height = box[3] - box[1]
        if width >= min_size and height >= min_size:
            filtered_boxes.append(box)
            filtered_scores.append(score)
            filtered_class_ids.append(class_id)

    for i, (box, score, class_id) in enumerate(zip(filtered_boxes, filtered_scores, filtered_class_ids)):
        print(f"Detection {i}: Class={class_id}, Score={score:.2f}, Box={box}")

    draw_detections_on_image(image, np.array(filtered_boxes), np.array(filtered_scores), np.array(filtered_class_ids), class_names)

    cv2.imwrite('output_with_detections.jpg', image)  # Save the result image
    # cv2.imshow('Detections', image)  # Display the image
    # cv2.waitKey(0)  # Wait for a key press to close the window
    # cv2.destroyAllWindows()  # Close the display window

class_names = [
    'person', 'bicycle', 'car', 'motorcycle', 'airplane', 'bus', 'train', 'truck', 'boat', 
    'traffic light', 'fire hydrant', 'stop sign', 'parking meter', 'bench', 'bird', 'cat', 
    'dog', 'horse', 'sheep', 'cow', 'elephant', 'bear', 'zebra', 'giraffe', 'backpack', 
    'umbrella', 'handbag', 'tie', 'suitcase', 'frisbee', 'skis', 'snowboard', 'sports ball', 
    'kite', 'baseball bat', 'baseball glove', 'skateboard', 'surfboard', 'tennis racket', 
    'bottle', 'wine glass', 'cup', 'fork', 'knife', 'spoon', 'bowl', 'banana', 'apple', 
    'sandwich', 'orange', 'broccoli', 'carrot', 'hot dog', 'pizza', 'donut', 'cake', 
    'chair', 'couch', 'potted plant', 'bed', 'dining table', 'toilet', 'tv', 'laptop', 
    'mouse', 'remote', 'keyboard', 'cell phone', 'microwave', 'oven', 'toaster', 'sink', 
    'refrigerator', 'book', 'clock', 'vase', 'scissors', 'teddy bear', 'hair drier', 
    'toothbrush'
]

image = cv2.imread('bus.jpg')

image = cv2.resize(image, (640, 640))

# Process the output to draw detections with score > 0.5 after NMS
print_yolo_detections(output_data[0], image, min_size=10)
