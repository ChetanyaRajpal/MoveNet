import tensorflow as tf
import numpy as np
import cv2
import math

def calculate_angle(p1, p2, p3):
    """
    Calculate the angle between three points.
    Args:
    - p1, p2, p3: Points (y, x, confidence)
    Returns:
    - Angle in degrees
    """
    y1, x1, _ = p1
    y2, x2, _ = p2
    y3, x3, _ = p3

    # Vectors
    v1 = [x1 - x2, y1 - y2]
    v2 = [x3 - x2, y3 - y2]

    # Dot product and magnitudes
    dot_product = v1[0] * v2[0] + v1[1] * v2[1]
    magnitude1 = math.sqrt(v1[0]**2 + v1[1]**2)
    magnitude2 = math.sqrt(v2[0]**2 + v2[1]**2)

    if magnitude1 * magnitude2 == 0:
        return 0

    # Angle in radians and convert to degrees
    angle = math.acos(dot_product / (magnitude1 * magnitude2))
    return math.degrees(angle)

def check_posture(keypoints, confidence_threshold=0.4):
    """
    Check if the posture is either sitting at 90 degrees or lying at 45 degrees.
    Args:
    - keypoints: Detected keypoints
    - confidence_threshold: Minimum confidence for reliable keypoints
    Returns:
    - Boolean: True if posture is correct, False otherwise
    """
    shaped = np.squeeze(keypoints)

    # Extract keypoints
    left_shoulder = shaped[5]
    right_shoulder = shaped[6]
    left_hip = shaped[11]
    right_hip = shaped[12]

    # Ensure all required keypoints meet the confidence threshold
    keypoints = [left_shoulder, right_shoulder, left_hip, right_hip]
    if any(kp[2] < confidence_threshold for kp in keypoints):
        return False

    # Calculate angles
    torso_angle = calculate_angle(left_shoulder, left_hip, right_hip)
    lying_angle = calculate_angle(left_shoulder, left_hip, [left_hip[0], left_hip[1] + 1, left_hip[2]])

    # Conditions for correct posture
    is_sitting = 75 <= torso_angle <= 100
    is_lying = 40 <= lying_angle <= 50

    return is_sitting or is_lying

EDGES = {
    (0, 1): 'm',
    (0, 2): 'c',
    (1, 3): 'm',
    (2, 4): 'c',
    (0, 5): 'm',
    (0, 6): 'c',
    (5, 7): 'm',
    (7, 9): 'm',
    (6, 8): 'c',
    (8, 10): 'c',
    (5, 6): 'y',
    (5, 11): 'm',
    (6, 12): 'c',
    (11, 12): 'y',
    (11, 13): 'm',
    (13, 15): 'm',
    (12, 14): 'c',
    (14, 16): 'c'
}

def draw_keypoints(image, keypoints, confidence_threshold):
    y, x, c = image.shape
    shaped = np.squeeze(np.multiply(keypoints, [y, x, 1]))

    for kp in shaped:
        ky, kx, kp_conf = kp
        if kp_conf > confidence_threshold:
            cv2.circle(image, (int(kx), int(ky)), 4, (0, 255, 0), -1)

def draw_connection(frame, keypoints, edges, confidence_threshold):
    y, x, c = frame.shape
    shaped = np.squeeze(np.multiply(keypoints, [y, x, 1]))
    for edge, color in edges.items():
        p1, p2 = edge
        y1, x1, c1 = shaped[p1]
        y2, x2, c2 = shaped[p2]
        if (c1 > confidence_threshold) & (c2 > confidence_threshold):
            cv2.line(frame, (int(x1), int(y1)), (int(x2), int(y2)), (255, 0, 0), 2)

# Load TFLite model
tflite_model_path = 'movenet_singlepose_lightning.tflite'
interpreter = tf.lite.Interpreter(model_path=tflite_model_path)
interpreter.allocate_tensors()

# Video capture
cap = cv2.VideoCapture(0)
while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    # Preprocess frame
    img = frame.copy()
    img = tf.image.resize_with_pad(np.expand_dims(img, axis=0), 192, 192)
    input_image = tf.cast(img, dtype=tf.int32)

    # Setup input and output
    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()

    # Make predictions
    interpreter.set_tensor(input_details[0]['index'], np.array(input_image))
    interpreter.invoke()
    keypoints_with_scores = interpreter.get_tensor(output_details[0]['index'])

    # Check posture and set frame color
    if check_posture(keypoints_with_scores, 0.4):
        color = (0, 255, 0)  # Green for correct posture
    else:
        color = (0, 0, 255)  # Red for incorrect posture

    # Render keypoints and connections
    draw_connection(frame, keypoints_with_scores, EDGES, 0.4)
    draw_keypoints(frame, keypoints_with_scores, 0.4)

    # Overlay frame color
    overlay = frame.copy()
    cv2.rectangle(overlay, (0, 0), (frame.shape[1], frame.shape[0]), color, thickness=-1)
    alpha = 0.2  # Transparency factor
    frame = cv2.addWeighted(overlay, alpha, frame, 1 - alpha, 0)

    # Display frame
    cv2.imshow('Movenet Posture Detection', frame)
    if cv2.waitKey(10) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
