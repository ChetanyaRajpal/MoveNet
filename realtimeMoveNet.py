# Game Plan
#Installing MoveNet
#Loading MoveNet using TFLite
#Real Time Rendering and Image Rendering
#Install and Import Dependencies
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import cv2
# from ai_edge_litert.interpreter import Interpreter

# Path to the SavedModel directory
# saved_model_dir = 'F:/Code/Python/MoveNet/movenet-tensorflow2-singlepose-lightning-v4'

# # Convert to TFLite format
# converter = tf.lite.TFLiteConverter.from_saved_model(saved_model_dir)
# tflite_model = converter.convert()

# # Save the .tflite model
# tflite_model_path = 'movenet_singlepose_lightning.tflite'
# with open(tflite_model_path, 'wb') as f:
#     f.write(tflite_model)

# print(f"TFLite model saved at {tflite_model_path}")

tflite_model_path = 'movenet_thunder.tflite'

# Load the TFLite model
interpreter = tf.lite.Interpreter(model_path=tflite_model_path)
interpreter.allocate_tensors()


print("TFLite model loaded successfully!")

def draw_keypoints(image, keypoints, confidence_threshold):
    y,x,c = frame.shape
    shaped = np.squeeze(np.multiply(keypoints, [y,x,1]))
    
    for kp in shaped:
        ky, kx, kp_conf = kp
        if kp_conf > confidence_threshold:
            cv2.circle(frame, (int(kx), int(ky)), 4, (0,255,0), -1)
            
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

def draw_connection(frame, keypoints, edges, confidence_threshold):
    y,x,c = frame.shape
    shaped = np.squeeze(np.multiply(keypoints, [y,x,1]))
    for edge, color in edges.items():
        p1,p2 = edge
        y1, x1, c1 = shaped[p1]
        y2, x2, c2 = shaped[p2]
        if (c1 > confidence_threshold) & (c2 > confidence_threshold):
            cv2.line(frame, (int(x1), int(y1)), (int(x2), int(y2)),(255,0,0), 2)

#Make Detections
cap = cv2.VideoCapture(0)
while cap.isOpened():
    ret, frame = cap.read()
    #Reshape Image as the model expects the image to be in the format of (192,192,3)
    img = frame.copy()
    img = tf.image.resize_with_pad(np.expand_dims(img, axis=0), 256,256)
    input_image = tf.cast(img, dtype = tf.float32)
    
    #Setup input and output
    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()

    #Make Predictions
    interpreter.set_tensor(input_details[0]['index'], np.array(input_image))
    interpreter.invoke()
    keypoints_with_scores = interpreter.get_tensor(output_details[0]['index'])
    print(keypoints_with_scores)
    
    #Rendering 
    draw_connection(frame, keypoints_with_scores, EDGES, 0.4)
    draw_keypoints(frame, keypoints_with_scores, 0.4)
    cv2.imshow('Movenet Lightning', frame)
    if cv2.waitKey(10) & 0xFF == ord('q'):
        break
cap.release()
cv2.destroyAllWindows()

