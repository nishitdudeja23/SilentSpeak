import numpy as np
import tensorflow as tf
import cv2
import mediapipe as mp
import json

# Define left and right pose landmarks
LPOSE = [13, 15, 17, 19, 21]  # Indices for left pose landmarks
RPOSE = [14, 16, 18, 20, 22]  # Indices for right pose landmarks

# Combine them into a single list for pose landmarks
POSE = LPOSE + RPOSE

X = [f'x_right_hand_{i}' for i in range(21)] + [f'x_left_hand_{i}' for i in range(21)] + [f'x_pose_{i}' for i in POSE]
Y = [f'y_right_hand_{i}' for i in range(21)] + [f'y_left_hand_{i}' for i in range(21)] + [f'y_pose_{i}' for i in POSE]
Z = [f'z_right_hand_{i}' for i in range(21)] + [f'z_left_hand_{i}' for i in range(21)] + [f'z_pose_{i}' for i in POSE]
FEATURE_COLUMNS = X + Y + Z

# Define indices for X, Y, Z, and other categories
X_IDX = [i for i, col in enumerate(FEATURE_COLUMNS) if "x_" in col]
Y_IDX = [i for i, col in enumerate(FEATURE_COLUMNS) if "y_" in col]
Z_IDX = [i for i, col in enumerate(FEATURE_COLUMNS) if "z_" in col]

RHAND_IDX = [i for i, col in enumerate(FEATURE_COLUMNS) if "right" in col]
LHAND_IDX = [i for i, col in enumerate(FEATURE_COLUMNS) if "left" in col]
RPOSE_IDX = [i for i, col in enumerate(FEATURE_COLUMNS) if "pose" in col and int(col[-2:]) in RPOSE]
LPOSE_IDX = [i for i, col in enumerate(FEATURE_COLUMNS) if "pose" in col and int(col[-2:]) in LPOSE]

# Load the character-to-prediction index mapping
with open(r"C:\Users\nishi\Videos\MS-ASL\character_to_prediction_index.json", "r") as f:
    char_to_num = json.load(f)

# Add special tokens
pad_token = 'P'
start_token = '<'
end_token = '>'
pad_token_idx = 59
start_token_idx = 60
end_token_idx = 61

char_to_num[pad_token] = pad_token_idx
char_to_num[start_token] = start_token_idx
char_to_num[end_token] = end_token_idx
num_to_char = {j: i for i, j in char_to_num.items()}

FRAME_LEN = 128

mp_hands, mp_pose = mp.solutions.hands, mp.solutions.pose
hands = mp_hands.Hands(static_image_mode=False, max_num_hands=1, min_detection_confidence=0.5)
pose = mp_pose.Pose(static_image_mode=False, min_detection_confidence=0.5, min_tracking_confidence=0.5)

mp_drawing = mp.solutions.drawing_utils

# Define preprocessing functions
def resize_pad(x):
    # Ensure x is a TensorFlow tensor
    x = tf.convert_to_tensor(x, dtype=tf.float32)
    
    # Get the current shape of x
    current_shape = tf.shape(x)
    
    # Determine the number of frames and feature dimensions
    current_frames = current_shape[0]
    num_features = current_shape[1]
    num_channels = current_shape[2] if len(current_shape) > 2 else 1
    
    # Define target shape
    target_frames = FRAME_LEN
    
    if current_frames < target_frames:
        # Calculate padding amount
        pad_amount = target_frames - current_frames
        
        # Define padding for the frames dimension
        paddings = tf.stack([[0, pad_amount], [0, 0]])
        
        if len(current_shape) == 3:
            paddings = tf.concat([paddings, [[0, 0]]], axis=0)  # Add padding for channels if necessary
        
        # Apply padding
        x = tf.pad(x, paddings, constant_values=0)
    else:
        # Resize to the target number of frames
        # Use tf.image.resize for tensors with at least 3 dimensions
        if len(current_shape) == 3:
            x = tf.image.resize(x, size=[target_frames, num_features], method='bilinear')
        else:
            # If tensor has 2 dimensions, just truncate to target_frames
            x = x[:target_frames, :]
    
    return x



def pre_process(x):
    x = np.array(x, dtype=np.float32)  # Ensure x is a float32 array

    if x.ndim == 1:
        x = x.reshape(-1, len(FEATURE_COLUMNS))  # Reshape to (num_samples, num_features)

    # Convert to TensorFlow tensor
    x = tf.convert_to_tensor(x, dtype=tf.float32)

    # Check number of features
    if x.shape[1] < len(FEATURE_COLUMNS):
        padding = [[0, 0], [0, len(FEATURE_COLUMNS) - x.shape[1]]]
        x = tf.pad(x, padding, constant_values=0)

    # Handle the case where there are fewer frames than needed
    x = resize_pad(x)

    # Ensure no NaN values
    x = tf.where(tf.math.is_nan(x), tf.zeros_like(x), x)

    # Flatten to match the expected feature size directly if necessary
    flattened_x = tf.reshape(x, [-1])

    # Ensure the expected size matches the actual size needed
    if flattened_x.shape[0] != 156:
        # Use slicing to fit the required shape if larger, remember to adjust source if size is less
        flattened_x = flattened_x[:156]

    return flattened_x

def prepare_input(landmarks):
    landmarks = np.array(landmarks, dtype=np.float32)
    preprocessed_data = pre_process(landmarks)
    input_data = np.expand_dims(preprocessed_data, axis=0)
    return input_data










# Load the TFLite model and allocate tensors
interpreter = tf.lite.Interpreter(model_path=r"C:\Users\nishi\Videos\MS-ASL\model.tflite")
interpreter.allocate_tensors()

# Get input and output tensors
input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()



def predict_real_time(landmark_frame):
    # Prepare the input data
    input_data = prepare_input(landmark_frame)
    input_data = input_data.astype(np.float32)

    # Set the tensor to point to the input data to be inferred
    interpreter.set_tensor(input_details[0]['index'], input_data)
    
    # Run inference
    interpreter.invoke()

    # Get the output data
    output_data = interpreter.get_tensor(output_details[0]['index'])
    
    print(f"Output data shape: {output_data.shape}")  # Debugging: verify output shape

    # Decode output: For each time step (12 in total), find the index of max probability
    predicted_sequence = np.argmax(output_data, axis=-1)
    print("Predicted indices: ", predicted_sequence)

# Example log to investigate each step in decoding
    decoded_characters = [num_to_char.get(idx, '?') for idx in predicted_sequence]
    print("Decoded characters: ", decoded_characters)

# Remove padding or special tokens
    predicted_text = "".join(char for char in decoded_characters if char not in ['P', '\<', '\>'])
    print("Filtered text: ", predicted_text)



def get_landmarks_from_video_frame(image):
    """
    Processes a given video frame to detect hand and pose landmarks,
    and returns the landmarks as a numpy array.

    Args:
    - image (numpy.ndarray): The video frame from which landmarks are to be extracted.

    Returns:
    - numpy.ndarray: The detected landmarks, shape (num_landmarks, 3) for landmarks with (x, y, z) coordinates.
                     Returns None if no landmarks are detected.
    """
    # Convert the BGR image to RGB
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    
    # Process the frame with MediaPipe
    hand_results = hands.process(image_rgb)
    pose_results = pose.process(image_rgb)

    if not hand_results.multi_hand_landmarks or not pose_results.pose_landmarks:
        return None

    # Extract right hand landmarks
    right_hand_landmarks = np.array([[landmark.x, landmark.y, landmark.z] for landmark in hand_results.multi_hand_landmarks[0].landmark]) if hand_results.multi_hand_landmarks else np.zeros((21, 3))
    
    # Extract pose landmarks
    pose_landmarks = np.array([[landmark.x, landmark.y, landmark.z] for i, landmark in enumerate(pose_results.pose_landmarks.landmark) if i in POSE]) if pose_results.pose_landmarks else np.zeros((len(POSE), 3))

    # Combine landmarks into a single array
    landmarks = np.concatenate([right_hand_landmarks.flatten(), pose_landmarks.flatten()])[:len(FEATURE_COLUMNS)]
    if landmarks.size < len(FEATURE_COLUMNS):
        padding = np.zeros(len(FEATURE_COLUMNS) - landmarks.size)
        landmarks = np.concatenate([landmarks, padding])
    
    return landmarks

# Real-time video processing and prediction
cap = cv2.VideoCapture(0)  # Use 0 for the default camera

while cap.isOpened():
    ret, frame = cap.read()  # 'ret' is a boolean indicating if the frame was successfully captured
    
    if not ret:
        print("Failed to capture image")
        break

    # Process the frame to extract landmarks
    landmarks = get_landmarks_from_video_frame(frame)

    if landmarks is not None:
        # Perform real-time prediction using the extracted landmarks
        predicted_text = predict_real_time(landmarks)
        
        # Display the prediction on the frame
        cv2.putText(frame, predicted_text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2, cv2.LINE_AA)

    # Show the frame with predictions
    cv2.imshow('ASL Real-Time Prediction', frame)

    # Exit loop if 'q' is pressed
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
