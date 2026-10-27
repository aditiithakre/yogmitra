import cv2
import mediapipe as mp

from mediapipe.tasks import python
from mediapipe.tasks.python import vision


# Path to model
MODEL_PATH = "pose_landmarker_heavy (1).task"


# Create Pose Landmarker
base_options = python.BaseOptions(
    model_asset_path=MODEL_PATH
)

options = vision.PoseLandmarkerOptions(
    base_options=base_options,
    running_mode=vision.RunningMode.VIDEO,
    num_poses=1,
    min_pose_detection_confidence=0.5,
    min_pose_presence_confidence=0.5,
    min_tracking_confidence=0.5
)

detector = vision.PoseLandmarker.create_from_options(options)


# Open webcam
cap = cv2.VideoCapture(0)

frame_timestamp = 0


while cap.isOpened():

    success, frame = cap.read()

    if not success:
        print("Could not read camera")
        break

    # Mirror image
    frame = cv2.flip(frame, 1)

    # Convert BGR -> RGB
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    # Convert to MediaPipe Image
    mp_image = mp.Image(
        image_format=mp.ImageFormat.SRGB,
        data=rgb_frame
    )

    # Detect pose
    result = detector.detect_for_video(
        mp_image,
        frame_timestamp
    )

    frame_timestamp += 1

    # Draw landmarks
    if result.pose_landmarks:

        landmarks = result.pose_landmarks[0]

        h, w, _ = frame.shape

        # Draw all joints
        for landmark in landmarks:

            x = int(landmark.x * w)
            y = int(landmark.y * h)

            cv2.circle(
                frame,
                (x, y),
                5,
                (0, 255, 0),
                -1
            )

        # Draw some connections
        connections = [
            (11, 13),  # Left shoulder -> left elbow
            (13, 15),  # Left elbow -> left wrist

            (12, 14),  # Right shoulder -> right elbow
            (14, 16),  # Right elbow -> right wrist

            (11, 12),  # Shoulders

            (11, 23),  # Left shoulder -> left hip
            (12, 24),  # Right shoulder -> right hip

            (23, 24),  # Hips

            (23, 25),  # Left hip -> left knee
            (25, 27),  # Left knee -> left ankle

            (24, 26),  # Right hip -> right knee
            (26, 28)   # Right knee -> right ankle
        ]

        for start, end in connections:

            x1 = int(landmarks[start].x * w)
            y1 = int(landmarks[start].y * h)

            x2 = int(landmarks[end].x * w)
            y2 = int(landmarks[end].y * h)

            cv2.line(
                frame,
                (x1, y1),
                (x2, y2),
                (255, 0, 0),
                2
            )

    # Show result
    cv2.imshow(
        "Live Yoga Pose Detection",
        frame
    )

    # Press Q to exit
    if cv2.waitKey(1) & 0xFF == ord("q"):
        break


cap.release()
cv2.destroyAllWindows()
detector.close()