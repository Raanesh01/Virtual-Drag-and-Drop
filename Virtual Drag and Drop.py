import cv2
import mediapipe as mp
import numpy as np

# Initialize Mediapipe Hands
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(static_image_mode=False, max_num_hands=1, min_detection_confidence=0.7,
                       min_tracking_confidence=0.7)
mp_drawing = mp.solutions.drawing_utils


# Define DragImage class
class DragImage:
    def __init__(self, image_path, posCenter, size=(150, 150)):
        self.image = cv2.imread(image_path, cv2.IMREAD_UNCHANGED)
        self.posCenter = posCenter
        self.size = size
        self.selected = False
        self.image = cv2.resize(self.image, self.size, interpolation=cv2.INTER_AREA)

    def is_cursor_inside(self, cursor):
        cx, cy = self.posCenter
        w, h = self.size
        return cx - w // 2 < cursor[0] < cx + w // 2 and cy - h // 2 < cursor[1] < cy + h // 2

    def update_position(self, cursor):
        if self.selected:
            self.posCenter = cursor

    def draw(self, frame):
        cx, cy = self.posCenter
        x1, y1 = cx - self.size[0] // 2, cy - self.size[1] // 2
        overlay_image(frame, self.image, x1, y1)


# Overlay PNG image with transparency
def overlay_image(frame, overlay, x, y):
    h, w = overlay.shape[:2]

    if overlay.shape[2] == 4:  # Check for alpha channel
        alpha_overlay = overlay[:, :, 3] / 255.0  # Normalize alpha channel
        alpha_frame = 1.0 - alpha_overlay

        for c in range(3):  # Blend each channel
            frame[y:y + h, x:x + w, c] = (
                    alpha_overlay * overlay[:, :, c] +
                    alpha_frame * frame[y:y + h, x:x + w, c]
            )
    else:
        frame[y:y + h, x:x + w] = overlay  # If no alpha channel, directly overlay


# Load images and initialize DragImage objects
image_paths = ["C:/Users/raanesh/PycharmProjects/pythonProject/Images/dachshund-puppies-being-together.jpg","C:/Users/raanesh/PycharmProjects/pythonProject/Images/dragons-fantasy-artificial-intelligence-image.jpg", "C:/Users/raanesh/PycharmProjects/pythonProject/Images/futuristic-dj-using-virtual-reality-glasses-headline-party-play-music.jpg", "C:/Users/raanesh/PycharmProjects/pythonProject/Images/hot-air-balloon-realistic-dreamscape.jpg", "C:/Users/raanesh/PycharmProjects/pythonProject/Images/singapore-skyline-night.jpg"]
drag_images = [DragImage(image_paths[i], [200 + i * 200, 300]) for i in range(len(image_paths))]

# Video capture
cap = cv2.VideoCapture(0)
cap.set(3, 1280)
cap.set(4, 720)
cap.set(cv2.CAP_PROP_BRIGHTNESS, 130)

fourcc = cv2.VideoWriter_fourcc(*'avc1')  # H.264 codec
out = cv2.VideoWriter('output.mp4', fourcc, 20.0, (1280, 720))

while True:
    success, frame = cap.read()
    if not success:
        print("Failed to capture video")
        break

    # Flip and process the frame
    frame = cv2.flip(frame, 1)
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    result = hands.process(rgb_frame)

    cursor = None
    crossing_fingers = False

    # Detect hands and landmarks
    if result.multi_hand_landmarks:
        for hand_landmarks in result.multi_hand_landmarks:
            mp_drawing.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)
            h, w, c = frame.shape

            # Index finger tip and middle finger tip
            index_tip = hand_landmarks.landmark[8]
            middle_tip = hand_landmarks.landmark[12]

            cursor = (int(index_tip.x * w), int(index_tip.y * h))
            cv2.circle(frame, cursor, 10, (0, 255, 0), cv2.FILLED)

            # Measure distance between index and middle finger
            ix, iy = int(index_tip.x * w), int(index_tip.y * h)
            mx, my = int(middle_tip.x * w), int(middle_tip.y * h)
            distance = np.sqrt((ix - mx) ** 2 + (iy - my) ** 2)

            # Crossing fingers if the distance is small
            crossing_fingers = distance < 30

    # Handle dragging logic
    for drag_img in drag_images:
        if crossing_fingers and cursor and drag_img.is_cursor_inside(cursor):
            drag_img.selected = True
        elif not crossing_fingers:
            drag_img.selected = False

        drag_img.update_position(cursor)

    # Draw the images
    for drag_img in drag_images:
        drag_img.draw(frame)

    # Display the frame
    cv2.imshow("Drag and Drop Images", frame)

    # Exit on pressing 'q'
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release resources
cap.release()
cv2.destroyAllWindows()
