import cv2
import mediapipe as mp

# Initialize webcam capture
cap = cv2.VideoCapture(0)

# Initialize MediaPipe hands solution
mp_hands = mp.solutions.hands
hands = mp_hands.Hands()
mp_draw = mp.solutions.drawing_utils

# Function to count fingers
def count_fingers(hand_landmarks, img):
    # Get height and width of the image
    height, width, _ = img.shape
    
    # Get landmarks
    landmarks = hand_landmarks.landmark
    
    # Count fingers (we ignore the thumb for now)
    finger_tips = [8, 12, 16, 20]
    finger_folded = []
    
    # Check if fingers are folded or not
    for tip in finger_tips:
        # Tip landmark (y)
        if landmarks[tip].y < landmarks[tip - 2].y:
            finger_folded.append(1)  # Finger is open
        else:
            finger_folded.append(0)  # Finger is closed
    
    # Thumb
    if landmarks[4].x > landmarks[3].x:
        finger_folded.append(1)  # Thumb is open
    else:
        finger_folded.append(0)  # Thumb is closed
    
    return sum(finger_folded)

while True:
    # Capture frame-by-frame
    _, img = cap.read()
    
    # Convert the BGR image to RGB
    imgRGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    
    # Process the image and detect hand landmarks
    results = hands.process(imgRGB)
    
    # If hand landmarks are detected
    if results.multi_hand_landmarks:
        for hand_landmarks in results.multi_hand_landmarks:
            # Count fingers
            fingers_count = count_fingers(hand_landmarks, img)
            
            # Display the number of fingers
            cv2.putText(img, f'Fingers: {fingers_count}', (10, 70), cv2.FONT_HERSHEY_SIMPLEX, 
                        2, (0, 255, 0), 3)
            
            # Draw hand landmarks and connections on the image
            mp_draw.draw_landmarks(img, hand_landmarks, mp_hands.HAND_CONNECTIONS)
    
    # Display the resulting frame
    cv2.imshow("Video", img)
    
    # Break the loop when 'q' is pressed
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release the capture and destroy windows
cap.release()
cv2.destroyAllWindows()
