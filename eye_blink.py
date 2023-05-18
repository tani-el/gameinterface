import cv2

def detect_blink(roi):
    gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
    _, threshold = cv2.threshold(gray, 60, 255, cv2.THRESH_BINARY)
    contours, _ = cv2.findContours(threshold, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if len(contours) == 1:
        return True
    else:
        return False

# Define the landmarks for each facial element
LEFT_EYE = [362, 382, 381, 380, 374, 373, 390, 249, 263, 466, 388, 387, 386, 385, 384, 398]
RIGHT_EYE = [33, 7, 163, 144, 145, 153, 154, 155, 133, 173, 157, 158, 159, 160, 161, 246]
LEFT_IRIS = [474, 475, 476, 477]
RIGHT_IRIS = [469, 470, 471, 472]
FACE_OUTLINE = [10, 338, 297, 332, 284, 251, 389, 356, 454, 323,
                361, 288, 397, 365, 379, 378, 400, 377, 152, 148,
                176, 149, 150, 136, 172, 58, 132, 93, 234, 127,
                162, 21, 54, 103, 67, 109]
FACE_HEAD_POSE_LANDMARKS = [1, 33, 61, 199, 291, 263]

# Load the pre-trained face and eye cascade classifiers
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
eye_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_eye.xml')

# Open the webcam
cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()
    frame = cv2.flip(frame, 1)

    # Convert frame to grayscale for face and eye detection
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Detect faces in the frame
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))

    for (x, y, w, h) in faces:
        # Draw a rectangle around the face
        cv2.rectangle(frame, (x, y), (x+w, y+h), (255, 0, 0), 2)

        # Extract the region of interest (ROI) within the face for eye detection
        roi_gray = gray[y:y+h, x:x+w]
        roi_color = frame[y:y+h, x:x+w]

        # Detect eyes within the ROI
        eyes = eye_cascade.detectMultiScale(roi_gray)

        for (ex, ey, ew, eh) in eyes:
            # Draw a rectangle around each eye
            cv2.rectangle(roi_color, (ex, ey), (ex+ew, ey+eh), (0, 255, 0), 2)

            # Check if the eye belongs to the left or right eye based on its position
            if ex < w // 2: 
                   landmarks = LEFT_EYE
            else:
                   landmarks = RIGHT_EYE


        # Extract the region of interest (ROI) within each eye for blinking detection
        eye_roi_color = roi_color[ey:ey+eh, ex:ex+ew]

        # Perform blink detection using the eye landmarks
        blink = detect_blink(eye_roi_color)

        if blink:
            print("Eye blink detected!")

        # Draw landmarks for the eye
        for landmark in landmarks:
              cv2.circle(roi_color, (ex + landmark, ey + landmark), 1, (0, 0, 255), -1)


		# Display the resulting frame
		
		#cv2.imshow('Eye Blink Detection', frame)

		# Exit the loop if 'q' is pressed
		#if cv2.waitKey(1) & 0xFF == ord('q'):
			#break