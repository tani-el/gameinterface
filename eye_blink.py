import cv2
import pygame
from pygame.locals import *
import numpy as np

def detect_blink(eyes):
    gray = cv2.cvtColor(eyes, cv2.COLOR_BGR2GRAY)
    _, threshold = cv2.threshold(gray, 60, 255, cv2.THRESH_BINARY)
    contours, _ = cv2.findContours(threshold, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if len(contours) == 1:
        return True
    else:
        return False

def draw_landmarks(frame, landmarks, color):
    for point in landmarks:
        x, y = point[0], point[1]
        cv2.circle(frame, (x, y), 1, color, -1)

LEFT_EYE = [362, 382, 381, 380, 374, 373, 390, 249, 263, 466, 388, 387, 386, 385, 384, 398]
RIGHT_EYE = [33, 7, 163, 144, 145, 153, 154, 155, 133, 173, 157, 158, 159, 160, 161, 246]

LEFT_IRIS = [474, 475, 476, 477]
RIGHT_IRIS = [469, 470, 471, 472]

FACE_OUTLINE = [10, 338, 297, 332, 284, 251, 389, 356, 454, 323,
                361, 288, 397, 365, 379, 378, 400, 377, 152, 148,
                176, 149, 150, 136, 172, 58, 132, 93, 234, 127,
                162, 21, 54, 103, 67, 109]

FACE_HEAD_POSE_LANDMARKS = [1, 33, 61, 199, 291, 263]

face_2d = []
face_3d = []

compensated_angle = [0, 0, 0]

pygame.init()
pygame.display.set_caption("Blink Detection")
screen = pygame.display.set_mode((640, 480))
clock = pygame.time.Clock()

cap = cv2.VideoCapture(0)
prev_blink = False
space_pressed = False

while True:
    ret, frame = cap.read()
    frame = cv2.flip(frame, 1)
    frame_copy = frame.copy()
    roi = frame[100:300, 200:400]

    cv2.rectangle(frame, (200, 100), (400, 300), (0, 255, 0), 2)

    blink = detect_blink(roi)

    if blink and not prev_blink and not space_pressed:
        print("눈 깜빡임!")
        space_pressed = True

    if not blink:
        space_pressed = False

    prev_blink = blink

    for event in pygame.event.get():
        if event.type == QUIT:
            pygame.quit()
            cap.release()
            cv2.destroyAllWindows()
            exit()

        if event.type == KEYDOWN and event.key == K_SPACE:
            space_pressed = True

        if event.type == KEYUP and event.key == K_SPACE:
            space_pressed = False

        # Draw landmarks
        draw_landmarks(frame, LEFT_EYE, (255, 0, 0))
        draw_landmarks(frame, RIGHT_EYE, (255, 0, 0))
        draw_landmarks(frame, LEFT_IRIS, (0, 0, 255))
        draw_landmarks(frame, RIGHT_IRIS, (0, 0, 255))
        draw_landmarks(frame, FACE_OUTLINE, (0, 255, 0))
        draw_landmarks(frame, FACE_HEAD_POSE_LANDMARKS, (0, 255, 0))

        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        frame = np.rot90(frame)
        frame = pygame.surfarray.make_surface(frame)
        screen.blit(frame, (0, 0))
        pygame.display.update()
        clock.tick(60)