import mediapipe as mp
import numpy as np
import cv2 as cv
# import calibGame as py
import dlib # for face and landmark detection
import imutils
# for calculating dist b/w the eye landmarks
from scipy.spatial import distance as dist
# to get the landmark ids of the left and right eyes
# you can do this manually too
from imutils import face_utils
import pygame
import threading
import queue
import time

import merge_game as Num_Game
# for test

STATE_CALIBRATION, STATE_NUM_GAME, STATE_SCORE, STATE_NONE = range(4)
    
keyInput = [True, True, True, True, True]

            
pos_x = 200
pos_y = 200

state = STATE_CALIBRATION

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

mp_face_mesh = mp.solutions.face_mesh
# capture = cv.VideoCapture('sample_vid.mp4')
capture = cv.VideoCapture(0)
w = capture.get(cv.CAP_PROP_FRAME_WIDTH)
h = capture.get(cv.CAP_PROP_FRAME_HEIGHT)






# defining a function to calculate the EAR
def calculate_EAR(eye):

   # calculate the vertical distances
   y1 = dist.euclidean(eye[1], eye[5])
   y2 = dist.euclidean(eye[2], eye[4])

   # calculate the horizontal distance
   x1 = dist.euclidean(eye[0], eye[3])

   # calculate the EAR
   EAR = (y1+y2) / x1
   return EAR



# Variables
blink_thresh = 0.45 - 0.05
succ_frame = 2
count_frame = 0

# Eye landmarks
(L_start, L_end) = face_utils.FACIAL_LANDMARKS_IDXS["left_eye"]
(R_start, R_end) = face_utils.FACIAL_LANDMARKS_IDXS['right_eye']

# Initializing the Models for Landmark and
# face Detection
detector = dlib.get_frontal_face_detector()
landmark_predict = dlib.shape_predictor(
   'shape_predictor_68_face_landmarks.dat')

blink_detected = False  # Blink가 감지되었는지 여부를 나타내는 전역 변수
blink_detected_time = 0  # Blink가 감지된 시간을 저장할 전역 변수


def detect_blink(frame, real_frame):
    global count_frame
    global blink_detected
    global blink_detected_time

    # blink 쿨타임 추가
    blink_cool_time = 4.0

    # Convert the frame to grayscale.
    gray = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)

    # Detect the faces in the frame.
    faces = detector(gray)

    # Loop over the faces.
    for face in faces:
        # Landmark detection.
        shape = landmark_predict(gray, face)
        # Convert the shape class directly to a list of (x,y) coordinates.
        shape = face_utils.shape_to_np(shape)
        # Extract the left and right eye landmarks.
        left_eye = shape[L_start:L_end]
        right_eye = shape[R_start:R_end]

        # Calculate the EAR
        left_EAR = calculate_EAR(left_eye)
        right_EAR = calculate_EAR(right_eye)

        # Calculate the average EAR.
        avg_ear = (left_EAR + right_EAR) / 2

        # Check if the average EAR is less than the blink threshold.
        if avg_ear < blink_thresh:
            # Blink가 감지된 후에는 5초 동안 감지를 쉬도록 처리
            if not blink_detected:
                blink_detected = True
                count_frame += 1  # incrementing the frame count
                blink_detected_time = time.time()
                cv.putText(real_frame, 'Blink!', (30, 30), cv.FONT_HERSHEY_DUPLEX, 1, (0, 0, 255), 1)
                return True
        else:
            if count_frame >= succ_frame and not blink_detected:
                cv.putText(real_frame, 'Blink Detecting...', (30, 30), cv.FONT_HERSHEY_DUPLEX, 1, (0, 200, 0), 1)
                return False
            else:
                # Blink가 감지된 후에는 4초 동안 감지를 쉬도록 처리
                if blink_detected:
                    if time.time() - blink_detected_time >= blink_cool_time:
                        blink_detected = False
                count_frame = 0
                cv.putText(real_frame, 'Blink Detecting...', (30, 30), cv.FONT_HERSHEY_DUPLEX, 1, (255, 0, 0), 1)

                # 감지까지 남은 시간을 화면에 표시
                remaining_time = max(blink_cool_time - (time.time() - blink_detected_time), 0)
                cv.putText(real_frame, f'Remaining Time: {remaining_time:.1f}s', (30, 60), cv.FONT_HERSHEY_DUPLEX, 1,
                           (0, 0, 255), 1)
                return False


# Num_Game 을 돌리기 위한 코드-------------------
game = Num_Game.NumGame(pos_x, pos_y)

# Create a queue for communication between threads
coord_queue = queue.Queue()

def run_game():
    while True:
        # Check if there are updated coordinates in the queue
        if not coord_queue.empty():
            pos_x, pos_y = coord_queue.get()
            game.set_target(pos_x, pos_y)
        
        game.run()
    # 점수판 함수 데모 제작중
'''    
        Num_Game.result_screen.fill((255, 255, 255))
        #Num game 점수
        Num_Game.result_screen.blit(Num_Game.result_text, Num_Game.result_rect)
        #T-RexRuner 점수(임시)
        #Dino_Run.result_screen.blit(Dino_Run.result_text, Dino_Run.result_rect)
        pygame.display.flip()
        # 결과 화면 유지
        running = True
        while running:
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    running = False
'''




# Create and start the game thread
game_thread = threading.Thread(target=run_game)
state = STATE_NUM_GAME
#-----------------------------------------------


with mp_face_mesh.FaceMesh(max_num_faces=1,
                           refine_landmarks=True,
                           min_detection_confidence=0.5,
                           min_tracking_confidence=0.5
                           ) as face_mesh:
    while True:
        # if capture.get(cv.CAP_PROP_POS_FRAMES) == capture.get(cv.CAP_PROP_FRAME_COUNT):
        #    capture.set(cv.CAP_PROP_POS_FRAMES, 0)
        ret, frame = capture.read()
        if not ret:
            break
        img_h, img_w = frame.shape[:2]
        frame = cv.flip(frame, 1)
        
        image = cv.cvtColor(frame, cv.COLOR_RGB2BGR)
        results = face_mesh.process(image)
        if results.multi_face_landmarks:
            mesh_points = np.array([np.multiply([p.x, p.y], [img_w, img_h]).astype(int)
                                    for p in results.multi_face_landmarks[0].landmark])

            # drawing left/right eye
            cv.polylines(frame, [mesh_points[LEFT_EYE]], True, (0, 255, 0), 2, cv.LINE_AA)
            cv.polylines(frame, [mesh_points[RIGHT_EYE]], True, (0, 255, 0), 2, cv.LINE_AA)

            # drawing face outline
            cv.polylines(frame, [mesh_points[FACE_OUTLINE]], True, (255, 255, 255), 2, cv.LINE_AA)
            # cv.polylines(frame, [mesh_points[FACE_HEAD_POSE_LANDMARKS ]], True, (255,255,255),2,cv.LINE_AA)

            # drawing left/right iris
            (l_cx, l_cy), l_rad = cv.minEnclosingCircle(mesh_points[LEFT_IRIS])
            (r_cx, r_cy), r_rad = cv.minEnclosingCircle(mesh_points[RIGHT_IRIS])
            l_center = np.array([l_cx, l_cy], dtype=np.int32)
            r_center = np.array([r_cx, r_cy], dtype=np.int32)
            cv.circle(frame, l_center, int(l_rad), (0, 0, 255), 2, cv.LINE_AA)
            cv.circle(frame, r_center, int(r_rad), (0, 0, 255), 2, cv.LINE_AA)
            # drawing all face mesh points as dots
            for pt in mesh_points:
                (cx, cy) = pt[0], pt[1]
                cv.circle(frame, [cx, cy], 1, (255, 255, 255), -1, cv.LINE_AA)

            for idx, lm in enumerate(results.multi_face_landmarks[0].landmark):
                if idx in FACE_HEAD_POSE_LANDMARKS:
                    if idx == 1:
                        nose_2d = (lm.x * img_w, lm.y * img_h)
                        nose_3d = (lm.x * img_w, lm.y * img_h, lm.z * 3000)

                    x, y = int(lm.x * img_w), int(lm.y * img_h)

                    face_2d.append([x, y])
                    face_3d.append([x, y, lm.z])

            face_2d = np.array(face_2d, dtype=np.float64)

            face_3d = np.array(face_3d, dtype=np.float64)

            focal_len = 1 * img_w

            camera_mat = np.array([[focal_len, 0, img_h / 2],
                                   [0, focal_len, img_w / 2],
                                   [0, 0, 1]])

            dist_mat = np.zeros((4, 1), dtype=np.float64)

            success, rot_vec, trans_vec = cv.solvePnP(face_3d, face_2d, camera_mat, dist_mat)

            rot_mat, jac = cv.Rodrigues(rot_vec)
            angles, mtxR, mtxQ, Qx, Qy, Qz = cv.RQDecomp3x3(rot_mat)
            x = angles[0] * 360
            y = angles[1] * 360
            z = angles[2] * 360

            # nose_3d_projection, jacobian = cv.projectPoints(nose_3d, rot_vec, trans_vec, camera_mat, dist_mat)

            p1 = (int(nose_2d[0]), int(nose_2d[1]))
            p2_y = int(nose_2d[1] - x * 10) + 100
            
            p2 = (int(nose_2d[0] + y * 10),  p2_y)  #p2 보정

            # Find the center point of the face
            face_center = np.mean(mesh_points[FACE_OUTLINE], axis=0).astype(int)

            # Find the center point of the left and right eye
            left_eye_center = np.mean(mesh_points[LEFT_EYE], axis=0).astype(int)
            right_eye_center = np.mean(mesh_points[RIGHT_EYE], axis=0).astype(int)

            # Draw a line from the face center to the midpoint of the left and right eye
            cv.line(frame, tuple((left_eye_center + right_eye_center) // 2), tuple(face_center), (255, 255, 255), 3)
            cv.line(frame, p1, p2, (255, 255, 0), 3)




            # blink
            # ==============================
            # frame
            _, fra = capture.read()
            fra = imutils.resize(fra, width=640)
            blink_check = detect_blink(fra, frame)
            if blink_check:
                print("blink!!")
                pygame.event.post(pygame.event.Event(pygame.MOUSEBUTTONDOWN, button=1))  # 마우스 버튼 누르기
                time.sleep(0.1)
                pygame.event.post(pygame.event.Event(pygame.MOUSEBUTTONUP, button=1))  # 마우스 버튼 떼기
            else:
                print("Nope")
            face_2d = []
            face_3d = []
            
            
        

        cv.imshow('Main', frame)
        key = cv.waitKey(1)

        if (key == ord('q')):
            break
        if (key == ord('c')):
            pass

        # ------
        # pygame
        # ------
        x, y = p2
        x += 50
        # y += 100
        if x > 600:
            x *= 1.15
        if y > 300:
            y *= 2.1
        else:
            y *= 2.0

        pos_x, pos_y = x, y
        # print(x, y)
        
        if state == STATE_CALIBRATION:
            # 여기에서 cali게임 실행하세요
            # 각 state가 끝날때 STATE = 다음 스테이트로 바꿔주세요
            pass

        # Num_Game 을 돌리기 위해 queue에다 넣어주고 threading함
        # Num_Game 을 돌리기 위한 코드
        if state == STATE_NUM_GAME:
            game_thread.start()
        game.set_target(pos_x, pos_y)

        if state == STATE_SCORE:
            # 여기에서 score 창 클래스 띄우기
            pass
        # Add updated coordinates to the queue
        # coord_queue.put((pos_x, pos_y))
        state = STATE_NONE





# Signal the game thread to exit
# Num_Game 을 돌리기 위한 코드
game_thread.join()




capture.release()
cv.destroyAllWindows()
# py.pygame.quit()
exit()