import mediapipe as mp
import numpy as np
import cv2 as cv
import pygame
import sys
import random
import dlib # for face and landmark detection
import imutils
# for calculating dist b/w the eye landmarks
from scipy.spatial import distance as dist
# to get the landmark ids of the left and right eyes
# you can do this manually too
from imutils import face_utils


# first commit


"""def calibration(x,y):
    global calibration_x, calibration_y"""
    
keyInput = [True, True, True, True, True]
num = [1,2,3,4,5,6,7,8,9,10,11,12,13]
clicknum = [1,2,3,4,5,6,7,8,9,10,11,12,13]
random.shuffle(num)


class SpriteObject(pygame.sprite.Sprite):
    def __init__(self, x, y, color, target):
        super().__init__()
        self.original_image = pygame.Surface((50, 50), pygame.SRCALPHA)
        pygame.draw.circle(self.original_image, color, (25, 25), 25)
        self.hover_image = pygame.Surface((50, 50), pygame.SRCALPHA)
        self.hover_image2 = pygame.Surface((50, 50), pygame.SRCALPHA)
        pygame.draw.circle(self.hover_image, color, (25, 25), 25)
        pygame.draw.circle(self.hover_image, (255, 255, 255), (25, 25), 25, 4)  # 흰색으로 변하는 부분
        pygame.draw.circle(self.hover_image2, color, (25, 25), 25)
        pygame.draw.circle(self.hover_image2, (255,255,0), (25, 25), 25, 4)
        self.target = target
        self.image = self.original_image
        self.rect = self.image.get_rect(center=(x, y))
        self.hover = False
        self.x = x
        self.y = y
        
    def get_x(self):
        return self.x
    def get_y(self):
        return self.y
    
            
    def new_target(self):
        self.tar = random.choice(num)-1
        setattr(self.list[self.tar],'target',True)
        
    def update(self):
        mouse_pos = self.x, self.y
        global keyInput
        global calibration_x, calibration_y
        
        # 현재 키보드 상태 감지
        keys = pygame.key.get_pressed()
        if self.target == True:
            self.image = self.hover_image2
        else:
            self.image = self.original_image
        
        # 스페이스바 입력 감지 예시
        if keys[pygame.K_SPACE]:  # 스페이스바가 눌렸을 때
            hover = self.rect.collidepoint(mouse_pos)
            self.hover = hover
            
            if hover:
                if self.target == True:
                    self.image = self.hover_image
                    self.target = False
                    tar = random.choice(num)-1
                    return tar


                else:
                    self.image = self.hover_image
                
            else:
                self.image = self.original_image
                
        
        elif keys[pygame.K_1] and keyInput[0]:
            
            calibration_x, calibration_y = 0, 0
            calibration_x, calibration_y = w // 2 - x, h // 2 - y                # 중앙 보정 KEY_1
            #print("보정 좌표값 : ", calibration_x, calibration_y)
            keyInput[0] = False
        elif keys[pygame.K_2] and keyInput[1]:
            #global calibration_x, calibration_yx
            calibration_x, calibration_y = 0, 0
            calibration_x, calibration_y = (w//4) + 25 -x,(h // 4)+25 - y        # 중간 왼위 보정 KEY_2
            #print("보정 좌표값 : ", calibration_x, calibration_y)
            keyInput[1] = False
        elif keys[pygame.K_3] and keyInput[2]:
            #global calibration_x, calibration_yx
            calibration_x, calibration_y = 0, 0
            calibration_x, calibration_y = (w // 4)*3 -25 -x,(h // 4)+25 -y      # 중간 오위 보정 KEY_3
            #print("보정 좌표값 : ", calibration_x, calibration_y)
            keyInput[2] = False
        
        elif keys[pygame.K_4] and keyInput[3]:
            #global calibration_x, calibration_yx
            calibration_x, calibration_y = 0, 0
            calibration_x, calibration_y = (w // 4)+25 -x,(h // 4)*3 - 25 -y      # 중간 왼아래 보정 KEY_4
            #print("보정 좌표값 : ", calibration_x, calibration_y)
            keyInput[3] = False
        elif keys[pygame.K_5] and keyInput[4]:
            #global calibration_x, calibration_yx
            calibration_x, calibration_y = 0, 0
            calibration_x, calibration_y = (w // 4)*3 -25 -x, (h // 4)*3 - 25 -y      # 중간 오아래 보정 KEY_5
            #print("보정 좌표값 : ", calibration_x, calibration_y)
            keyInput[4] = False
       

            
     
class pygame_Calib():
    
    def __init__(self,x,y):

        self.white = (255, 255, 255)
        self.black = (0, 0, 0)
        self.yellow = (255,255,0)
        global calibration_x, calibration_y 
        calibration_x, calibration_y= 0, 0

        self.x = x
        self.y = y
        
        pygame.init()
        pygame.display.set_caption("Simple PyGame Example")
        
        self.clock = pygame.time.Clock()
        
        self.window = pygame.display.set_mode((w, h))
        #self.font40 = pygame.font.SysFont(None, 40)
        self.clock = pygame.time.Clock()
        
        self.sprite_object = SpriteObject(*self.window.get_rect().center, (128, 128, 0),False)
        self.group = pygame.sprite.Group([
            SpriteObject((self.window.get_width() // 4)+25,(self.window.get_height() // 4)+25, (128, 0, 0),False),
            SpriteObject((self.window.get_width() // 4)*3 -25,(self.window.get_height() // 4)+25,(0, 128, 0),False), 
            SpriteObject((self.window.get_width() // 4)+25,(self.window.get_height() // 4)*3 - 25, (0, 0, 128),False),
            SpriteObject((self.window.get_width() // 4)*3 -25, (self.window.get_height() // 4)*3 - 25, (128, 128, 0),False),
            SpriteObject(self.window.get_width() // 2, self.window.get_height() // 2, (0, 96, 128),False), #중앙
            SpriteObject(self.window.get_width()-25, self.window.get_height()-25, (128, 0, 96),False), #우하
            SpriteObject(25, 25, (64, 0, 128),False), #좌상
            SpriteObject(25, self.window.get_height()-25, (128, 64, 0),False),#좌하
            SpriteObject(self.window.get_width()-25, 25, (32, 128, 0),False),#우상
            SpriteObject(self.window.get_width() // 2, 25, (255, 102, 204),False),#중상
            SpriteObject(self.window.get_width() // 2, self.window.get_height()-25, (0, 102, 255),False), #중하
            SpriteObject(25, self.window.get_height()//2, (153, 255, 153),False),  #중좌
            SpriteObject(self.window.get_width()-25, self.window.get_height()//2, (255, 255, 102),False) # 중우
        ])        
        
        self.window.fill(self.black)
        self.group.draw(self.window)
        pygame.draw.circle(self.window, self.white, (pos_x, pos_y), 10)
        

        self.tar = random.choice(num)-1
        self.list = self.group.sprites()
        setattr(self.list[self.tar],'target',True)
        self.group.update()
        myfont = pygame.font.SysFont(None,30)
        
        
        self.numbers = [
            myfont.render(str(clicknum[0]),True,self.white),
            myfont.render(str(clicknum[1]),True,self.white),
            myfont.render(str(clicknum[2]),True,self.white),
            myfont.render(str(clicknum[3]),True,self.white),
            myfont.render(str(clicknum[4]),True,self.white),
            myfont.render(str(clicknum[5]),True,self.white),
            myfont.render(str(clicknum[6]),True,self.white),
            myfont.render(str(clicknum[7]),True,self.white),
            myfont.render(str(clicknum[8]),True,self.white),
            myfont.render(str(clicknum[9]),True,self.white),
            myfont.render(str(clicknum[10]),True,self.white),
            myfont.render(str(clicknum[11]),True,self.white),
            myfont.render(str(clicknum[12]),True,self.white)
        ]
        self.draw_pygame() 
        

    
    def draw_pygame(self):
        global calibration_x, calibration_y
        

        if calibration_x != 0 and self.x > 0:
            self.x *= 1.0 + calibration_x / self.x
        if calibration_y != 0 and self.y > 0:
            self.y *= 1.0 + calibration_y / self.y

        self.window.blit(self.numbers[1], ((self.window.get_width() // 4)+15,(self.window.get_height() // 4)+15))
        self.window.blit(self.numbers[2], ((self.window.get_width() // 4)*3 -35,(self.window.get_height() // 4)+15))
        self.window.blit(self.numbers[3], ((self.window.get_width() // 4)+15,(self.window.get_height() // 4)*3 - 35))
        self.window.blit(self.numbers[4], ((self.window.get_width() // 4)*3 -35, (self.window.get_height() // 4)*3 - 35))
        self.window.blit(self.numbers[0], ((self.window.get_width() // 2)-10, (self.window.get_height() // 2)-10))
        self.window.blit(self.numbers[5], (15, 15))
        self.window.blit(self.numbers[10], (15, self.window.get_height()-35))
        self.window.blit(self.numbers[7], (self.window.get_width()-35, 15))
        self.window.blit(self.numbers[6], ((self.window.get_width() // 2) -10, 15))
        self.window.blit(self.numbers[11], ((self.window.get_width() // 2) -10, self.window.get_height()-35))
        self.window.blit(self.numbers[8], (15, (self.window.get_height()//2)-10))
        self.window.blit(self.numbers[9], ((self.window.get_width()-35, (self.window.get_height()//2)-10)))
        self.window.blit(self.numbers[12], (self.window.get_width()-35, self.window.get_height()-35))
        
        self.group.update()
        pygame.display.update()
        
    
                           

pos_x = 200
pos_y = 200

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
capture = cv.VideoCapture(0)
w = capture.get(cv.CAP_PROP_FRAME_WIDTH)
h = capture.get(cv.CAP_PROP_FRAME_HEIGHT)
#capture.set(cv.CAP_PROP_FRAME_WIDTH, w*2) # 가로
#capture.set(cv.CAP_PROP_FRAME_HEIGHT, h*2) # 세로




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
blink_thresh = 0.45
succ_frame = 2
count_frame = 0

# Eye landmarks
(L_start, L_end) = face_utils.FACIAL_LANDMARKS_IDXS["left_eye"]
(R_start, R_end) = face_utils.FACIAL_LANDMARKS_IDXS['right_eye']

# Initializing the Models for Landmark and
# face Detection
detector = dlib.get_frontal_face_detector()
landmark_predict = dlib.shape_predictor('shape_predictor_68_face_landmarks.dat')

def detect_blink(frame):

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

            # Blink detected!
            return True

        # No blink detected.
        return False




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
            _, fra = capture.read()
            fra = imutils.resize(fra, width=640)
            blink_check = detect_blink(fra)
            if blink_check:
                print("blink!!")
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

        
        # 시선 기초 보정값
        x, y = p2
        x += 50
        #y += 100
        if x > 600:
            x *= 1.15
        if y > 300:
            y *= 2.1
        else:
            y *= 2.0   
        




        pos_x, pos_y = x, y
        print(x, y)
        #class에서 그리기 위치
        
        pygame_Calib(x,y)
capture.release()
cv.destroyAllWindows()
pygame.quit()
exit()