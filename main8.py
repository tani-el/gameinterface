import mediapipe as mp
import numpy as np
import cv2 as cv
import pygame
import sys
import random

# first commit

SCREEN_WIDTH = 800
SCREEN_HEIGHT = 800

white = (255, 255, 255)
black = (0, 0, 0)
yellow = (255,255,0)

calibration_x, calibration_y = 0, 0

num = [1,2,3,4,5,6,7,8,9,10,11,12,13]
clicknum = [1,2,3,4,5,6,7,8,9,10,11,12,13]
random.shuffle(num)
tar = random.choice(num)-1



def calibration(x,y):
    global calibration_x, calibration_y
    
keyInput = [True, True, True, True, True]

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
       
#dssdsddd
    def update(self, x, y):
        mouse_pos = x, y
        global keyInput
        # 현재 키보드 상태 감지
        keys = pygame.key.get_pressed()
        if self.target == True:
            self.image = self.hover_image2
        else:
            self.image = self.original_image


        global calibration_x, calibration_y
        # 스페이스바 입력 감지 예시
        if keys[pygame.K_SPACE]:  # 스페이스바가 눌렸을 때
            hover = self.rect.collidepoint(mouse_pos)
            self.hover = hover
            
            if hover:
                if self.target == True:
                    self.image = self.hover_image
                    self.target = False
                    tar = random.choice(num)-1

                else:
                    self.image = self.hover_image
                
            else:
                self.image = self.original_image
                
        
        elif keys[pygame.K_1] and keyInput[0]:
            
            calibration_x, calibration_y = 0, 0
            calibration_x, calibration_y = window.get_width() // 2 - x, window.get_height() // 2 - y                # 중앙 보정 KEY_1
            #print("보정 좌표값 : ", calibration_x, calibration_y)
            keyInput[0] = False

        elif keys[pygame.K_2] and keyInput[1]:
            #global calibration_x, calibration_yx
            calibration_x, calibration_y = 0, 0
            calibration_x, calibration_y = (window.get_width()//4) + 25 -x,(window.get_height() // 4)+25 - y        # 중간 왼위 보정 KEY_2
            #print("보정 좌표값 : ", calibration_x, calibration_y)
            keyInput[1] = False

        elif keys[pygame.K_3] and keyInput[2]:
            #global calibration_x, calibration_yx
            calibration_x, calibration_y = 0, 0
            calibration_x, calibration_y = (window.get_width() // 4)*3 -25 -x,(window.get_height() // 4)+25 -y      # 중간 오위 보정 KEY_3
            #print("보정 좌표값 : ", calibration_x, calibration_y)
            keyInput[2] = False
        
        elif keys[pygame.K_4] and keyInput[3]:
            #global calibration_x, calibration_yx
            calibration_x, calibration_y = 0, 0
            calibration_x, calibration_y = (window.get_width() // 4)+25 -x,(window.get_height() // 4)*3 - 25 -y      # 중간 왼아래 보정 KEY_4
            #print("보정 좌표값 : ", calibration_x, calibration_y)
            keyInput[3] = False

        elif keys[pygame.K_5] and keyInput[4]:
            #global calibration_x, calibration_yx
            calibration_x, calibration_y = 0, 0
            calibration_x, calibration_y = (window.get_width() // 4)*3 -25 -x, (window.get_height() // 4)*3 - 25 -y      # 중간 오아래 보정 KEY_5
            #print("보정 좌표값 : ", calibration_x, calibration_y)
            keyInput[4] = False


            




pygame.init()
pygame.display.set_caption("Simple PyGame Example")
screen = pygame.display.set_mode((SCREEN_WIDTH, SCREEN_HEIGHT))

pos_x = 200
pos_y = 200

clock = pygame.time.Clock()

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

#capture.set(cv.CAP_PROP_FRAME_WIDTH, w*2) # 가로
#capture.set(cv.CAP_PROP_FRAME_HEIGHT, h*2) # 세로

window = pygame.display.set_mode((w, h))
font40 = pygame.font.SysFont(None, 40)
clock = pygame.time.Clock()




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

        

       
        
        sprite_object = SpriteObject(*window.get_rect().center, (128, 128, 0),False)

        group = pygame.sprite.Group([
            SpriteObject((window.get_width() // 4)+25,(window.get_height() // 4)+25, (128, 0, 0),False),
            SpriteObject((window.get_width() // 4)*3 -25,(window.get_height() // 4)+25,(0, 128, 0),False), 
            SpriteObject((window.get_width() // 4)+25,(window.get_height() // 4)*3 - 25, (0, 0, 128),False),
            SpriteObject((window.get_width() // 4)*3 -25, (window.get_height() // 4)*3 - 25, (128, 128, 0),False),
            SpriteObject(window.get_width() // 2, window.get_height() // 2, (0, 96, 128),False), #중앙
            SpriteObject(window.get_width()-25, window.get_height()-25, (128, 0, 96),False), #우하
            SpriteObject(25, 25, (64, 0, 128),False), #좌상
            SpriteObject(25, window.get_height()-25, (128, 64, 0),False),#좌하
            SpriteObject(window.get_width()-25, 25, (32, 128, 0),False),#우상
            SpriteObject(window.get_width() // 2, 25, (255, 102, 204),False),#중상
            SpriteObject(window.get_width() // 2, window.get_height()-25, (0, 102, 255),False), #중하
            SpriteObject(25, window.get_height()//2, (153, 255, 153),False),  #중좌
            SpriteObject(window.get_width()-25, window.get_height()//2, (255, 255, 102),False) # 중우
        ])



        #print(x, y)
        
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
        
        
        # calibration 보정

        group.update(x, y)
        
        if calibration_x != 0 and x > 0:
            x *= 1.0 + calibration_x / x

        if calibration_y != 0 and y > 0:
            y *= 1.0 + calibration_y / y

        
        

        list = group.sprites()
        setattr(list[tar],'target',True)
        
        



        myfont = pygame.font.SysFont(None,30)
        

        numbers = [
            myfont.render(str(clicknum[0]),True,white),
            myfont.render(str(clicknum[1]),True,white),
            myfont.render(str(clicknum[2]),True,white),
            myfont.render(str(clicknum[3]),True,white),
            myfont.render(str(clicknum[4]),True,white),
            myfont.render(str(clicknum[5]),True,white),
            myfont.render(str(clicknum[6]),True,white),
            myfont.render(str(clicknum[7]),True,white),
            myfont.render(str(clicknum[8]),True,white),
            myfont.render(str(clicknum[9]),True,white),
            myfont.render(str(clicknum[10]),True,white),
            myfont.render(str(clicknum[11]),True,white),
            myfont.render(str(clicknum[12]),True,white)
        ]

       
        
        

        mouse_buttons = {1: "left", 2: "middle", 3: "right"}
        button_name = lambda b: mouse_buttons[b] if b in mouse_buttons else "#" + str(b)
        text = "Wait for event ..."

        # pygame.time.Clock().tick(60)


        clock.tick(60)
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                sys.exit()

        key_event = pygame.key.get_pressed()
        if key_event[pygame.K_ESCAPE]:      # ESC 입력시 프로그램 종료
            sys.exit()



        
        
        pos_x, pos_y = x, y
        print(x, y)
        screen.fill(black)
        group.draw(window)
        pygame.draw.circle(screen, white, (pos_x, pos_y), 10)
        screen.blit(numbers[1], ((window.get_width() // 4)+15,(window.get_height() // 4)+15))
        screen.blit(numbers[2], ((window.get_width() // 4)*3 -35,(window.get_height() // 4)+15))
        screen.blit(numbers[3], ((window.get_width() // 4)+15,(window.get_height() // 4)*3 - 35))
        screen.blit(numbers[4], ((window.get_width() // 4)*3 -35, (window.get_height() // 4)*3 - 35))
        screen.blit(numbers[0], ((window.get_width() // 2)-10, (window.get_height() // 2)-10))
        screen.blit(numbers[5], (15, 15))
        screen.blit(numbers[10], (15, window.get_height()-35))
        screen.blit(numbers[7], (window.get_width()-35, 15))
        screen.blit(numbers[6], ((window.get_width() // 2) -10, 15))
        screen.blit(numbers[11], ((window.get_width() // 2) -10, window.get_height()-35))
        screen.blit(numbers[8], (15, (window.get_height()//2)-10))
        screen.blit(numbers[9], ((window.get_width()-35, (window.get_height()//2)-10)))
        screen.blit(numbers[12], (window.get_width()-35, window.get_height()-35))
        
        pygame.display.update()


capture.release()
cv.destroyAllWindows()
pygame.quit()
exit()