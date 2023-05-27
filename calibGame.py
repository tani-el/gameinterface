import matplotlib
matplotlib.use("Agg")
import pygame
import sys
import random
import main8 as M
# 오류 발생중


clock = pygame.time.Clock()

SCREEN_WIDTH = 800
SCREEN_HEIGHT = 800



calibration_x, calibration_y = 0, 0


def calibration(x,y):
    global calibration_x, calibration_y

SCREEN_WIDTH = 800
SCREEN_HEIGHT = 800
font40 = pygame.font.SysFont(None, 40)
clock = pygame.time.Clock()

num = [1,2,3,4,5,6,7,8,9,10,11,12,13]
clicknum = [1,2,3,4,5,6,7,8,9,10,11,12,13]
random.shuffle(num)
tar = random.choice(num)-1

white = (255, 255, 255)
black = (0, 0, 0)
yellow = (255,255,0)

def draw_Game():
    
    pos_x, pos_y = x, y
    print(x, y)
    window.fill(black)
    group.draw(window)
    pygame.draw.circle(window, white, (pos_x, pos_y), 10)
    window.blit(numbers[1], ((window.get_width() // 4)+15,(window.get_height() // 4)+15))
    window.blit(numbers[2], ((window.get_width() // 4)*3 -35,(window.get_height() // 4)+15))
    window.blit(numbers[3], ((window.get_width() // 4)+15,(window.get_height() // 4)*3 - 35))
    window.blit(numbers[4], ((window.get_width() // 4)*3 -35, (window.get_height() // 4)*3 - 35))
    window.blit(numbers[0], ((window.get_width() // 2)-10, (window.get_height() // 2)-10))
    window.blit(numbers[5], (15, 15))
    window.blit(numbers[10], (15, window.get_height()-35))
    window.blit(numbers[7], (window.get_width()-35, 15))
    window.blit(numbers[6], ((window.get_width() // 2) -10, 15))
    window.blit(numbers[11], ((window.get_width() // 2) -10, window.get_height()-35))
    window.blit(numbers[8], (15, (window.get_height()//2)-10))
    window.blit(numbers[9], ((window.get_width()-35, (window.get_height()//2)-10)))
    window.blit(numbers[12], (window.get_width()-35, window.get_height()-35))

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
window = pygame.display.set_mode((SCREEN_WIDTH, SCREEN_HEIGHT))

        
        

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
x, y = M.p2
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

pygame.display.update()

    

