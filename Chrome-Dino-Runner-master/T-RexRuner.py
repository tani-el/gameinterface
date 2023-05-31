from random import randrange as rnd
from itertools import cycle
from random import choice
from PIL import  Image
import pygame
import time
import os


clock = pygame.time.Clock()
FPS = 30
active = True
time_elapsed = 0


pygame.init()
speed = 3.5

# 현재 스크립트 파일의 경로를 가져옴
script_dir = os.path.dirname(os.path.abspath(__file__))

# 같은 폴더에 있는 파일의 경로를 생성
file_path = os.path.join(script_dir, 'resources.png')

# extracting game items and characters form the resource.png image.
player_init = Image.open(file_path).crop((77,5,163,96)).convert("RGBA")
player_init = player_init.resize(list(map(lambda x:x//2 , player_init.size)))

player_frame_1 = Image.open(file_path).crop((1679,2,1765,95)).convert("RGBA")
player_frame_1 = player_frame_1.resize(list(map(lambda x:x//2 , player_frame_1.size)))

player_frame_2 = Image.open(file_path).crop((1767,2,1853,95)).convert("RGBA")
player_frame_2 = player_frame_2.resize(list(map(lambda x:x//2 , player_frame_2.size)))

player_frame_3 = Image.open(file_path).crop((1855,2,1941,95)).convert("RGBA")
player_frame_3 = player_frame_3.resize(list(map(lambda x:x//2 , player_frame_3.size)))

player_frame_31 = Image.open(file_path).crop((1943,2,2029,95)).convert("RGBA")
player_frame_31 = player_frame_31.resize(list(map(lambda x:x//2 , player_frame_31.size)))

player_frame_4 = Image.open(file_path).crop((2030,2,2117,95)).convert("RGBA")
player_frame_4 = player_frame_4.resize(list(map(lambda x:x//2 , player_frame_4.size)))

player_frame_5 = Image.open(file_path).crop((2207,2,2323,95)).convert("RGBA")
player_frame_5 = player_frame_5.resize(list(map(lambda x:x//2 , player_frame_5.size)))

player_frame_6 = Image.open(file_path).crop((2324,2,2441,95)).convert("RGBA")
player_frame_6 = player_frame_6.resize(list(map(lambda x:x//2 , player_frame_6.size)))

cloud = Image.open(file_path).crop((166,2,257,29)).convert("RGBA")
cloud = cloud.resize(list(map(lambda x:x//2 , cloud.size)))

fly_1 = Image.open(file_path).crop((258,10,353,80)).convert("RGBA")
fly_1 = fly_1.resize(list(map(lambda x:x//2 , fly_1.size)))

fly_2 = Image.open(file_path).crop((354,1,443,62)).convert("RGBA")
fly_2 = fly_2.resize(list(map(lambda x:x//2 , fly_2.size)))

ground = Image.open(file_path).crop((2,102,2401,127)).convert("RGBA")
ground = ground.resize(list(map(lambda x:x//2 , ground.size)))

obstacle1 = Image.open(file_path).crop((446,2,479,71)).convert("RGBA")
obstacle1 = obstacle1.resize(list(map(lambda x:x//2 , obstacle1.size)))

obstacle2 = Image.open(file_path).crop((446,2,547,71)).convert("RGBA")
obstacle2 = obstacle2.resize(list(map(lambda x:x//2 , obstacle2.size)))

obstacle3 = Image.open(file_path).crop((446,2,581,71)).convert("RGBA")
obstacle3 = obstacle3.resize(list(map(lambda x:x//2 , obstacle3.size)))

obstacle4 = Image.open(file_path).crop((653,2,701,101)).convert("RGBA")
obstacle4 = obstacle4.resize(list(map(lambda x:x//2 , obstacle4.size)))

obstacle5 = Image.open(file_path).crop((653,2,701,101)).convert("RGBA")
obstacle5 = obstacle5.resize(list(map(lambda x:x//2 , obstacle5.size)))

obstacle5 = Image.open(file_path).crop((653,2,749,101)).convert("RGBA")
obstacle5 = obstacle5.resize(list(map(lambda x:x//2 , obstacle5.size)))

obstacle6 = Image.open(file_path).crop((851,2,950,101)).convert("RGBA")
obstacle6 = obstacle6.resize(list(map(lambda x:x//2 , obstacle6.size)))

speed_identifier = lambda x: 2 if x >= 30 else 8 if x < 8 else 5
cust_speed = speed_identifier(speed)
running = cycle([player_frame_3]*cust_speed+[player_frame_31]*cust_speed)
crouch = cycle([player_frame_5]*cust_speed+ [player_frame_6]*cust_speed)
crouch_scope = [player_frame_5]+[player_frame_6]
obstacles = [obstacle1,obstacle2, obstacle3,obstacle4,obstacle5,obstacle6]
flys = cycle([fly_1]*cust_speed+[fly_2]*cust_speed)


gameDisplay = pygame.display.set_mode((600,200))
pygame.display.set_caption('T-Rex Runner')
clock = pygame.time.Clock()
state = player_frame_1
crashed = False
lock = False
bg = (0, 150)
bg1 = (600,150)
start = False
height = 110
jumping = False
slow_motion = False
c1 = (rnd(30, 600), rnd(0, 100))
c2 = (rnd(50,600), rnd(0, 100))
c3 = (rnd(30,700), rnd(0, 100))
c4 = (rnd(30,600),rnd(0, 100))
obs1 = (rnd(600, 600+500-400), 130)
obs2 = (rnd(600+100+500+1000, 1200+500-400+1000), 130)
obs3 = (rnd(1700+3000, 2000-200+3000), 130)
fly_obj = (rnd(600, 600+500-400), 115)

obast1 = choice(obstacles)
if obast1 in [obstacle4, obstacle5, obstacle6]:obs1 = (obs1[0], 115)
obast2 = choice(obstacles)
if obast2 in [obstacle4, obstacle5, obstacle6]:obs2 = (obs2[0], 115)
obast3 = choice(obstacles)
if obast3 in [obstacle4, obstacle5, obstacle6]:obs3 = (obs3[0], 115)


running_time = True
while not crashed:
    gameDisplay.fill((255,255,255))
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            crashed = True
            running_time = False
        if event.type==pygame.KEYDOWN:
            start = True
            if event.key == pygame.K_DOWN:
                slow_motion = True
                state = crouch
            if event.key == pygame.K_UP:
                if height >= 110:jumping = True
        if event.type==pygame.KEYUP:
            slow_motion = False
            if event.key == pygame.K_DOWN:
                state = running
    # print(start)
    player = state if type(state) != cycle else next(state)
    gameDisplay.blit(pygame.image.fromstring(cloud.tobytes(), cloud.size, 'RGBA'), c1)
    gameDisplay.blit(pygame.image.fromstring(cloud.tobytes(), cloud.size, 'RGBA'), c2)
    gameDisplay.blit(pygame.image.fromstring(cloud.tobytes(), cloud.size, 'RGBA'), c3)
    gameDisplay.blit(pygame.image.fromstring(cloud.tobytes(), cloud.size, 'RGBA'), c4)

    flyer = flys if type(flys) != cycle else next(flys)

    c1 = (c1[0]-1, c1[1])
    c2 = (c2[0]-1, c2[1])
    c3 = (c3[0]-1, c3[1])
    c4 = (c4[0]-1, c4[1])

    if c1[0]<= -50:
        c1 = (640, c1[1])
    if c2[0]<= -50:
        c2 = (700, c2[1])
    if c3[0]<= -50:
        c3 = (600, c3[1])
    if c4[0]<= -50:
        c4 = (800, c4[1])

    gameDisplay.blit(pygame.image.fromstring(ground.tobytes(), ground.size, 'RGBA'), bg)
    gameDisplay.blit(pygame.image.fromstring(ground.tobytes(), ground.size, 'RGBA'), bg1)
    if jumping:
        if height>=110-100:
            height -= 4
        if height <= 110-100:
            jumping = False
    if height<110 and not jumping:
        if slow_motion == True:
            height += 1.5
        else:height += 3
    player = gameDisplay.blit(pygame.image.fromstring(player.tobytes(), player.size, 'RGBA'), (5,height))

    
    gameDisplay.blit(pygame.image.fromstring(obast1.tobytes(), obast1.size, 'RGBA'), obs1)
    gameDisplay.blit(pygame.image.fromstring(obast2.tobytes(), obast2.size, 'RGBA'), obs2)
    gameDisplay.blit(pygame.image.fromstring(obast3.tobytes(), obast3.size, 'RGBA'), obs3)

    gameDisplay.blit(pygame.image.fromstring(fly_2.tobytes(), fly_2.size, 'RGBA'), fly_obj)
    if obs1[0]<=-2500:
        obs1 = (rnd(600, 600+500), 130)
        obast1 = choice(obstacles)
        if obast1 in [obstacle4, obstacle5, obstacle6]:obs1 = (obs1[0], 115)
    if obs2[0]<=-3000:
        obs2 = (rnd(600+100+500, 1200+500), 130)
        obast2 = choice(obstacles)
        if obast2 in [obstacle4, obstacle5, obstacle6]:obs2 = (obs2[0], 115)
    if obs3[0]<=-3500:
        obs3 = (rnd(1700, 2000), 130) 
        obast3 = choice(obstacles) 
        if obast3 in [obstacle4, obstacle5, obstacle6]:obs3 = (obs3[0], 115)
    if fly_obj[0] <=-3600:
        fly_obj = (rnd(600, 700), 60)
    player_stading_cub = (5, height, 5+43,height+46)
    if height< 100:
        start=True
    if start:
        obs1 = (obs1[0]-speed, obs1[1])
        obs2 = (obs2[0]-speed, obs2[1])
        obs3 = (obs3[0]-speed, obs3[1])
        fly_obj = (fly_obj[0]-speed, fly_obj[1])
        obs1_cub = (obs1[0], obs1[1], obs1[0]+obast1.size[0],obs1[1]+obast1.size[1])
        obs2_cub = (obs2[0], obs2[1], obs2[0]+obast2.size[0],obs2[1]+obast2.size[1])
        obs3_cub = (obs3[0], obs3[1], obs3[0]+obast3.size[0],obs3[1]+obast3.size[1])

        fly_cub = (fly_obj[0], fly_obj[1], fly_obj[0]+fly_1.size[0], fly_obj[1]+fly_1.size[1])
        if not lock:
            bg = (bg[0]-speed, bg[1])
            if bg[0]<=-(600):
                lock = 1
        if -bg[0]>=600 and lock:
            bg1 = (bg1[0]-speed, bg1[1])
            bg = (bg[0]-speed, bg[1])
            if -bg1[0]>=600:bg = (600,150)
        if -bg1[0]>=600 and lock:
            bg = (bg[0]-speed, bg1[1])
            bg1 = (bg1[0]-speed, bg1[1])
            if -bg[0]>=600:bg1 = (600,150)

        if obs1_cub[0]<=player_stading_cub[2]-10<=obs1_cub[2] and obs1_cub[1]<=player_stading_cub[3]-10<=obs1_cub[3]-5:
            start=False
            state = player_frame_4
        if obs2_cub[0]<=player_stading_cub[2]-10<=obs2_cub[2] and obs2_cub[1]<=player_stading_cub[3]-10<=obs2_cub[3]-5:
            start=False
            state = player_frame_4
        if obs3_cub[0]<=player_stading_cub[2]-10<=obs3_cub[2] and obs3_cub[1]<=player_stading_cub[3]-10<=obs3_cub[3]-5:
            start=False
            state = player_frame_4
        if fly_cub[0]<=player_stading_cub[2]-10<=fly_cub[2] and fly_cub[1]<=player_stading_cub[3]-10<=fly_cub[3]-5:
            start=False
            state= player_frame_4
        if not start:
            #print("Timer resumed.")
            active = True
        elif start:
            #print("Timer paused.")
            active = False

        if active:
            time_elapsed += clock.get_time()
        print(time_elapsed//20)

    pygame.display.update()
    clock.tick(120)