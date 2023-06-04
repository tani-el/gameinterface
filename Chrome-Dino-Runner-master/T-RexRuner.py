from random import randrange as rnd
from itertools import cycle
from random import choice
from PIL import  Image
import pygame
import time
import os


# clock = pygame.time.Clock()
# FPS = 30
# active = True
# time_elapsed = 0



class DinoGame:
    def __init__(self):
        pygame.init()
        self.clock = pygame.time.Clock()
        self.FPS = 30
        self.active = True
        self.time_elapsed = 0
        self.speed = 15

        # 현재 스크립트 파일의 경로를 가져옴
        script_dir = os.path.dirname(os.path.abspath(__file__))

        # 같은 폴더에 있는 파일의 경로를 생성
        file_path = os.path.join(script_dir, 'resources.png')

        # extracting game items and characters form the resource.png image.
        self.player_init = Image.open(file_path).crop((77, 5, 163, 96)).convert("RGBA")
        self.player_init = self.player_init.resize(list(map(lambda x: x // 2, self.player_init.size)))

        self.player_frame_1 = Image.open(file_path).crop((1679, 2, 1765, 95)).convert("RGBA")
        self.player_frame_1 = self.player_frame_1.resize(list(map(lambda x: x // 2, self.player_frame_1.size)))

        self.player_frame_2 = Image.open(file_path).crop((1767, 2, 1853, 95)).convert("RGBA")
        self.player_frame_2 = self.player_frame_2.resize(list(map(lambda x: x // 2, self.player_frame_2.size)))

        self.player_frame_3 = Image.open(file_path).crop((1855, 2, 1941, 95)).convert("RGBA")
        self.player_frame_3 = self.player_frame_3.resize(list(map(lambda x: x // 2, self.player_frame_3.size)))

        self.player_frame_31 = Image.open(file_path).crop((1943, 2, 2029, 95)).convert("RGBA")
        self.player_frame_31 = self.player_frame_31.resize(list(map(lambda x: x // 2, self.player_frame_31.size)))

        self.player_frame_4 = Image.open(file_path).crop((2030, 2, 2117, 95)).convert("RGBA")
        self.player_frame_4 = self.player_frame_4.resize(list(map(lambda x: x // 2, self.player_frame_4.size)))

        self.player_frame_5 = Image.open(file_path).crop((2207, 2, 2323, 95)).convert("RGBA")
        self.player_frame_5 = self.player_frame_5.resize(list(map(lambda x: x // 2, self.player_frame_5.size)))

        self.player_frame_6 = Image.open(file_path).crop((2324, 2, 2441, 95)).convert("RGBA")
        self.player_frame_6 = self.player_frame_6.resize(list(map(lambda x: x // 2, self.player_frame_6.size)))

        self.cloud = Image.open(file_path).crop((166, 2, 257, 29)).convert("RGBA")
        self.cloud = self.cloud.resize(list(map(lambda x: x // 2, self.cloud.size)))

        self.fly_1 = Image.open(file_path).crop((258, 10, 353, 80)).convert("RGBA")
        self.fly_1 = self.fly_1.resize(list(map(lambda x:x//2 , self.fly_1.size)))

        self.fly_2 = Image.open(file_path).crop((354,1,443,62)).convert("RGBA")
        self.fly_2 = self.fly_2.resize(list(map(lambda x:x//2 , self.fly_2.size)))

        self.ground = Image.open(file_path).crop((2,102,2401,127)).convert("RGBA")
        self.ground = self.ground.resize(list(map(lambda x:x//2 , self.ground.size)))

        self.obstacle1 = Image.open(file_path).crop((446,2,479,71)).convert("RGBA")
        self.obstacle1 = self.obstacle1.resize(list(map(lambda x:x//2 , self.obstacle1.size)))

        self.obstacle2 = Image.open(file_path).crop((446,2,547,71)).convert("RGBA")
        self.obstacle2 = self.obstacle2.resize(list(map(lambda x:x//2 , self.obstacle2.size)))

        self.obstacle3 = Image.open(file_path).crop((446,2,581,71)).convert("RGBA")
        self.obstacle3 = self.obstacle3.resize(list(map(lambda x:x//2 , self.obstacle3.size)))

        self.obstacle4 = Image.open(file_path).crop((653,2,701,101)).convert("RGBA")
        self.obstacle4 = self.obstacle4.resize(list(map(lambda x:x//2 , self.obstacle4.size)))

        self.obstacle5 = Image.open(file_path).crop((653,2,701,101)).convert("RGBA")
        self.obstacle5 = self.obstacle5.resize(list(map(lambda x:x//2 , self.obstacle5.size)))

        self.obstacle5 = Image.open(file_path).crop((653,2,749,101)).convert("RGBA")
        self.obstacle5 = self.obstacle5.resize(list(map(lambda x:x//2 , self.obstacle5.size)))

        self.obstacle6 = Image.open(file_path).crop((851,2,950,101)).convert("RGBA")
        self.obstacle6 = self.obstacle6.resize(list(map(lambda x:x//2 , self.obstacle6.size)))

        self.speed_identifier = lambda x: 2 if x >= 30 else 8 if x < 8 else 5
        self.cust_speed = self.speed_identifier(self.speed)
        self.running = cycle([self.player_frame_3]*self.cust_speed+[self.player_frame_31]*self.cust_speed)
        self.crouch = cycle([self.player_frame_5]*self.cust_speed+ [self.player_frame_6]*self.cust_speed)
        self.crouch_scope = [self.player_frame_5]+[self.player_frame_6]
        self.obstacles = [self.obstacle1,self.obstacle2, self.obstacle3,self.obstacle4,self.obstacle5,self.obstacle6]
        self.flys = cycle([self.fly_1]*self.cust_speed+[self.fly_2]*self.cust_speed)


        self.gameDisplay = pygame.display.set_mode((600,200))
        pygame.display.set_caption('T-Rex Runner')
        self.clock = pygame.time.Clock()
        self.state = self.player_frame_1
        self.crashed = False
        self.lock = False
        self.bg = (0, 150)
        self.bg1 = (600,150)
        self.start = False
        self.height = 110
        self.jumping = False
        self.slow_motion = False
        self.c1 = (rnd(30, 600), rnd(0, 100))
        self.c2 = (rnd(50,600), rnd(0, 100))
        self.c3 = (rnd(30,700), rnd(0, 100))
        self.c4 = (rnd(30,600),rnd(0, 100))
        self.obs1 = (rnd(600, 600+500-400), 130)
        self.obs2 = (rnd(600+100+500+1000, 1200+500-400+1000), 130)
        self.obs3 = (rnd(1700+3000, 2000-200+3000), 130)
        self.fly_obj = (rnd(600, 600+500-400), 115)

        self.obast1 = choice(self.obstacles)
        if self.obast1 in [self.obstacle4, self.obstacle5, self.obstacle6]:self.obs1 = (self.obs1[0], 115)
        self.obast2 = choice(self.obstacles)
        if self.obast2 in [self.obstacle4, self.obstacle5, self.obstacle6]:self.obs2 = (self.obs2[0], 115)
        self.obast3 = choice(self.obstacles)
        if self.obast3 in [self.obstacle4, self.obstacle5, self.obstacle6]:self.obs3 = (self.obs3[0], 115)


    def run(self):
        while self.active:
            self.game_loop()

    def game_loop(self):
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                self.active = False
            if event.type == pygame.KEYDOWN:
                if event.key == pygame.K_DOWN:
                    self.slow_motion = True
                    self.state = self.crouch
                if event.key == pygame.K_UP:
                    if self.height >= 110:
                        self.jumping = True
            if event.type == pygame.KEYUP:
                self.slow_motion = False
                if event.key == pygame.K_DOWN:
                    self.state = self.running

        # ...
        self.player = self.state if type(self.state) != cycle else next(self.state)

        self.gameDisplay.blit(
            pygame.image.fromstring(self.cloud.tobytes(), self.cloud.size, 'RGBA'),
            self.c1
        )
        self.gameDisplay.blit(
            pygame.image.fromstring(self.cloud.tobytes(), self.cloud.size, 'RGBA'),
            self.c2
        )
        self.gameDisplay.blit(
            pygame.image.fromstring(self.cloud.tobytes(), self.cloud.size, 'RGBA'),
            self.c3
        )
        self.gameDisplay.blit(
            pygame.image.fromstring(self.cloud.tobytes(), self.cloud.size, 'RGBA'),
            self.c4
        )


        self.flyer = self.flys if type(self.flys) != cycle else next(self.flys)

        self.c1 = (self.c1[0]-1, self.c1[1])
        self.c2 = (self.c2[0]-1, self.c2[1])
        self.c3 = (self.c3[0]-1, self.c3[1])
        self.c4 = (self.c4[0]-1, self.c4[1])

        if self.c1[0]<= -50:
            self.c1 = (640, self.c1[1])
        if self.c2[0]<= -50:
            self.c2 = (700, self.c2[1])
        if self.c3[0]<= -50:
            self.c3 = (600, self.c3[1])
        if self.c4[0]<= -50:
            self.c4 = (800, self.c4[1])
        # ...

        self.gameDisplay.blit(
            pygame.image.fromstring(self.ground.tobytes(), self.ground.size, 'RGBA'),
            self.bg
        )
        self.gameDisplay.blit(
            pygame.image.fromstring(self.ground.tobytes(), self.ground.size, 'RGBA'),
            self.bg1
        )
        if self.jumping:
            if self.height>=110-100:
                self.height -= 4
            if self.height <= 110-100:
                self.jumping = False
        if self.height<110 and not self.jumping:
            if self.slow_motion == True:
                self.height += 1.5
            else:self.height += 3
        self.player = self.gameDisplay.blit(pygame.image.fromstring(self.player.tobytes(), self.player.size, 'RGBA'), (5,self.height))

        self.gameDisplay.blit(pygame.image.fromstring(self.obast1.tobytes(), self.obast1.size, 'RGBA'), self.obs1)
        self.gameDisplay.blit(pygame.image.fromstring(self.obast2.tobytes(), self.obast2.size, 'RGBA'), self.obs2)
        self.gameDisplay.blit(pygame.image.fromstring(self.obast3.tobytes(), self.obast3.size, 'RGBA'), self.obs3)

        self.gameDisplay.blit(pygame.image.fromstring(self.fly_2.tobytes(), self.fly_2.size, 'RGBA'), self.fly_obj)
        if self.obs1[0]<=-2500:
            self.obs1 = (rnd(600, 600+500), 130)
            self.obast1 = choice(self.obstacles)
            if self.obast1 in [self.obstacle4, self.obstacle5, self.obstacle6]:self.obs1 = (self.obs1[0], 115)
        if self.obs2[0]<=-3000:
            self.obs2 = (rnd(600+100+500, 1200+500), 130)
            self.obast2 = choice(self.obstacles)
            if self.obast2 in [self.obstacle4, self.obstacle5, self.obstacle6]:self.obs2 = (self.obs2[0], 115)
        if self.obs3[0]<=-3500:
            self.obs3 = (rnd(1700, 2000), 130) 
            self.obast3 = choice(self.obstacles) 
            if self.obast3 in [self.obstacle4, self.obstacle5, self.obstacle6]:self.obs3 = (self.obs3[0], 115)
        if self.fly_obj[0] <=-3600:
            self.fly_obj = (rnd(600, 700), 60)
        player_stading_cub = (5, self.height, 5+43,self.height+46)

        if self.height< 100:
            self.start=True

        if self.start:
            self.obs1 = (self.obs1[0]-self.speed, self.obs1[1])
            self.obs2 = (self.obs2[0]-self.speed, self.obs2[1])
            self.obs3 = (self.obs3[0]-self.speed, self.obs3[1])
            self.fly_obj = (self.fly_obj[0]-self.speed, self.fly_obj[1])
            self.obs1_cub = (self.obs1[0], self.obs1[1], self.obs1[0]+self.obast1.size[0],self.obs1[1]+self.obast1.size[1])
            self.obs2_cub = (self.obs2[0], self.obs2[1], self.obs2[0]+self.obast2.size[0],self.obs2[1]+self.obast2.size[1])
            self.obs3_cub = (self.obs3[0], self.obs3[1], self.obs3[0]+self.obast3.size[0],self.obs3[1]+self.obast3.size[1])

            self.fly_cub = (self.fly_obj[0], self.fly_obj[1], self.fly_obj[0]+self.fly_1.size[0], self.fly_obj[1]+self.fly_1.size[1])
            if not self.lock:
                self.bg = (self.bg[0]-self.speed, self.bg[1])
                if self.bg[0]<=-(600):
                    self.lock = 1
            if -self.bg[0]>=600 and self.lock:
                self.bg1 = (self.bg1[0]-self.speed, self.bg1[1])
                self.bg = (self.bg[0]-self.speed, self.bg[1])
                if -self.bg1[0]>=600:self.bg = (600,150)
            if -self.bg1[0]>=600 and self.lock:
                self.bg = (self.bg[0]-self.speed, self.bg1[1])
                self.bg1 = (self.bg1[0]-self.speed, self.bg1[1])
                if -self.bg[0]>=600:self.bg1 = (600,150)

            if self.obs1_cub[0]<=player_stading_cub[2]-10<=self.obs1_cub[2] and self.obs1_cub[1]<=player_stading_cub[3]-10<=self.obs1_cub[3]-5:
                self.start=False
                self.state = self.player_frame_4
            if self.obs2_cub[0]<=player_stading_cub[2]-10<=self.obs2_cub[2] and self.obs2_cub[1]<=player_stading_cub[3]-10<=self.obs2_cub[3]-5:
                self.start=False
                self.state = self.player_frame_4
            if self.obs3_cub[0]<=player_stading_cub[2]-10<=self.obs3_cub[2] and self.obs3_cub[1]<=player_stading_cub[3]-10<=self.obs3_cub[3]-5:
                self.start=False
                self.state = self.player_frame_4
            if self.fly_cub[0]<=player_stading_cub[2]-10<=self.fly_cub[2] and self.fly_cub[1]<=player_stading_cub[3]-10<=self.fly_cub[3]-5:
                self.start=False
                self.state= self.player_frame_4
            if not self.start:
                #print("Timer resumed.")
                active = True
            elif self.start:
                #print("Timer paused.")
                active = False
            # ...

            player = self.state if type(self.state) != cycle else next(self.state)
            player_rect = pygame.Rect(5, self.height, 5 + 43, self.height + 46)
            self.gameDisplay.blit(
                pygame.image.fromstring(player.tobytes(), player.size, 'RGBA'),
                player_rect
            )
            # if active:
            #     time_elapsed += clock.get_time()
            #     print(time_elapsed//20)
            # ...
            
            pygame.display.update()
            self.gameDisplay.fill((0, 0, 0))
            self.clock.tick(self.FPS)

if __name__ == '__main__':
    dino_game = DinoGame()
    dino_game.run()
    pygame.quit()

