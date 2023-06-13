import pygame
import random
import threading
import time
import os
from PIL import  Image

window = pygame.display.set_mode((450, 800))

class NumGame:
    def __init__(self, x, y):
        # 게임 초기화
        pygame.init()

        # 게임 화면 초기화
        self.screen_width = 450
        self.screen_height = 900
        self.screen = pygame.display.set_mode((450, 800))
        pygame.display.set_caption("Loading...")

        # x, y 좌표
        self.x, self.y = x, y

        # 게임 변수
        self.score = 0

        # 폰트 설정
        self.font = pygame.font.Font(None, 36)

        # 숫자 생성
        self.numbers = random.sample(range(1, 101), 9)
        self.max_number = max(self.numbers)

        self.is_check = False
        self.init = 0
        self.game_count = 0  # 게임 횟수 변수

        # 게임 화면 설정
        self.Num_screen = pygame.display.set_mode((450, 800))
        pygame.display.set_caption("숫자클릭게임")

        # 결과 텍스트 생성
        self.result_text = self.font.render("score: ", True, (0, 0, 0))  # 최종 점수 출력
        self.result_rect = self.result_text.get_rect(center=(225, 225))

        self.running = True
        self.clicked_indices = []  # 클릭된 버튼의 인덱스 리스트



        # jump
        # #-------------------------
        self.floor = 700

        self.box_x = 100
        self.box_y = self.floor
        self.box_width = 50
        self.box_height = 50
        self.jump_height = 160
        self.is_jumping = False
        self.jump_count = 0
        # self.gravity = 2
        self.clock = pygame.time.Clock()

         # 현재 스크립트 파일의 경로를 가져옴
        script_dir = os.path.dirname(os.path.abspath(__file__))

        # 같은 폴더에 있는 파일의 경로를 생성
        file_path = os.path.join(script_dir, 'resources.png')
        file_path_Circle = os.path.join(script_dir, 'Circle.png')
        file_path_X = os.path.join(script_dir, 'X.png')

        # extracting game items and characters form the resource.png image.
        self.player_init = Image.open(file_path).crop((77, 5, 163, 96)).convert("RGBA")
        self.player_init = self.player_init.resize(list(map(lambda x: x // 2, self.player_init.size)))


        # 이미지 표시 여부
        self.show_image_circle = False
        self.show_image_Red_X = False

        self.Circle = pygame.image.load(file_path_Circle)
        self.Circle = pygame.transform.scale(self.Circle, (100, 100))
        self.Red_X = pygame.image.load(file_path_X)
        self.Red_X = pygame.transform.scale(self.Red_X, (100, 100))

        #장애물
        self.obstacle_x = self.screen_width
        self.obstacle_y = self.floor
        self.obstacle_width = 50
        self.obstacle_height = 50
        self.obstacle_speed = random.randint(4, 7)  # 랜덤한 속도 설정
        self.obstacle_timer = time.time() + random.uniform(4, 6)

        self.collision_occurred = False

    def handle_events(self):
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                self.quit()
            elif event.type == pygame.KEYDOWN:
                if event.key == pygame.K_ESCAPE:
                    self.quit()
                elif event.key == pygame.K_SPACE:
                    self.jump()


    def update(self):
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                self.running = False
            elif event.type == pygame.MOUSEBUTTONDOWN:
                if event.button == 1:  # 왼쪽 마우스 버튼 클릭
                    pos = pygame.mouse.get_pos()
                    clicked_number = None
                    for i, rect in enumerate(self.number_rects):
                        if rect.collidepoint(pos):
                            if clicked_number is None or self.numbers[i] > clicked_number:
                                clicked_number = self.numbers[i]  # 숫자 갱신
                                self.init += 1
                                if clicked_number == self.max_number:
                                    time.sleep(0.2)
                                    self.is_check = True
                                self.clicked_indices.append(i)  # 클릭된 버튼의 인덱스 추가
                    if clicked_number is not None:
                        if self.is_check:  # 최댓값이 맞다면 점수를 높임
                            self.show_image_circle = True
                            self.score += 10
                            self.is_check = False
                            self.numbers = random.sample(range(1, 101), 9)
                            self.max_number = max(self.numbers)
                        else:
                            self.score -= 10
                            self.show_image_Red_X = True

            elif event.type == pygame.MOUSEBUTTONUP:
                if event.button == 1:  # 왼쪽 마우스 버튼 뗐을 때
                    self.clicked_indices = []  # 클릭된 버튼의 인덱스 리스트 초기화
                    self.show_image_circle = False # 그림 초기화
                    self.show_image_Red_X = False
            
            if event.type == pygame.KEYDOWN:
                if event.key == pygame.K_ESCAPE:
                    pygame.quit()
        
        # 아래쪽 게임부분
        #------------------------
            # 점프 키 구현
            if event.type == pygame.KEYDOWN:
                if event.key == pygame.K_SPACE:
                    self.jump()
        
        # 점프
        if self.is_jumping:
            if self.jump_count >= -self.jump_height:
                self.box_y -= int((self.jump_count * abs(self.jump_count)) * 0.0005)
                self.jump_count -= 3
            else:
                self.is_jumping = False
                self.jump_count = 0
                self.box_y = self.floor

        # 장애물
        if time.time() >= self.obstacle_timer:
            self.obstacle_x = self.screen_width
            self.obstacle_y = self.floor
            self.obstacle_speed = random.randint(3, 5)
            self.obstacle_timer = time.time() + random.uniform(3, 5)

        self.obstacle_x -= self.obstacle_speed

        if self.check_collision() and not self.collision_occurred:
            self.score -= 5
            self.collision_occurred = True
        elif not self.check_collision():
            self.collision_occurred = False

        if self.score < 0:
            self.score = 0

        # 게임 종료 조건 확인
        if self.init == 10:
            self.game_count += 1
            if self.game_count == 10:
                self.running = False
                self.result_text = self.font.render("score: " + str(self.score), True, (0, 0, 0))  # 최종 점수 출력

    # 점프 
    def jump(self):
        if not self.is_jumping:
            self.is_jumping = True
            self.jump_count = self.jump_height
    # 충돌 체크
    def check_collision(self):
        box_rect = pygame.Rect(self.box_x, self.box_y, self.box_width, self.box_height)
        obstacle_rect = pygame.Rect(self.obstacle_x, self.obstacle_y, self.obstacle_width, self.obstacle_height)
        return box_rect.colliderect(obstacle_rect)


    def draw(self):
        # 화면 초기화
        self.screen.fill((0, 0, 0))

        # 숫자 그리기
        self.number_rects = []
        for i, number in enumerate(self.numbers):
            rect = pygame.Rect((i % 3) * 130 + 50, (i // 3) * 130 + 50, 100, 100)
            if i in self.clicked_indices:
                pygame.draw.rect(self.screen, (255, 255, 255), rect)  # 클릭된 버튼은 하얀색으로 표시
            else:
                pygame.draw.rect(self.screen, (0, 0, 255), rect)  # 클릭되지 않은 버튼은 파란색으로 표시
            number_text = self.font.render(str(number), True, (255, 255, 255))
            number_rect = number_text.get_rect(center=rect.center)
            self.screen.blit(number_text, number_rect)
            self.number_rects.append(rect)
            if i in self.clicked_indices:
                if self.show_image_circle:
                    self.screen.blit(self.Circle, (self.x-50, self.y-50)) # 맞으면 동그라미
                elif self.show_image_Red_X:
                    self.screen.blit(self.Red_X, (self.x-50, self.y-50)) # 틀리면 x

        # 커서 그리기
        pos_x, pos_y = self.x, self.y
        pygame.draw.circle(window, (255, 255, 255), (pos_x, pos_y), 10)
        pygame.mouse.set_pos(pos_x, pos_y)

        # 점수 그리기
        score_text = self.font.render("Score: " + str(self.score), True, (255, 255, 255))
        self.screen.blit(score_text, (10, 10))

        # 캐릭터 그리기
        pygame.draw.rect(self.screen, (255, 0, 0), (self.box_x, self.box_y, self.box_width, self.box_height))
        pygame.draw.rect(self.screen, (0, 0, 255), (self.obstacle_x, self.obstacle_y, self.obstacle_width, self.obstacle_height))
       

        # 화면 업데이트
        pygame.display.flip()

    # x, y 좌표
    def set_target(self, x, y):
        self.x, self.y = x, y


    def run(self):
        while self.running:
            
            self.update()
            
            self.draw()
            self.clock.tick(60)
            # print(self.x, self.y)
            
                    

        # 결과 화면 업데이트
        self.Num_screen.fill((255, 255, 255))
        self.Num_screen.blit(self.result_text, self.result_rect)
        pygame.display.flip()
        # 결과 화면 유지
        running = True
        while running:
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    running = False

        # 게임 종료
        pygame.quit()

# 게임 시작
# game = NumGame(200, 200)
# game.run()