import pygame
import random
import threading
import time

window = pygame.display.set_mode((800, 800))

class NumGame:
    def __init__(self, x, y):
        # 게임 초기화
        pygame.init()

        # 게임 화면 초기화
        self.screen = pygame.display.set_mode((450, 900))
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
        self.Num_screen = pygame.display.set_mode((450, 900))
        pygame.display.set_caption("숫자클릭게임")

        # 결과 텍스트 생성
        self.result_text = self.font.render("score: ", True, (0, 0, 0))  # 최종 점수 출력
        self.result_rect = self.result_text.get_rect(center=(225, 225))

        self.running = True
        self.clicked_indices = []  # 클릭된 버튼의 인덱스 리스트



        # jump
        #-------------------------
        self.box_x = 100
        self.box_y = 800
        self.box_width = 50
        self.box_height = 50
        self.jump_height = 160
        self.is_jumping = False
        self.jump_count = 0
        # self.gravity = 2
        self.clock = pygame.time.Clock()


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
                            self.score += 10
                            self.is_check = False
                            self.numbers = random.sample(range(1, 101), 9)
                            self.max_number = max(self.numbers)
                        else:
                            self.score -= 10

            elif event.type == pygame.MOUSEBUTTONUP:
                if event.button == 1:  # 왼쪽 마우스 버튼 뗐을 때
                    self.clicked_indices = []  # 클릭된 버튼의 인덱스 리스트 초기화
            

            elif event.type == pygame.KEYDOWN:
                if event.key == pygame.K_ESCAPE:
                    pygame.quit()
        
        # 점프
        if self.is_jumping:
            if self.jump_count >= -self.jump_height:
                self.box_y -= int((self.jump_count * abs(self.jump_count)) * 0.0005)
                self.jump_count -= 3
            else:
                self.is_jumping = False
                self.jump_count = 0
                self.box_y = 800

        # 게임 종료 조건 확인
        if self.init == 10:
            self.game_count += 1
            if self.game_count == 10:
                self.running = False
                self.result_text = self.font.render("score: " + str(self.score), True, (0, 0, 0))  # 최종 점수 출력


    def jump(self):
        if not self.is_jumping:
            self.is_jumping = True
            self.jump_count = self.jump_height


    def draw(self):
        # 화면 초기화
        self.screen.fill((0, 0, 0))

        # 숫자 그리기
        self.number_rects = []
        for i, number in enumerate(self.numbers):
            rect = pygame.Rect((i % 3) * 130 + 50, (i // 3) * 130 + 50, 100, 100)
            if i in self.clicked_indices:
                pygame.draw.rect(self.screen, (255, 0, 0), rect)  # 클릭된 버튼은 빨간색으로 표시
            else:
                pygame.draw.rect(self.screen, (0, 0, 255), rect)  # 클릭되지 않은 버튼은 파란색으로 표시
            number_text = self.font.render(str(number), True, (255, 255, 255))
            number_rect = number_text.get_rect(center=rect.center)
            self.screen.blit(number_text, number_rect)
            self.number_rects.append(rect)

        # 커서 그리기
        pos_x, pos_y = self.x, self.y
        pygame.draw.circle(window, (255, 255, 255), (pos_x, pos_y), 10)
        pygame.mouse.set_pos(pos_x, pos_y)

        # 점수 그리기
        score_text = self.font.render("Score: " + str(self.score), True, (255, 255, 255))
        self.screen.blit(score_text, (10, 10))

        # 캐릭터 그리기
        # self.screen.fill((255, 255, 255))
        pygame.draw.rect(self.screen, (255, 0, 0), (self.box_x, self.box_y, self.box_width, self.box_height))


        # 화면 업데이트
        pygame.display.flip()

    # x, y 좌표
    def set_target(self, x, y):
        self.x, self.y = x, y


    def run(self):
        while self.running:
            self.handle_events()
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
game = NumGame(200, 200)
game.run()