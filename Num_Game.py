import pygame
import random
import threading
import time

window = pygame.display.set_mode((800, 800))

class NumGame:
    def __init__(self, x, y):
        # 게임 초기화
        pygame.init()

        # 게임 화면 설정
        self.screen = pygame.display.set_mode((450, 450))
        pygame.display.set_caption("숫자 클릭 게임")

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

        # 결과 화면 설정
        self.result_screen = pygame.display.set_mode((450, 450))
        pygame.display.set_caption("result")

        # 결과 텍스트 생성
        self.result_text = self.font.render("score: ", True, (0, 0, 0))  # 최종 점수 출력
        self.result_rect = self.result_text.get_rect(center=(225, 225))

        self.running = True

    def update(self):
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                self.running = False
            elif event.type == pygame.MOUSEBUTTONDOWN:
                if event.button == 1:  # 왼쪽 마우스 버튼 클릭
                    pos = pygame.mouse.get_pos()
                    clicked_number = None
                    for number, rect in zip(self.numbers, self.number_rects):
                        if rect.collidepoint(pos):
                            if clicked_number is None or number > clicked_number:
                                clicked_number = number  # 숫자 갱신
                                self.init += 1
                                if self.max_number == number:
                                    self.is_check = True
                    if clicked_number is not None:
                        if self.is_check:  # 최댓값이 맞다면 점수를 높임
                            self.score += 10
                            self.is_check = False
                            self.numbers = random.sample(range(1, 101), 9)
                            self.max_number = max(self.numbers)
                        else:
                            self.score -= 10

        # 게임 종료 조건 확인
        if self.init == 10:
            self.game_count += 1
            if self.game_count == 10:
                self.running = False
                self.result_text = self.font.render("score: " + str(self.score), True, (0, 0, 0))  # 최종 점수 출력

    def draw(self):
        # 화면 초기화
        self.screen.fill((0, 0, 0))

        # 숫자 그리기
        self.number_rects = []
        for i, number in enumerate(self.numbers):
            rect = pygame.Rect((i % 3) * 130 + 50, (i // 3) * 130 + 50, 100, 100)
            pygame.draw.rect(self.screen, (0, 0, 255), rect)
            number_text = self.font.render(str(number), True, (255, 255, 255))
            number_rect = number_text.get_rect(center=rect.center)
            self.screen.blit(number_text, number_rect)
            self.number_rects.append(rect)

        # 점수 그리기
        score_text = self.font.render("Score: " + str(self.score), True, (0, 0, 0))
        self.screen.blit(score_text, (10, 10))

        # 커서 그리기
        pos_x, pos_y = self.x, self.y
        pygame.draw.circle(window, (255, 255, 255), (pos_x, pos_y), 10)

        # 화면 업데이트
        pygame.display.flip()

    # x, y 좌표
    def set_target(self, x, y):
        self.x, self.y = x, y

    def run(self):
        while self.running:
            self.update()
            self.draw()
            # print(self.x, self.y)
        # 결과 화면 업데이트
        self.result_screen.fill((255, 255, 255))
        self.result_screen.blit(self.result_text, self.result_rect)
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
# game = NumGame()
# game.run()