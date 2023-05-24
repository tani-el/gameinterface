import pygame
import random

# 게임 초기화
pygame.init()

# 게임 화면 설정
screen = pygame.display.set_mode((450, 450))
pygame.display.set_caption("숫자 클릭 게임")

# 게임 변수
score = 0

# 폰트 설정
font = pygame.font.Font(None, 36)

# 숫자 생성
def generate_numbers():
    return random.sample(range(1, 101), 9)

numbers = generate_numbers()
max_number = max(numbers)

is_check=False

class NumGame():
    # 게임 루프
    running = True
    while running:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
            elif event.type == pygame.MOUSEBUTTONDOWN:
                if event.button == 1:  # 왼쪽 마우스 버튼 클릭
                    pos = pygame.mouse.get_pos()
                    clicked_number = None
                    for number, rect in zip(numbers, number_rects):
                        if rect.collidepoint(pos):
                            if clicked_number is None or number > clicked_number:
                                clicked_number = number  # 숫자 갱신
                                if max_number == number:
                                    is_check = True
                    if clicked_number is not None:
                        if is_check == True: #최댓값이 맞다면 점수를 높임
                            score += 10
                            is_check = False
                    numbers = generate_numbers()  # 숫자 갱신

        # 화면 초기화
        screen.fill((255, 255, 255))

        # 숫자 그리기
        number_rects = []
        for i, number in enumerate(numbers):
            rect = pygame.Rect((i % 3) * 130 + 50, (i // 3) * 130 + 50, 100, 100)
            pygame.draw.rect(screen, (0, 0, 255), rect)
            number_text = font.render(str(number), True, (255, 255, 255))
            number_rect = number_text.get_rect(center=rect.center)
            screen.blit(number_text, number_rect)
            number_rects.append(rect)

        # 점수 그리기
        score_text = font.render("Score: " + str(score), True, (0, 0, 0))
        screen.blit(score_text, (10, 10))

        # 화면 업데이트
        pygame.display.flip()

# 게임 종료
pygame.quit()