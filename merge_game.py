import Num_Game
import Runner
import threading
import pygame

# 아직 수정중 작동하지 않음

dino_game = Runner.DinoGame()
num_game = Num_Game.NumGame(200, 200)

# 게임 1 초기화
def initialize_game_1():
    # 게임 1 초기화 로직 작성
    dino_game.__init__

# 게임 1 업데이트
def update_game_1():
    # 게임 1 업데이트 로직 작성
    pass

# 게임 1 그리기
def draw_game_1(screen):
    # 게임 1 그리기 로직 작성
    dino_game.run()

# 게임 2 초기화
def initialize_game_2():
    # 게임 2 초기화 로직 작성
    num_game.set_target(200, 200)

# 게임 2 업데이트
def update_game_2():
    # 게임 2 업데이트 로직 작성
    num_game.update()
    num_game.draw()

# 게임 2 그리기
def draw_game_2(screen):
    # 게임 2 그리기 로직 작성
    pass

# pygame 초기화
pygame.init()

# 윈도우 크기 설정
window_width = 1400
window_height = 1400
window_size = (window_width, window_height)

# 창 생성
screen = pygame.display.set_mode(window_size)

# 게임 1 초기화
initialize_game_1()

# 게임 2 초기화
initialize_game_2()


def game_thread(game):
    game.run()

# 게임 루프
running = True
while running:
    # 이벤트 처리
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            running = False

    # 게임 1 업데이트
    update_game_1()

    # 게임 2 업데이트
    update_game_2()

    # 창을 가로로 나누는 위치 계산
    divider_pos = window_width // 2

    # 게임 1 그리기
    game1_surface = screen.subsurface((0, 0, divider_pos, window_height//2))
    # draw_game_1(game1_surface)

    # 게임 2 그리기
    game2_surface = screen.subsurface((0, window_height//2, 0, window_height//2))
    # draw_game_2(game2_surface)

     # 스레드 생성 및 실행
    thread1 = threading.Thread(draw_game_1(game1_surface))
    thread2 = threading.Thread(draw_game_2(game2_surface))
    thread1.start()
    thread2.start()

    # 스레드 종료 대기
    thread1.join()
    thread2.join()



    # 화면 업데이트
    pygame.display.flip()

# pygame 종료
pygame.quit()
