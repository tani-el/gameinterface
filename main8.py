from calibration import *
import calibGame as py
# import Num_Game   # merge_game으로 변경시켜놓음
import threading
import queue
import merge_game as Num_Game
# for test


# Num_Game 을 돌리기 위한 코드-------------------
game = Num_Game.NumGame(pos_x, pos_y)

# Create a queue for communication between threads
coord_queue = queue.Queue()

def run_game():
    while True:
        # Check if there are updated coordinates in the queue
        if not coord_queue.empty():
            pos_x, pos_y = coord_queue.get()
            game.set_target(pos_x, pos_y)
        
        game.run()
    # 점수판 함수 데모 제작중
'''    
        Num_Game.result_screen.fill((255, 255, 255))
        #Num game 점수
        Num_Game.result_screen.blit(Num_Game.result_text, Num_Game.result_rect)
        #T-RexRuner 점수(임시)
        #Dino_Run.result_screen.blit(Dino_Run.result_text, Dino_Run.result_rect)
        pygame.display.flip()
        # 결과 화면 유지
        running = True
        while running:
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    running = False
'''




# Create and start the game thread
game_thread = threading.Thread(target=run_game)
game_thread.start()
#-----------------------------------------------





# Signal the game thread to exit
# Num_Game 을 돌리기 위한 코드
game_thread.join()




capture.release()
cv.destroyAllWindows()
exit()