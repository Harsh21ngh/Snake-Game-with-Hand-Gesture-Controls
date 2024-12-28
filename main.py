import pygame
import random
import cv2
import mediapipe as mp

# Initialize pygame
pygame.init()

# Constants
WIDTH, HEIGHT = 600, 600
RED = (255, 0, 0)
GREEN = (0, 255, 0)
BLUE = (0, 0, 255)
HYBRID = (23, 234, 223)
SNAKE_SIZE = 30
FPS = 10  # Lowered FPS for slower updates
SNAKE_MOVE_STEP = 10  # Reduced movement step size

# Initialize Pygame window
Snake_Window = pygame.display.set_mode([WIDTH, HEIGHT])
pygame.display.set_caption("Snake Game")
Clock = pygame.time.Clock()
Score_font = pygame.font.SysFont(None, 30)

# Functions for game handling
def write_score(text, color, x, y):
    screen_text = Score_font.render(text, True, color)
    Snake_Window.blit(screen_text, [x, y])

def create_body(Snake_Window, color, Snake_list):
    for x, y in Snake_list:
        pygame.draw.rect(Snake_Window, color, [x, y, SNAKE_SIZE, SNAKE_SIZE])

def reset_game():
    """ Resets the game variables. """
    return 300, 300, 0, random.randint(2, 19) * 30, random.randint(2, 19) * 30, 0, 0, 1, []

def Game_loop():
    """ Main game loop. """
    game_exit = False
    game_over = False
    high_score = 0

    # Initialize game variables
    Snake_head_x, Snake_head_y, Score, Food_x, Food_y, Speed_x, Speed_y, Snake_length, Snake_list = reset_game()
    
    # Setup webcam
    cam = cv2.VideoCapture(0)
    if not cam.isOpened():
        print("Error: Could not access the webcam.")
        return

    mp_hands = mp.solutions.hands
    hands = mp_hands.Hands()
    mp_drawing = mp.solutions.drawing_utils  # Utility for drawing landmarks

    while not game_exit:
        if game_over:
            Snake_Window.fill(RED)
            write_score(f"Game Over! Score: {Score}. High Score: {high_score}", GREEN, 10, 100)
            write_score("Press Enter to Restart or Q to Quit", GREEN, 10, 150)
            pygame.display.update()

            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    game_exit = True
                if event.type == pygame.KEYDOWN:
                    if event.key == pygame.K_RETURN:  # Restart game
                        Snake_head_x, Snake_head_y, Score, Food_x, Food_y, Speed_x, Speed_y, Snake_length, Snake_list = reset_game()
                        game_over = False
                    elif event.key == pygame.K_q:  # Quit game
                        game_exit = True

        else:
            success, img = cam.read()
            if not success:
                print("Warning: Camera feed lost. Exiting game.")
                break

            # Convert image to RGB
            image = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            image = cv2.flip(image, 1)  # Flip for better user experience
            output = hands.process(image)

            if output.multi_hand_landmarks:
                for landmarks in output.multi_hand_landmarks:
                    # Draw landmarks on the image
                    mp_drawing.draw_landmarks(img, landmarks, mp_hands.HAND_CONNECTIONS)

                    for id, pos in enumerate(landmarks.landmark):
                        height, width, _ = image.shape
                        pos_x, pos_y = int(width * pos.x), int(height * pos.y)
                        if id == 8:  # Using index finger for movement
                            if pos_x > 350:
                                Speed_x, Speed_y = SNAKE_MOVE_STEP, 0
                            elif pos_x < 250:
                                Speed_x, Speed_y = -SNAKE_MOVE_STEP, 0
                            elif pos_y < 250:
                                Speed_x, Speed_y = 0, -SNAKE_MOVE_STEP
                            elif pos_y > 350:
                                Speed_x, Speed_y = 0, SNAKE_MOVE_STEP

            # Process events
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    game_exit = True

            # Snake movement logic
            Snake_head_x += Speed_x
            Snake_head_y += Speed_y
            Snake_list.append([Snake_head_x, Snake_head_y])

            # Keep snake list length consistent
            if len(Snake_list) > Snake_length:
                del Snake_list[0]

            # Check collision with food
            if abs(Snake_head_x - Food_x) < SNAKE_SIZE and abs(Snake_head_y - Food_y) < SNAKE_SIZE:
                Food_x = random.randint(2, 19) * 30
                Food_y = random.randint(2, 19) * 30
                Score += 5
                Snake_length += 1

            # Check collision with boundaries
            if Snake_head_x < 0 or Snake_head_x >= WIDTH or Snake_head_y < 50 or Snake_head_y >= HEIGHT:
                game_over = True

            # Check collision with itself
            if [Snake_head_x, Snake_head_y] in Snake_list[:-1]:
                game_over = True

            # Update high score
            if Score > high_score:
                high_score = Score

            # Draw everything
            Snake_Window.fill(BLUE)
            write_score(f"Score: {Score} | High Score: {high_score}", RED, 10, 10)
            create_body(Snake_Window, RED, Snake_list)
            pygame.draw.rect(Snake_Window, HYBRID, [0, 0, WIDTH, 50])
            pygame.draw.rect(Snake_Window, GREEN, [Food_x, Food_y, SNAKE_SIZE, SNAKE_SIZE])

            pygame.display.update()

            # Display webcam feed with hand tracking
            cv2.imshow("Snake Game - Hand Tracking", img)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                game_exit = True  # Exit only when 'q' is pressed explicitly
            Clock.tick(FPS)

    cam.release()
    cv2.destroyAllWindows()

Game_loop()
