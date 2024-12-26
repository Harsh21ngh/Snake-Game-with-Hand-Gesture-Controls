import cv2
import mediapipe as mp
import pygame
import random
import sys

# Initialize Pygame
pygame.init()

# Screen dimensions
WIDTH, HEIGHT = 800, 600
CELL_SIZE = 20

# Colors
WHITE = (255, 255, 255)
GREEN = (0, 128, 0)
RED = (255, 0, 0)

# Initialize screen
screen = pygame.display.set_mode((WIDTH, HEIGHT))
pygame.display.set_caption("Snake Game with Hand Gesture Control")

# Clock for controlling the game's frame rate
clock = pygame.time.Clock()

# Snake and food setup
snake_pos = [[100, 50], [90, 50], [80, 50]]
snake_direction = 'RIGHT'
change_to = snake_direction
food_pos = [random.randrange(1, (WIDTH // CELL_SIZE)) * CELL_SIZE,
            random.randrange(1, (HEIGHT // CELL_SIZE)) * CELL_SIZE]
food_spawn = True

# Hand tracking setup
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(min_detection_confidence=0.7, min_tracking_confidence=0.7)
mp_drawing = mp.solutions.drawing_utils

# Function to detect hand gestures
def get_hand_gesture(frame):
    frame = cv2.flip(frame, 1)  # Flip the frame horizontally for natural gesture recognition
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    result = hands.process(rgb_frame)
    gesture = None

    if result.multi_hand_landmarks:
        for hand_landmarks in result.multi_hand_landmarks:
            # Draw the hand landmarks on the frame
            mp_drawing.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)

            landmarks = hand_landmarks.landmark
            thumb_tip = landmarks[mp_hands.HandLandmark.THUMB_TIP]
            index_tip = landmarks[mp_hands.HandLandmark.INDEX_FINGER_TIP]
            middle_tip = landmarks[mp_hands.HandLandmark.MIDDLE_FINGER_TIP]

            # Basic gesture detection based on finger positions
            if abs(thumb_tip.x - index_tip.x) < 0.05 and abs(index_tip.x - middle_tip.x) < 0.05:
                # Closed fist gesture
                gesture = 'STOP'

            # Detect hand direction
            x_coords = [lm.x for lm in landmarks]
            y_coords = [lm.y for lm in landmarks]

            if max(x_coords) - min(x_coords) > max(y_coords) - min(y_coords):
                if landmarks[mp_hands.HandLandmark.WRIST].x < 0.5:
                    gesture = 'LEFT'
                else:
                    gesture = 'RIGHT'
            else:
                if landmarks[mp_hands.HandLandmark.WRIST].y < 0.5:
                    gesture = 'UP'
                else:
                    gesture = 'DOWN'

    # Resize the frame for better visibility
    frame = cv2.resize(frame, (800, 600))

    # Display the camera feed with hand landmarks
    cv2.imshow("Hand Gesture", frame)

    return gesture

# Main game loop
def game_loop():
    global snake_pos, snake_direction, change_to, food_pos, food_spawn

    # Initialize the camera
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("Error: Camera not accessible")
        sys.exit()

    score = 0
    game_started = False
    running = True

    while running:
        # Handle events
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False

        # Get hand gesture input
        ret, frame = cap.read()
        if not ret:
            print("Error: Unable to read from camera")
            break

        gesture = get_hand_gesture(frame)

        if not game_started:
            if gesture == 'STOP':
                print("Fist detected! Starting the game.")
                game_started = True
            else:
                continue

        if gesture == 'UP' and not snake_direction == 'DOWN':
            change_to = 'UP'
        if gesture == 'DOWN' and not snake_direction == 'UP':
            change_to = 'DOWN'
        if gesture == 'LEFT' and not snake_direction == 'RIGHT':
            change_to = 'LEFT'
        if gesture == 'RIGHT' and not snake_direction == 'LEFT':
            change_to = 'RIGHT'

        # Update snake direction
        snake_direction = change_to

        # Move the snake
        if snake_direction == 'UP':
            snake_pos[0][1] -= CELL_SIZE
        if snake_direction == 'DOWN':
            snake_pos[0][1] += CELL_SIZE
        if snake_direction == 'LEFT':
            snake_pos[0][0] -= CELL_SIZE
        if snake_direction == 'RIGHT':
            snake_pos[0][0] += CELL_SIZE

        # Snake body growing mechanism
        if snake_pos[0] == food_pos:
            score += 1
            food_spawn = False
        else:
            snake_pos.pop()

        if not food_spawn:
            food_pos = [random.randrange(1, (WIDTH // CELL_SIZE)) * CELL_SIZE,
                        random.randrange(1, (HEIGHT // CELL_SIZE)) * CELL_SIZE]
        food_spawn = True

        # Game over conditions
        if (snake_pos[0][0] < 0 or snake_pos[0][0] >= WIDTH or
            snake_pos[0][1] < 0 or snake_pos[0][1] >= HEIGHT):
            print("Game Over! Your score: ", score)
            running = False

        # Check if the snake collides with itself
        for block in snake_pos[1:]:
            if snake_pos[0] == block:
                print("Game Over! Your score: ", score)
                running = False

        # Update the snake's body
        snake_pos.insert(0, list(snake_pos[0]))

        # Draw everything
        screen.fill(WHITE)
        for pos in snake_pos:
            pygame.draw.rect(screen, GREEN, pygame.Rect(pos[0], pos[1], CELL_SIZE, CELL_SIZE))
        pygame.draw.rect(screen, RED, pygame.Rect(food_pos[0], food_pos[1], CELL_SIZE, CELL_SIZE))

        # Refresh game screen
        pygame.display.update()

        # Frame rate
        clock.tick(8)

    cap.release()
    cv2.destroyAllWindows()
    pygame.quit()

if __name__ == "__main__":
    game_loop()
