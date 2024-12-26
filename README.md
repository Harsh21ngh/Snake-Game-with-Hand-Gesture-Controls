# Snake Game with Hand Gesture Controls

This is an implementation of the classic Snake Game, but with a twist — it is controlled using hand gestures via a webcam. The game uses OpenCV for webcam feed processing and MediaPipe for detecting hand landmarks to control the movement of the snake.

## Features
- **Hand Gesture Control**: Control the snake's movement using your hand's index finger position.
- **Webcam Integration**: The game utilizes the webcam feed to track your hand gestures.
- **Dynamic Difficulty**: The game speed and snake length increase as you eat more food.
- **Score Tracking**: The game keeps track of your current score and high score.
- **Simple UI**: A minimalistic and clean interface for gameplay.

## Requirements

To run the project, you need to have Python 3.x installed, along with the following libraries:
- `pygame`
- `opencv-python`
- `mediapipe`

You can install the required libraries using `pip`:

```bash
pip install pygame opencv-python mediapipe
