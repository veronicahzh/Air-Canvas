# Air-Canvas

Air Canvas is a real-time computer vision drawing app that lets users draw in the air using hand gestures captured through a webcam. It uses **MediaPipe** for hand tracking and **OpenCV** for video processing and rendering, allowing users to draw, pause, and clear the canvas without touching the screen.

## Features

- **Draw in the air** using your index finger
- **Stop drawing** by making a fist
- **Clear the canvas** with an open palm
- **Live webcam overlay** showing both the camera feed and drawing
- **Real-time hand landmark detection** with MediaPipe

## Demo Controls

The program recognizes three hand gestures:

- **Index finger only up** → Draw
- **Fist** → Stop drawing
- **Open palm** → Clear the canvas

Press **Q** to quit the application.

## Technologies Used

- **Python**
- **OpenCV**
- **MediaPipe**
- **NumPy**

## How It Works

The application captures live video from the webcam and flips the frame horizontally for a mirror effect. MediaPipe then detects hand landmarks and tracks fingertip positions.

Based on the relative positions of the fingers, the program classifies hand gestures into drawing, stopping, or clearing modes:

- When only the **index finger** is raised, the app tracks the fingertip and draws lines between consecutive points.
- When a **fist** is detected, drawing pauses.
- When an **open palm** is detected, the canvas resets.

The drawing is stored on a separate canvas and blended with the webcam feed in real time.

## Future Improvements

- Add color selection gestures
- Add brush size controls
- Support saving drawings as image files
- Improve gesture recognition for more stability
- Extend from 2D drawing to 3D stroke visualization