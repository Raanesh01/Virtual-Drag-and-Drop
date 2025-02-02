# Virtual-Drag-and-Drop
Virtual Drag and Drop is an interactive project built using OpenCV and MediaPipe Hands, allowing users to manipulate images through hand gestures. By detecting finger positions in real-time, the system enables users to drag an image by crossing their first two fingers and drop it when they release the gesture.

##  Features :
1)  Hand Gesture Recognition using MediaPipe Hands
2)  Drag & Drop functionality with finger crossing detection
3)  Smooth and real-time tracking with OpenCV
4)  Supports multiple images for interaction

## Demo : 
1)  Cross your first two fingers → Drag the image
2)  Release the fingers → Drop the image

## Tech Stack :
1) Python
2) OpenCV
3) MediaPipe

## Installation & Setup : 

1️) Clone the Repository
git clone https://github.com/Raanesh01/Virtual-Drag-and-Drop.git
cd virtual-drag-drop

2) Install Dependencies
pip install opencv-python mediapipe numpy

3) Run the Script
python "virtual drag and drop.py"

##  How It Works
1) Uses MediaPipe Hands to detect finger positions
2) Identifies crossed fingers as a drag action
3) Tracks finger movement to move the image
4) Detects finger release to drop the image




