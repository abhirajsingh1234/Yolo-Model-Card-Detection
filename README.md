# Yolo-Model-Card-Detection
A real-time card detection system using a custom-trained YOLO model. The application guides users with on-screen instructions to center the card in view. Once aligned and held steady, the system automatically captures, crops, and saves the card image before closing.


# Real-Time Card Detection & Auto-Capture System

This project uses a custom YOLO model to detect a card in a live camera feed, guide the user to position it at the center, and automatically capture and crop the card once it is perfectly aligned. The system closes automatically after saving the cropped image.

---

## 🚀 Features

- **YOLO-based real-time card detection**
- **Live directional guidance**  
  (`Move Left`, `Move Right`, `Move Up`, `Move Down`)
- **Center alignment and "Hold Still" check**
- **Automatic capture & crop**
- **Auto-close after saving the image**
- **Lightweight & fast execution**

---

## 📂 Project Structure

project/
│── model/
│ └── best.pt # Trained YOLO model
│── saved_cards/
│ └── card_001.jpg # Auto-saved cropped cards
│── main.py # Main detection + guidance script
│── README.md

yaml
Copy code

---

## 🛠 Requirements

Install dependencies:

```bash
pip install ultralytics opencv-python numpy
▶️ Usage
Run the application:

bash
Copy code
python main.py
Steps for the user:

Show the card to the camera.

Follow the real-time instructions on the screen.

When the card is perfectly centered and steady:

The system captures the image.

Crops the card.

Saves it in the saved_cards/ folder.

Program exits automatically.

🧠 How It Works
YOLO detects the card and returns bounding box coordinates.

The program compares the card’s center with the screen center.

It prints directional guidance until the card is aligned.

Once the card stays centered for a fixed duration:

A snapshot is captured.

The card area is cropped using YOLO coordinates.

It is saved with a timestamp.

The app closes automatically.

📷 Output Example
saved_cards/card_20251116_193212.jpg

🔧 Model Training
The YOLO model is trained using custom-labeled card images in YOLO format:

kotlin
Copy code
images/
labels/
data.yaml
Train command:

bash
Copy code
yolo task=detect mode=train model=yolov8n.pt data=data.yaml epochs=50 imgsz=640
🤝 Contributions
Feel free to fork the project and submit improvements or issues.
