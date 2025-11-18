from ultralytics import YOLO
import cv2
import os
from datetime import datetime

model = YOLO(r"C:\Users\ACER\OneDrive\डेस्कटॉप\E-Card Detection\exp\weights\best.pt")

cap = cv2.VideoCapture(0)

# Create captured folder if it doesn't exist
output_folder = "captured"
os.makedirs(output_folder, exist_ok=True)

# Tracking variables for stability
stable_frames = 0
required_stable_frames = 15  # Number of consecutive "perfect" frames before capture

while True:
    ret, frame = cap.read()
    if not ret:
        break

    results = model(frame, conf=0.5)

    for r in results:
        boxes = r.boxes.xyxy
        if len(boxes) > 0:
            x1, y1, x2, y2 = map(int, boxes[0])
            
            # Draw box
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)

            # Compute center
            cx = (x1 + x2) // 2
            cy = (y1 + y2) // 2

            # Frame center
            h, w, _ = frame.shape
            fx = w // 2
            fy = h // 2

            # Instruction logic
            instruction = ""
            is_centered = False

            if cx < fx - 40:
                instruction = "Move Right →"
                stable_frames = 0
            elif cx > fx + 40:
                instruction = "Move Left ←"
                stable_frames = 0
            elif cy < fy - 40:
                instruction = "Move Down ↓"
                stable_frames = 0
            elif cy > fy + 40:
                instruction = "Move Up ↑"
                stable_frames = 0
            else:
                instruction = "Perfect! Hold Still ✔"
                is_centered = True
                stable_frames += 1

            # Display instruction
            cv2.putText(frame, instruction, (30, 40),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 255), 3)
            
            # Display stability counter
            if is_centered:
                cv2.putText(frame, f"Capturing in {required_stable_frames - stable_frames}...", 
                            (30, 80), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 0), 2)

            # Auto-capture when stable
            if stable_frames >= required_stable_frames:
                # Crop the detected card
                cropped_card = frame[y1:y2, x1:x2]
                
                # Generate filename with timestamp
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                filename = os.path.join(output_folder, f"card_{timestamp}.jpg")
                
                # Save cropped image
                cv2.imwrite(filename, cropped_card)
                
                print(f"✓ Card captured and saved to: {filename}")
                
                # Release resources and exit
                cap.release()
                cv2.destroyAllWindows()
                exit()

    cv2.imshow("Card Detection", frame)

    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

cap.release()
cv2.destroyAllWindows()