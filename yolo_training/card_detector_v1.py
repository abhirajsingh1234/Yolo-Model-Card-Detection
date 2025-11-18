from ultralytics import YOLO
import cv2

model = YOLO(r"C:\Users\ACER\OneDrive\डेस्कटॉप\E-Card Detection\exp\weights\best.pt")  # Your trained model

cap = cv2.VideoCapture(0)

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
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0,255,0), 2)

            # Compute center
            cx = (x1 + x2) // 2
            cy = (y1 + y2) // 2

            # Frame center
            h, w, _ = frame.shape
            fx = w // 2
            fy = h // 2

            # Instruction logic
            instruction = ""

            if cx < fx - 40:
                instruction = "Move Right →"
            elif cx > fx + 40:
                instruction = "Move Left ←"
            elif cy < fy - 40:
                instruction = "Move Down ↓"
            elif cy > fy + 40:
                instruction = "Move Up ↑"
            else:
                instruction = "Perfect! Hold Still ✔"

            cv2.putText(frame, instruction, (30, 40),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 255), 3)

    cv2.imshow("Card Detection", frame)

    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

cap.release()
cv2.destroyAllWindows()
