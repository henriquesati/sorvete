import cv2
from ultralytics import YOLO

model = YOLO("yolov8n.pt", verbose=True)

cap = cv2.VideoCapture("../video.mp4")

# Obter informações do vídeo original
frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
fps = int(cap.get(cv2.CAP_PROP_FPS))

# Configurar o VideoWriter para salvar o vídeo de saída
out = cv2.VideoWriter(
    "output_video.mp4", 
    cv2.VideoWriter_fourcc(*"mp4v"), 
    fps, 
    (frame_width, frame_height)
)

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    results = model(frame)

    for result in results:
        if result.boxes is not None:
            boxes = result.boxes.xyxy.cpu().numpy()

            for box in boxes:
                x1, y1, x2, y2 = map(int, box[:4])
                
                # Desenhar retângulo
                cv2.rectangle(frame, 
                              (x1, y1), 
                              (x2, y2), 
                              (0, 255, 0),  # Cor verde
                              2)            # Espessura da linha
        cv2.imshow("YOLOv8 Detection", frame)
    
    # Escrever o frame processado no vídeo de saída
    out.write(frame)
    
    # Pressione 'q' para sair
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
out.release()
cv2.destroyAllWindows()