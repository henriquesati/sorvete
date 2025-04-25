import cv2
from ultralytics import YOLO
from deep_sort_realtime.deepsort_tracker import DeepSort

cap = cv2.VideoCapture("../video.mp4")
frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
fps = int(cap.get(cv2.CAP_PROP_FPS))
out = cv2.VideoWriter(
    "output_video.mp4", 
    cv2.VideoWriter_fourcc(*"mp4v"), 
    fps, 
    (frame_width, frame_height)
)
model = YOLO("yolov8n.pt", verbose=True)

def init_track_components():
    id_history = {} #armazena id e frames consecutivos de identificação
    tracker = DeepSort(max_age=30, n_init=5, nn_budget=100) #n_init controla a conf do track reduzindo falso positivo
    return  id_history, tracker


id_history, tracker = init_track_components()


while cap.isOpened():
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
    ret, frame = cap.read()
    if not ret:
        break

    # Processamento YOLO
    results = model(frame)
    
    for result in results:
        if result.boxes is not None:
            # Extrair detecções
            boxes = result.boxes.xyxy.cpu().numpy()
            confs = result.boxes.conf.cpu().numpy()
            detections = [(box, conf, 0) for box, conf in zip(boxes, confs)]

            # Atualizar tracker
            tracks = tracker.update_tracks(detections, frame=frame)

            # Criar lista paralela de boxes originais
            original_boxes = [box for box, _, _ in detections]

            # Iterar simultaneamente por tracks e boxes
            for track, original_box in zip(tracks, original_boxes):
                if track.is_confirmed():
                    # Usar coordenadas originais do YOLO
                    x1, y1, x2, y2 = map(int, original_box)
                    track_id = track.track_id 
                    print(f"Track ID: {track.track_id} | BBox: {x1}, {y1}, {x2}, {y2}")
                    
                    # Desenhar retângulo verde para tracks confirmados
                    cv2.rectangle(
                        frame, 
                        (x1, y1), 
                        (x2, y2), 
                        (0, 255, 0),  # Cor verde
                        2
                    )
                    cv2.putText(
                        frame,
                        f"ID: {track_id}",
                        (x1, y1 - 10),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.5,
                        (0, 255, 0),
                        2
                    )

            cv2.imshow("YOLOv8 Tracking", frame)
    
    out.write(frame)

cap.release()
out.release()
cv2.destroyAllWindows()