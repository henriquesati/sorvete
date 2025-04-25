#introduzir contador.ç

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
model = YOLO("yolov8n.pt", verbose=True) # Carregar o modelo YOLOv8n com confiança e IOU ajustados


def render_persons_counter(frame, persons_history):
    cv2.putText(frame, f"estimativa pessoas unicas: {len(persons_history)}", (20, 40), 
                cv2.FONT_HERSHEY_SIMPLEX, 1, (0,255,0), 2)

def draw_rectangle(
    frame,
    start_point,
    end_point,
    color=(0, 255, 0),
    thickness=2
):
    cv2.rectangle(
        frame,
        start_point,
        end_point,
        color,
        thickness
    )

def draw_person_id(frame, track_id, position, color=(0, 255, 0), font_scale=0.5, thickness=2, font=cv2.FONT_HERSHEY_SIMPLEX, line_type=cv2.LINE_AA): 
    cv2.putText(frame, f"ID: {track_id}", position, font, font_scale, color, thickness, line_type)

def init_track_components():
    id_history = {} 
    tracker = DeepSort(max_age=30, n_init=10, nn_budget=100)
    persons_history = set()
    return  id_history, tracker, persons_history


id_history, tracker, persons_history = init_track_components()


while cap.isOpened():
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
    ret, frame = cap.read()
    if not ret:
        break
    render_persons_counter(frame, persons_history)
    # Processamento YOLO
    results = model.predict(frame,  conf=0.53, iou=0.5)[0]
    for result in results:
        if result.boxes is not None:
            boxes = result.boxes.xyxy.cpu().numpy()
            confs = result.boxes.conf.cpu().numpy()
            detections = [(box, conf, 0) for box, conf in zip(boxes, confs)]

            tracks = tracker.update_tracks(detections, frame=frame)

            # Criar lista paralela de boxes originais
            original_boxes = [box for box, _, _ in detections]

            # Iterar simultaneamente por tracks e boxes
            for track, original_box in zip(tracks, original_boxes):
                if track.is_confirmed():
                    persons_history.add(track.track_id)
                    x1, y1, x2, y2 = map(int, original_box)
                    track_id = track.track_id 
                    
                    draw_rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                    draw_person_id(frame, track_id, (x1, y1 - 10), color=(0, 255, 0), font_scale=0.5, thickness=2)
            cv2.imshow("YOLOv8 Tracking", frame)
    
    out.write(frame)

cap.release()
out.release()
cv2.destroyAllWindows()