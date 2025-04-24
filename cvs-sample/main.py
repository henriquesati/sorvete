import cv2

# Carrega o classificador Haar Cascade
body_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_fullbody.xml')

cap = cv2.VideoCapture('../video.mp4')

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    
    # Detecção com parâmetros ajustados
    bodies = body_cascade.detectMultiScale(
        gray,
        scaleFactor=1.05,  # Valores típicos: 1.01 a 1.3
        minNeighbors=5,     # Aumente para menos falsos positivos
        minSize=(30, 30)
    )
    
    # Desenha os retângulos
    for (x, y, w, h) in bodies:
        cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)

    cv2.imshow('Detecção Haar', frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()