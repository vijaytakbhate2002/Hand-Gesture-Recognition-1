import prediction_pipeline as pp
import numpy as np
import cv2 

cp = cv2.VideoCapture(0)
while True:
    success, frame = cp.read()
    if not success:
        break
    if cv2.waitKey(10) & 0xFF == ord('q'):
        break

    result = pp.predictionPipe(image=frame)
    frame = cv2.flip(frame, 1)
    if result is not None:
        x_min, x_max, y_min, y_max = result['co_ordinates']
        frame = cv2.rectangle(frame, (x_min, y_min), (x_max, y_max), (0, 0, 255), 2)
        label_position = (0,30)
        predicted_class = np.argmax(result['prediction'][0])
        frame = cv2.putText(frame, f"Predicted Class = {str(predicted_class)}", label_position, cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
    cv2.imshow("Hand Landmark Detection", frame)
cp.release()
cv2.destroyAllWindows()