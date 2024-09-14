import prediction_pipeline as pp
import numpy as np
import cv2 
from config import config
from processes.class_to_char import getChar

cp = cv2.VideoCapture(0)
blank = np.zeros((90,1700,3))
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
        predicted_class = np.argmax(result['prediction'][0])
        res = getChar(predicted_class=predicted_class)
        if res:
            if res == 'switch':
                if config.CURRENT_HALF == 'first':
                    config.CURRENT_HALF = 'second'
                else:
                    config.CURRENT_HALF = 'first'

            config.SESSION_TEXT += res
            config.CONFERMATION_LIS = [-1] * config.CONFERMATION_RATE 
        text = str(''.join(config.SESSION_TEXT))
        if config.CURRENT_HALF == 'first':
            result = config.FIRST_HALF[predicted_class]
        else:
            result = config.SECOND_HALF[predicted_class]
        cv2.putText(frame, f"Result = {result}", (0,40),  cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 255), 2)
    cv2.imshow("Hand Landmark Detection", frame)
    config.SESSION_IMG = frame

print(config.SESSION_TEXT)
print(config.CONFERMATION_LIS)
cp.release()
cv2.destroyAllWindows()