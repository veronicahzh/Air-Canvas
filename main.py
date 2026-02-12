import cv2
import numpy as np
import mediapipe as mp

mp_hands = mp.solutions.hands
mp_draw = mp.solutions.drawing_utils

# gesture helpers

def finger_is_up(hand_landmarks, tip_id, pip_id): # returns true if fingertip is above PIP
    return hand_landmarks.landmark[tip_id].y < hand_landmarks.landmark[pip_id].y

def is_fist(hand_landmarks):
    idx = finger_is_up(hand_landmarks, 8, 6)
    mid = finger_is_up(hand_landmarks, 12, 10)
    ring = finger_is_up(hand_landmarks, 16, 14)
    pink = finger_is_up(hand_landmarks, 20, 18)
    return not (idx or mid or ring or pink)

def is_open_palm(hand_landmarks):
    idx = finger_is_up(hand_landmarks, 8, 6)
    mid = finger_is_up(hand_landmarks, 12, 10)
    ring = finger_is_up(hand_landmarks, 16, 14)
    pink = finger_is_up(hand_landmarks, 20, 18)
    return idx and mid and ring and pink

def is_draw_gesture(hand_landmarks):
    # Draw when only index finger is up
    idx = finger_is_up(hand_landmarks, 8, 6)
    mid = finger_is_up(hand_landmarks, 12, 10)
    ring = finger_is_up(hand_landmarks, 16, 14)
    pink = finger_is_up(hand_landmarks, 20, 18)
    return idx and not (mid or ring or pink)

# main program

def main():
    cap = cv2.VideoCapture(1) # mac camera
    
    if not cap.isOpened():
        return

    canvas = None
    prev_point = None # last fingertip loc

    # mp hands object
    with mp_hands.Hands(
        static_image_mode = False,
        max_num_hands = 1,
        model_complexity = 1,
        min_detection_confidence = 0.6,
        min_tracking_confidence = 0.6
    ) as hands:
        
        while True:
            ret, frame = cap.read()
            if not ret:
                print("ERROR: cap.read() failed (no frame).")
                break

            # flip image
            frame = cv2.flip(frame, 1)
            h, w = frame.shape[:2]

            # create canvas once
            if canvas is None:
                canvas = np.zeros((h, w, 3), dtype = np.uint8)

            # convert to RGB for mp
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            results = hands.process(rgb)

            mode_text = "NO HAND"

            if results.multi_hand_landmarks:
                hand_landmarks = results.multi_hand_landmarks[0]

                # get index fingertip position
                x = int(hand_landmarks.landmark[8].x * w)
                y = int(hand_landmarks.landmark[8].y * h)
                current_point = (x, y)

                # draw green cursor
                cv2.circle(frame, current_point, 6, (0, 255, 0), -1)

                if is_open_palm(hand_landmarks):
                    canvas[:] = 0
                    prev_point = None
                    mode_text = "CLEAR (open palm)"

                elif is_fist(hand_landmarks):
                    prev_point = None
                    mode_text = "STOP (fist)"

                elif is_draw_gesture(hand_landmarks):
                    mode_text = "DRAW (index only)"

                    # draw line between prev and curr fingertip
                    if prev_point is not None:
                        cv2.line(
                            canvas,
                            prev_point,
                            current_point,
                            (255, 255, 255),
                            6,
                            cv2.LINE_AA
                        )

                    prev_point = current_point

                else:
                    prev_point = None
                    mode_text = "IDLE"

                # draw hand skeleton
                mp_draw.draw_landmarks(
                    frame,
                    hand_landmarks,
                    mp_hands.HAND_CONNECTIONS
                )
            
            # overlay drawing onto camera feed
            combined = cv2.addWeighted(frame, 1.0, canvas, 0.9, 0)

            # display mode
            cv2.putText(
                combined,
                mode_text,
                (20, 40),
                cv2.FONT_HERSHEY_SIMPLEX,
                1.0,
                (255, 255, 255),
                2
            )

            cv2.imshow("Air Canvas - Press Q to Quit", combined)

            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()