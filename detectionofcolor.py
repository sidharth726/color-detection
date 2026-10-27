# importing the libraries
import cv2
import numpy as np
import time

# Initialize webcam
cap = cv2.VideoCapture(0)
if not cap.isOpened():
    print("Error: Could not open webcam.")
    exit()

# Extended set of color ranges in HSV
color_ranges = {
    'Red1': [(0, 120, 70), (10, 255, 255)],
    'Red2': [(170, 120, 70), (180, 255, 255)],
    'Orange': [(11, 100, 100), (25, 255, 255)],
    'Yellow': [(25, 100, 100), (35, 255, 255)],
    'Green': [(36, 50, 70), (89, 255, 255)],
    'Cyan': [(85, 100, 100), (95, 255, 255)],
    'Blue': [(90, 50, 70), (128, 255, 255)],
    'Purple': [(129, 50, 70), (158, 255, 255)],
    'Magenta': [(140, 50, 70), (170, 255, 255)],
    'White': [(0, 0, 200), (180, 25, 255)],
    'Gray': [(0, 0, 40), (180, 18, 230)],
    'Black': [(0, 0, 0), (180, 255, 40)]
}

# BGR colors for rectangle and text
color_bgr = {
    'Red': (0, 0, 255),
    'Orange': (0, 165, 255),
    'Yellow': (0, 255, 255),
    'Green': (0, 255, 0),
    'Cyan': (255, 255, 0),
    'Blue': (255, 0, 0),
    'Purple': (128, 0, 128),
    'Magenta': (255, 0, 255),
    'White': (255, 255, 255),
    'Gray': (128, 128, 128),
    'Black': (0, 0, 0)
}

screenshot_counter = 1

while True:
    ret, frame = cap.read()
    if not ret:
        print("Can't receive frame. Exiting ...")
        break

    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

    height, width, _ = frame.shape
    center_x, center_y = width // 2, height // 2

    detected_color = None
    max_area = 0
    best_contour = None
    best_box_color = (0, 255, 255)

    # Handle red separately
    red_mask1 = cv2.inRange(hsv, np.array(
        color_ranges['Red1'][0]), np.array(color_ranges['Red1'][1]))
    red_mask2 = cv2.inRange(hsv, np.array(
        color_ranges['Red2'][0]), np.array(color_ranges['Red2'][1]))
    red_mask = cv2.add(red_mask1, red_mask2)

    # Loop through all colors (including combined red)
    for color_name, (lower, upper) in color_ranges.items():
        if color_name in ['Red1', 'Red2']:
            mask = red_mask
        else:
            mask = cv2.inRange(hsv, np.array(lower), np.array(upper))

        contours, _ = cv2.findContours(
            mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        for contour in contours:
            area = cv2.contourArea(contour)
            if area > 1000:  # ignore small noisy detections
                x, y, w, h = cv2.boundingRect(contour)
                # Check if contour is near the center
                if (center_x > x and center_x < x + w) and (center_y > y and center_y < y + h):
                    if area > max_area:
                        max_area = area
                        detected_color = color_name.replace(
                            '1', '').replace('2', '')
                        best_contour = (x, y, w, h)
                        best_box_color = color_bgr.get(
                            detected_color, (0, 255, 255))

    # Draw only the most dominant color box
    if detected_color and best_contour:
        x, y, w, h = best_contour
        cv2.rectangle(frame, (x, y), (x + w, y + h), best_box_color, 3)
        cv2.putText(frame, detected_color, (x, y - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, best_box_color, 2)

    cv2.imshow("Dominant Color Detection", frame)

    key = cv2.waitKey(1) & 0xFF
    if key == ord('q'):
        break
    elif key == ord('s'):  # Save screenshot
        filename = f"screenshot_{screenshot_counter}.png"
        cv2.imwrite(filename, frame)
        print(f"Screenshot saved: {filename}")
        screenshot_counter += 1
        time.sleep(0.5)

cap.release()
cv2.destroyAllWindows()
