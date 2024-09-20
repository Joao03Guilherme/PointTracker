import cv2
import numpy as np
import matplotlib.pyplot as plt
import numpy.ma as ma
import argparse
import pandas as pd

def main():
    # Parse command-line arguments
    parser = argparse.ArgumentParser(description='Process a video file to track points.')
    parser.add_argument('video_file', type=str, help='Path to the video file.')
    parser.add_argument('--save_plot', '-sp', type=str, help='Filename to save the plot image (PNG).')
    parser.add_argument('--save_data', '-sd', type=str, help='Filename to save the tracking data (CSV).')
    args = parser.parse_args()

    file_name = args.video_file

    # Step 1: Read and display the first frame
    cap = cv2.VideoCapture(file_name)
    ret, frame = cap.read()
    cap.release()

    if not ret:
        print("Error: Could not read the first frame.")
        exit()

    img = frame.copy()
    img_display = img.copy()

    # Step 2: Interactive selection and adjustment of points
    points = []  # Stores the points [[x, y], ...]
    selected_point_index = None  # Index of the point being moved
    radius = 5  # Radius for point selection

    def select_and_move_points(event, x, y, flags, param):
        nonlocal img_display, img, points, selected_point_index
        if event == cv2.EVENT_LBUTTONDOWN:
            # Check if the click is near an existing point
            for idx, pt in enumerate(points):
                distance = np.hypot(x - pt[0], y - pt[1])
                if distance <= radius:
                    selected_point_index = idx
                    print(f"Selected point {idx+1} for moving.")
                    break
            else:
                if len(points) < 3:
                    # Add new point
                    points.append([x, y])
                    selected_point_index = len(points) - 1
                    print(f"Point {selected_point_index+1} added at ({x}, {y}).")
        elif event == cv2.EVENT_MOUSEMOVE:
            if selected_point_index is not None:
                # Move the selected point
                points[selected_point_index] = [x, y]
        elif event == cv2.EVENT_LBUTTONUP:
            selected_point_index = None  # Stop moving the point

        # Update the display image
        img_display = img.copy()
        for idx, pt in enumerate(points):
            color = (0, 0, 255) if idx == 0 else (0, 255, 0)
            cv2.circle(img_display, tuple(map(int, pt)), radius, color, -1)
            label = ''
            if idx == 0:
                label = 'Origin'
            elif idx == 1:
                label = 'Scale Point 1'
            elif idx == 2:
                label = 'Scale Point 2'
            cv2.putText(img_display, label, (int(pt[0])+5, int(pt[1])-5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)
        if len(points) >= 3:
            cv2.line(img_display, tuple(map(int, points[1])), tuple(map(int, points[2])), (0, 255, 0), 2)  # Scale line

    cv2.namedWindow('Select and Adjust Points', cv2.WINDOW_NORMAL)
    cv2.setMouseCallback('Select and Adjust Points', select_and_move_points)

    print("Instructions:")
    print("1. Click on the image to select the origin point (red).")
    print("2. Click to add two scale points (green).")
    print("3. Click and drag the scale points to adjust their positions.")
    print("4. Press 'Enter' to proceed after you have selected and adjusted all points.")

    while True:
        cv2.imshow('Select and Adjust Points', img_display)
        key = cv2.waitKey(1) & 0xFF
        if key == 13:  # Enter key
            if len(points) == 3:
                break
            else:
                print(f"Please select 3 points before pressing 'Enter'. Currently selected {len(points)} points.")
        elif key == ord('q'):
            print("Selection canceled.")
            exit()

    cv2.destroyAllWindows()

    if len(points) != 3:
        print("Not enough points selected.")
        exit()

    origin_point = tuple(points[0])
    x_scale_point1 = tuple(points[1])
    x_scale_point2 = tuple(points[2])

    # Step 3: Compute scale and coordinate transformation
    # Compute the vector along the X-axis in pixel coordinates
    dx = x_scale_point2[0] - x_scale_point1[0]
    dy = x_scale_point2[1] - x_scale_point1[1]
    distance_pixels = np.sqrt(dx**2 + dy**2)

    real_world_distance = float(input("Enter the real-world distance between the two X-scale points (in cm): "))
    pixels_per_cm = distance_pixels / real_world_distance

    # Calculate rotation angle to align X-axis with the line between x_scale_point1 and x_scale_point2
    angle = np.arctan2(dy, dx)
    cos_theta = np.cos(angle)
    sin_theta = np.sin(angle)

    print(f"Pixels per cm: {pixels_per_cm}")
    print(f"Origin point (pixels): {origin_point}")
    print(f"X-scale points: {x_scale_point1}, {x_scale_point2}")
    print(f"Angle (radians): {angle}")

    # Step 4: Adjust tracking code
    # Initialize variables
    cap = cv2.VideoCapture(file_name)
    time = []
    green_x = []
    green_y = []
    blue_x = []
    blue_y = []
    frame_count = 0
    fps = cap.get(cv2.CAP_PROP_FPS)

    # Tracking loop
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break  # End of video
        frame_count += 1
        current_time = frame_count / fps
        time.append(current_time)

        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

        # Mask for green color
        lower_green = np.array([40, 70, 70])
        upper_green = np.array([80, 255, 255])
        mask_green = cv2.inRange(hsv, lower_green, upper_green)

        # Mask for blue color
        lower_blue = np.array([90, 70, 70])
        upper_blue = np.array([130, 255, 255])
        mask_blue = cv2.inRange(hsv, lower_blue, upper_blue)

        # Find contours for green dot
        contours_green, _ = cv2.findContours(mask_green, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if contours_green:
            c = max(contours_green, key=cv2.contourArea)
            M = cv2.moments(c)
            if M['m00'] != 0:
                cx_green = M['m10'] / M['m00']
                cy_green = M['m01'] / M['m00']
                green_x.append(cx_green)
                green_y.append(cy_green)
            else:
                green_x.append(None)
                green_y.append(None)
        else:
            green_x.append(None)
            green_y.append(None)

        # Find contours for blue dot
        contours_blue, _ = cv2.findContours(mask_blue, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if contours_blue:
            c = max(contours_blue, key=cv2.contourArea)
            M = cv2.moments(c)
            if M['m00'] != 0:
                cx_blue = M['m10'] / M['m00']
                cy_blue = M['m01'] / M['m00']
                blue_x.append(cx_blue)
                blue_y.append(cy_blue)
            else:
                blue_x.append(None)
                blue_y.append(None)
        else:
            blue_x.append(None)
            blue_y.append(None)

    cap.release()

    # Step 5: Adjust positions relative to origin and rotate coordinates
    # Adjust positions relative to the origin
    green_x_adj = [x - origin_point[0] if x is not None else None for x in green_x]
    green_y_adj = [y - origin_point[1] if y is not None else None for y in green_y]
    blue_x_adj = [x - origin_point[0] if x is not None else None for x in blue_x]
    blue_y_adj = [y - origin_point[1] if y is not None else None for y in blue_y]

    # Rotate coordinates to align with the X-axis defined by the scale points
    def rotate_coords(x, y, cos_theta, sin_theta):
        x_rot = x * cos_theta + y * sin_theta
        y_rot = -x * sin_theta + y * cos_theta
        return x_rot, y_rot

    green_x_rot = []
    green_y_rot = []
    for x, y in zip(green_x_adj, green_y_adj):
        if x is not None and y is not None:
            x_r, y_r = rotate_coords(x, y, cos_theta, sin_theta)
            green_x_rot.append(x_r)
            green_y_rot.append(y_r)
        else:
            green_x_rot.append(None)
            green_y_rot.append(None)

    blue_x_rot = []
    blue_y_rot = []
    for x, y in zip(blue_x_adj, blue_y_adj):
        if x is not None and y is not None:
            x_r, y_r = rotate_coords(x, y, cos_theta, sin_theta)
            blue_x_rot.append(x_r)
            blue_y_rot.append(y_r)
        else:
            blue_x_rot.append(None)
            blue_y_rot.append(None)

    # Step 6: Convert pixels to centimeters
    green_x_cm = [x / pixels_per_cm if x is not None else None for x in green_x_rot]
    green_y_cm = [y / pixels_per_cm if y is not None else None for y in green_y_rot]
    blue_x_cm = [x / pixels_per_cm if x is not None else None for x in blue_x_rot]
    blue_y_cm = [y / pixels_per_cm if y is not None else None for y in blue_y_rot]

    # Step 7: No interpolation - Keep missing data as None
    # Prepare data for plotting and saving

    # Step 8: Save data to CSV if requested
    if args.save_data:
        # Prepare data dictionary
        data = {
            'Time (s)': time,
            'Green_X_cm': green_x_cm,
            'Green_Y_cm': green_y_cm,
            'Blue_X_cm': blue_x_cm,
            'Blue_Y_cm': blue_y_cm
        }
        df = pd.DataFrame(data)
        df.to_csv(args.save_data, index=False)
        print(f"Tracking data saved to {args.save_data}")

    # Step 9: Plot the positions
    plt.figure(figsize=(12, 8))

    plt.subplot(2, 2, 1)
    plt.plot(time, green_x_cm, 'g-', label='Green X Position')
    plt.xlabel('Time (s)')
    plt.ylabel('X Position (cm)')
    plt.title('Green Dot X Position vs Time')
    plt.legend()
    plt.grid(True)
    plt.tick_params(axis='both', which='both')

    plt.subplot(2, 2, 2)
    plt.plot(time, green_y_cm, 'g-', label='Green Y Position')
    plt.xlabel('Time (s)')
    plt.ylabel('Y Position (cm)')
    plt.title('Green Dot Y Position vs Time')
    plt.legend()
    plt.grid(True)
    plt.tick_params(axis='both', which='both')

    plt.subplot(2, 2, 3)
    plt.plot(time, blue_x_cm, 'b-', label='Blue X Position')
    plt.xlabel('Time (s)')
    plt.ylabel('X Position (cm)')
    plt.title('Blue Dot X Position vs Time')
    plt.legend()
    plt.grid(True)
    plt.tick_params(axis='both', which='both')

    plt.subplot(2, 2, 4)
    plt.plot(time, blue_y_cm, 'b-', label='Blue Y Position')
    plt.xlabel('Time (s)')
    plt.ylabel('Y Position (cm)')
    plt.title('Blue Dot Y Position vs Time')
    plt.legend()
    plt.grid(True)
    plt.tick_params(axis='both', which='both')

    plt.tight_layout()

    # Step 10: Save plot to PNG if requested
    if args.save_plot:
        plt.savefig(args.save_plot)
        print(f"Plot image saved to {args.save_plot}")

    plt.show()

if __name__ == "__main__":
    main()
