import cv2
import math
import numpy as np
import datetime
from datetime import datetime
import csv
import matplotlib.pyplot as plt
import matplotlib.dates as mdates

cap = cv2.VideoCapture(0)

pressure = 0.14
x_dot = [0, 0, 0, 0]
y_dot = [0, 0, 0, 0]
iter = 0
theta = 0
angle = 0
plot_list = []
list_time = []
list_theta = []
v_1 = None
v_2 = None

input("Press enter to continue")

time_start = datetime.now()

with open(f'soft_robot {pressure} bar.csv', 'w', newline='') as csvfile:
    spamwriter = csv.writer(csvfile, delimiter=',',
                            quotechar='|', quoting=csv.QUOTE_MINIMAL)
    spamwriter.writerow(['datetime', 'pressure', 'vector 1', 'vector 2', 'Theta', 'Angle'])
    while True: 
        contours_list = []
        ret, frame = cap.read()

        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

        lower_red = np.array([0, 50, 100])
        upper_red = np.array([10, 255, 255])
        mask1 = cv2.inRange(hsv, lower_red, upper_red)

        lower_red = np.array([160, 50, 100])
        upper_red = np.array([179, 255, 255])
        mask2 = cv2.inRange(hsv, lower_red, upper_red)

        mask = mask1 + mask2
        mask = cv2.erode(mask, None, iterations=1)
        mask = cv2.dilate(mask, None, iterations=1)

        contours, _ = cv2.findContours(mask.copy(), cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

        for contour in contours: 
            x, y, w, h = cv2.boundingRect(contour)
            cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)
            contours_list.append((x,y))

        if len(contours_list) >= 4:
            v_1 = np.asarray((contours_list[1][0] - contours_list[0][0], contours_list[1][1] - contours_list[0][1]))
            v_2 = np.asarray((contours_list[3][0] - contours_list[2][0], contours_list[3][1] - contours_list[2][1]))
            print(f"Vector 1 in ({v_1[0]}, {v_1[1]}) and Vector 2 in ({v_2[0]}, {v_2[1]})")

            magv_1 = np.sqrt(v_1.dot(v_1))
            magv_2 = np.sqrt(v_2.dot(v_2))
            theta = np.arccos((v_1.dot(v_2))/(magv_1*magv_2))
            angle = (theta*180)/math.pi
            print(f"Angle is {theta} radians")

        time_now = datetime.now()
        delta_time = time_now - time_start
        # delta_time = str(delta_time)
        list_time.append(delta_time.total_seconds())
        list_theta.append(theta)
        # plot_list.append((delta_time, theta))
        spamwriter.writerow([str(delta_time), pressure, v_1, v_2, theta, angle])
        
        cv2.putText(frame, str(delta_time),(20,40), cv2.FONT_HERSHEY_PLAIN,2,(0,255,0),2,cv2.LINE_AA)
        cv2.imshow('Red Color Detection', frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

# Plotting
plt.plot(list_time, list_theta)
plt.title(f'Plot Soft Robot {pressure} Bar')
plt.xlabel('Timestamp')
plt.ylabel('Theta')
plt.grid(True)
plt.show()

plt.gca().xaxis.set_major_formatter(plt.FuncFormatter(lambda x, _: '{:.0f}s'.format(x)))
plt.gcf().autofmt_xdate()

cap.release()
cv2.destroyAllWindows()