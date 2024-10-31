import numpy as np
from matplotlib import pyplot as plt
import cv2 as cv,os
import math
CURRENT_DIR = os.path.dirname(__file__)
path_video=os.path.join(CURRENT_DIR,'project2.avi')
video= cv.VideoCapture(path_video)

coordinates = np.array([[0, 0], [0, 27.9], [21.6,27.9], [21.6, 0]])

def find_homography(p1, p2):
    
    if p1.shape[0] < 4 or p2.shape[0] < 4:
        raise ValueError("Not enough points to find homography")

    # Construct the matrix A
    A = []
    for i in range(p1.shape[0]):
        x, y = p1[i]
        u, v = p2[i]
        A.append([-x, -y, -1, 0, 0, 0, x*u, y*u, u])
        A.append([0, 0, 0, -x, -y, 1, x*v, y*v, v])
    A = np.array(A)

    # Solve for the homography matrix H using SVD
    _, _, Vt = np.linalg.svd(A)
    H = Vt[-1].reshape(3, 3)
    H /= H[2, 2]

    return H

def find_rotation_translation(H, K):
    # Calculate inverse of the intrinsic matrix
    K_inv = np.linalg.inv(K)

    # Normalize the rotation vectors by their norm
    r1_hat = H[:, 0:1] / np.linalg.norm(H[:, 0])
    r2_hat = H[:, 1:2] / np.linalg.norm(H[:, 1])
    r3_hat = np.cross(r1_hat.T, r2_hat.T).T

    # Construct rotation matrix
    R = np.hstack([r1_hat, r2_hat, r3_hat])

    # Calculate translation vector
    t = K_inv @ H[:, 2] / np.linalg.norm(K_inv @ H[:, 2])

    # Construct pose matrix
    P = np.hstack([R, t.reshape(-1, 1)])

    return P

def hough_lines(edges):
    
    theta = np.arange(-90, 90)
    cos_t = np.cos(theta)
    sin_t = np.sin(theta)
    a=edges.shape[0]
    b=edges.shape[1]
    c=int(np.ceil(np.sqrt(a**2+b**2)))
    perpendicular_range=np.arange(-c,c+1,1)
    accumulator = np.zeros((len(perpendicular_range), len(theta)))
    y_indexes, x_indexes = np.nonzero(edges)

    for i in range(len(x_indexes)):
        x = x_indexes[i]
        y = y_indexes[i]

        for j in range(len(theta)):
            rho = int(round(x * cos_t[j] + y * sin_t[j])) + c
            accumulator[rho, j] += 1

    # Find the indices of the highest values in the accumulator
    rhos, thetas = np.nonzero(accumulator >=62)

    # Convert rhos and thetas to pixel and degrees respectively
    lines = []
    for i in range(len(rhos)):
        rho = perpendicular_range[rhos[i]]
        angle = np.rad2deg(theta[thetas[i]])
        lines.append((rho, angle))


    # Convert the lines from polar coordinates to Cartesian coordinates
    cartesian_lines = []
    for line in lines:
        rho, theta = line
        a = np.cos(theta)
        b = np.sin(theta)
        x0 = a * (rho)
        y0 = b * (rho)
        x1 = int(x0 + 1500 * (-b))
        y1 = int(y0 + 1500 * (a))
        x2 = int(x0 - 1500* (-b))
        y2 = int(y0 - 1500* (a))
        cartesian_lines.append([(y1, x1), (y2, x2)])
        cv.line(frame,(y1,x1),(y2,x2),(255,0,0),1)

    # Find the intersections of the lines
    intersections = []
    for i in range(len(cartesian_lines)):
        for j in range(i+1, len(cartesian_lines)):
            line1 = cartesian_lines[i]
            line2 = cartesian_lines[j]
            intersection = get_intersection(line1, line2)
            if intersection is not None:
                intersections.append(intersection)

    # Convert the list of intersections to a NumPy array
    intersections.sort()
    corners = np.array(intersections)

    return corners[:4,:]

def get_intersection(line1, line2):
    # Compute the intersection point of two lines
    x1, y1 = line1[0]
    x2, y2 = line1[1]
    x3, y3 = line2[0]
    x4, y4 = line2[1]

    denom = (x1-x2)*(y3-y4) - (y1-y2)*(x3-x4)
    if denom == 0:
        # The lines are parallel
        return None
    else:
        px = ((x1*y2-y1*x2)*(x3-x4)-(x1-x2)*(x3*y4-y3*x4))/denom
        py = ((x1*y2-y1*x2)*(y3-y4)-(y1-y2)*(x3*y4-y3*x4))/denom
        return (int(px), int(py))
    
roll_list, pitch_list, yaw_list = [], [], []
x_list, y_list, z_list = [], [], []

while(video.isOpened()):
    # xy=np.empty((0,2))
    ret,frame=video.read()
    if ret==True:
        
        blur = cv.blur(frame,(65,65))
        hsv=cv.cvtColor(blur,cv.COLOR_BGR2HSV)
        whitepaper1=np.array([0,0,235])  
        whitepaper2=np.array([180,100,255])
        mask=cv.inRange(hsv,whitepaper1,whitepaper2)
        edge=cv.Canny(mask,10,200)
        g=hough_lines(edge)
        T=find_rotation_translation(find_homography(g,coordinates),K = np.array([[1.38e+03, 0, 9.46e+02], [0, 1.38e+03, 5.27e+02], [0, 0, 1]]))
        R=T[:,:3]
        t=T[:,3]

        roll, pitch, yaw = [math.atan2(R[i, j], R[i, k]) for i, j, k in [(0, 1, 2), (1, 2, 0), (2, 0, 1)]]
        tx, ty, tz = t

        roll_list.append(roll)
        pitch_list.append(pitch)
        yaw_list.append(yaw)
        x_list.append(tx)
        y_list.append(ty)
        z_list.append(tz)
        
        cv.imshow('lines',frame)
        # cv.imshow('frame',edge)
        key =cv.waitKey(1)
        if key==ord('q'):
            break
    else:
        break
video.release()
cv.destroyAllWindows()

# plt.plot(roll_list, label='Roll')
# plt.plot(pitch_list, label='Pitch')
# plt.plot(yaw_list, label='Yaw')
# plt.legend()
# plt.xlabel('Frame')
# plt.ylabel('Angle (rad)')
# plt.show()

# plt.plot(x_list, label='X')
# plt.plot(y_list, label='Y')
# plt.plot(z_list, label='Z')
# plt.legend()
# plt.xlabel('Frame')
# plt.ylabel('Translation (cm)')
# plt.show()