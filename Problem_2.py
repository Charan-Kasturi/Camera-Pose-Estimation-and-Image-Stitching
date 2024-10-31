import cv2 as cv,numpy as np,os
import matplotlib.pyplot as plt
sift = cv.SIFT_create()

CURRENT_DIR = os.path.dirname(__file__)

# Importing images
image_paths=[]

for i in range(1,5):
    image_paths.append(os.path.join(CURRENT_DIR, f'image_{i}.jpg'))

img1 = cv.imread(image_paths[0], cv.IMREAD_GRAYSCALE)
h, w = img1.shape[:2]
ratio = float(500) / max(h, w)
new_h = int(ratio * h)
new_w = int(ratio * w)

# Resize the image
img1 = cv.resize(img1, (new_w, new_h))

img2 = cv.imread(image_paths[1], cv.IMREAD_GRAYSCALE)
# Resize the image
img2 = cv.resize(img2, (new_w, new_h))

img3 = cv.imread(image_paths[2], cv.IMREAD_GRAYSCALE)
# Resize the image
img3 = cv.resize(img3, (new_w, new_h))

img4 = cv.imread(image_paths[3], cv.IMREAD_GRAYSCALE)
# Resize the image
img4 = cv.resize(img4, (new_w, new_h))

def trimmer(frame):
    while not np.any(frame[0]):
        frame = frame[1:]
    while not np.any(frame[-1]):
        frame = frame[:-2]
    while not np.any(frame[:,0]):
        frame = frame[:,1:]
    while not np.any(frame[:,-1]):
        frame = frame[:,:-2]
    return frame

kp1, des1 = sift.detectAndCompute(img1, None)
kp2, des2 = sift.detectAndCompute(img2, None)
kp3, des3 = sift.detectAndCompute(img3, None)
kp4, des4 = sift.detectAndCompute(img4, None)

FLANN_INDEX_KDTREE = 0
index_params = dict(algorithm=FLANN_INDEX_KDTREE, trees=5)
search_params = dict(checks=50)

flann = cv.FlannBasedMatcher(index_params, search_params)

matches12 = flann.knnMatch(des1, des2, k=2)
matches23 = flann.knnMatch(des2, des3, k=2)
matches34 = flann.knnMatch(des3, des4, k=2)

good12 = []
if len(matches12) > 0:
    for m, n in matches12:
        if m.distance < 0.7 * n.distance:
            good12.append([m])

good23 = []
if len(matches23) > 0:
    for m, n in matches23:
        if m.distance < 0.7 * n.distance:
            good23.append([m])

good34 = []
if len(matches34) > 0:
    for m, n in matches34:
        if m.distance < 0.7 * n.distance:
            good34.append([m])

img_matches12 = cv.drawMatchesKnn(img1, kp1, img2, kp2, good12, None, flags=cv.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS)
img_matches23 = cv.drawMatchesKnn(img2, kp2, img3, kp3, good23, None, flags=cv.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS)
img_matches34 = cv.drawMatchesKnn(img3, kp3, img4, kp4, good34, None, flags=cv.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS)

cv.imshow('Matches 1-2', img_matches12)
cv.imshow('Matches 2-3', img_matches23)
cv.imshow('Matches 3-4', img_matches34)
cv.waitKey(0)

MIN_MATCH_COUNT = 10





def compute_homography(kp1, kp2,matches):
    if len(matches) < MIN_MATCH_COUNT:
        return None

    src_pts = np.float32([ kp1[m[0].queryIdx].pt for m in matches ]).reshape(-1,1,2)
    dst_pts = np.float32([ kp2[m[0].trainIdx].pt for m in matches ]).reshape(-1,1,2)
    src_pts = np.hstack((src_pts.reshape(-1,2), np.ones((len(src_pts), 1))))
    dst_pts = np.hstack((dst_pts.reshape(-1,2), np.ones((len(dst_pts), 1))))
   

    if src_pts.shape[0] < 4 or dst_pts.shape[0] < 4:
        raise ValueError("Not enough points to find homography")

    
    A = []
    for i in range(len(src_pts)):
        x, y = src_pts[i,0:2]
        u, v = dst_pts[i,0:2]
        A.append([x,y,1, 0, 0, 0, -x*u, -y*u, -u])
        A.append([0, 0, 0, x*0.5, y, 1, -x*v, -y*v, -v])
    A = np.array(A)

    _, _, Vt = np.linalg.svd(A)
    H = Vt[-1].reshape(3, 3)
    H /= H[2, 2]

    return H

H12 = compute_homography(kp1, kp2, good12)
H23 = compute_homography(kp2, kp3, good23)
H34 = compute_homography(kp3, kp4, good34)


# Compute the size of the output image
corners1 = np.float32([[0, 0], [0, h], [w, h], [w, 0]]).reshape(-1, 1, 2)
corners2 = cv.perspectiveTransform(corners1, H12)
corners3 = cv.perspectiveTransform(corners2, H23)
corners4 = cv.perspectiveTransform(corners3, H34)

x_min = int(min(corners1[:,:,0].min(), corners2[:,:,0].min(), corners3[:,:,0].min(), corners4[:,:,0].min()))
y_min = int(min(corners1[:,:,1].min(), corners2[:,:,1].min(), corners3[:,:,1].min(), corners4[:,:,1].min()))
x_max = int(max(corners1[:,:,0].max(), corners2[:,:,0].max(), corners3[:,:,0].max(), corners4[:,:,0].max()))
y_max = int(max(corners1[:,:,1].max(), corners2[:,:,1].max(), corners3[:,:,1].max(), corners4[:,:,1].max()))

# Create a translation matrix to shift the image
translation_matrix = np.array([[1, 0, -x_min], [0, 1, -y_min], [0, 0, 1]])

# Warp the images using the homographies
pano1 = cv.warpPerspective(img1, translation_matrix.dot(H12), (x_max - x_min, y_max - y_min))
pano1=trimmer(pano1)
# pano1=cv.cvtColor(pano1,cv.COLOR_GRAY2BGR)
pano2 = cv.warpPerspective(img2, translation_matrix.dot(H23), (x_max - x_min, y_max - y_min))
pano2=trimmer(pano2)
pano3 = cv.warpPerspective(img3, translation_matrix.dot(H34), (x_max - x_min, y_max - y_min))
pano3=trimmer(pano3)

cv.imshow('Panorama1', pano1)
cv.imshow('Panorama2', pano2)
cv.imshow('Panorama3', pano3)
cv.waitKey(0)

print(pano1.shape,pano2.shape,pano3.shape)
# Combine the images
pano = np.zeros((pano1.shape[0]+pano2.shape[0]+pano3.shape[0],pano1.shape[1]+pano2.shape[1]+pano3.shape[1]), dtype=np.uint8)
# print(pano.shape)
pano[:pano1.shape[0], :pano1.shape[1]] = cv.flip(pano1,0)
pano[:pano1.shape[0], :pano1.shape[1]] = pano1
# print(pano2.shape,pano1.shape[1]+pano2.shape[1])
pano[:pano3.shape[0], pano1.shape[1]+pano2.shape[1]:] = cv.flip(pano3,0)
pano[:pano3.shape[0], pano1.shape[1]+pano2.shape[1]:] = pano3
pano[:pano2.shape[0], pano1.shape[1]:pano1.shape[1]+pano2.shape[1]] = cv.flip(pano2,0)
pano[:pano2.shape[0], pano1.shape[1]:pano1.shape[1]+pano2.shape[1]] = pano2

# Show the resulting panorama
pano=trimmer(pano)
print(pano.shape,'shape')
pano=pano[100:550,:]
cv.imshow('Panorama', pano)

cv.waitKey(0)