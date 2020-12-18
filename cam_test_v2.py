import cv2
import numpy as np
import matplotlib.pyplot as plt

cam = cv2.VideoCapture(0)

      
if cam.isOpened():
    while True:
        ret,img = cam.read()
        img_l = img[:,0:288]
        img_c_1 = img[0:(480//2),288:288+64]
        img_c_2 = img[(480//2):-60,288:288+64] #img_c_2 = img[(480//2):-1,288:288+64]
        img_r = img[:,288+64:-1]


        b,g,r = cv2.split(img_c_2)
        #r = np.array(r)+1
        #r = np.log(r)
        #r = r/np.max(r)*255
        #r = r.astype(int)
        g = g//2
        b = b//2
        print(np.max(r))
        img_c_2 = cv2.merge((b,g,r))


        rows, cols = img_c_2.shape[:2]

        exp = 2
        scale = 1

        mapy, mapx = np.indices((rows,cols),dtype=np.float32)

        mapx = 2*mapx/(cols-1)-1
        mapy = 2*mapy/(rows-1)-1

        r, theta = cv2.cartToPolar(mapx,mapy)

        r[r<scale] = r[r<scale] **exp

        mapx,mapy = cv2.polarToCart(r,theta)

        mapx = ((mapx+1)*(cols-1))/2
        mapy = ((mapy+1)*(rows-1))/2

        distorted = cv2.remap(img_c_2,mapx,mapy,cv2.INTER_LINEAR)

        if ret:
            cv2.imshow('img_l',img_l)
            cv2.imshow('img_c_1',img_c_1)
            cv2.imshow('img_c_2',distorted)
            cv2.imshow('img_r',img_r)
            if cv2.waitKey(1) != -1:
                break
        else:
            break

cam.release()
cv2.destroyAllWindows()
