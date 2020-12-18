import cv2
cam = cv2.VideoCapture(0)

ret,img = cam.read()
        
if cam.isOpened():
    while True:
        ret,img = cam.read()
        img_l = img[:,0:288]
        img_c = img[:,288:288+64]
        img_r = img[:,288+64:-1]
        if ret:
            cv2.imshow('img_l',img_l)
            cv2.imshow('img_c',img_c)
            cv2.imshow('img_r',img_r)
            if cv2.waitKey(1) != -1:
                break
        else:
            break

cam.release()
cv2.destroyAllWindows()
