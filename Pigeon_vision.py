import cv2
import numpy as np
import time

from PIL import Image, ImageDraw, ImageFilter

#축소시켜 이미지처리할 사이즈 크기
thumb_width = 320
thumb_height = 240

#볼록 렌즈, exp > 1.1
def lens_distortion(pil_img, exp, scale):
    
    img = np.array(pil_img)
 
    img_c_2 = img[150:250,100:200]

    rows, cols = img_c_2.shape[:2]

    mapy, mapx = np.indices((rows,cols),dtype=np.float32)

    mapx = 2*mapx/(cols-1)-1
    mapy = 2*mapy/(rows-1)-1

    r, theta = cv2.cartToPolar(mapx,mapy)

    r[r<scale] = r[r<scale] **exp

    mapx,mapy = cv2.polarToCart(r,theta)

    mapx = ((mapx+1)*(cols-1))/2
    mapy = ((mapy+1)*(rows-1))/2

    distorted = cv2.remap(img_c_2,mapx,mapy,cv2.INTER_LINEAR)

    img[150:250,100:200] = distorted

    pil_img = Image.fromarray(img)

    return pil_img

#마스크 씌워질 이미지 빨간색 처리
def red_processing(pil_img, thre_red = 80, thre_rest_color = 50):
    arr_img = np.array(pil_img)
    indices = np.where(arr_img[:,:,2]>thre_red)
    
    b = arr_img[indices[0], indices[1], 0]
    
    g = arr_img[indices[0], indices[1], 1]

    r = arr_img[indices[0], indices[1], 2]

    for i in range(r.size):
        if b[i] < thre_rest_color and g[i] < thre_rest_color: 
            r[i] = 255
            b[i] = 0
            g[i] = 0

    arr_img[indices[0], indices[1]] = np.swapaxes(np.array((b,g,r)),0,1)

    red_pil_img = Image.fromarray(arr_img.astype(np.uint8))

    return red_pil_img

#마스킹
def mask_circle(pil_img, red_pil_img, blur_radius, offset=70):  
    background = pil_img 
    gaussianBlur = ImageFilter.GaussianBlur(2) #배경 blur 정도
    blur_background = background.filter(gaussianBlur)

    offset = blur_radius * 2 + offset #마스크 blur 범위
    mask = Image.new("L", pil_img.size)
    draw = ImageDraw.Draw(mask)
    draw.ellipse((offset, offset+100, pil_img.size[0] - offset, pil_img.size[1]+100 - offset), fill=255)
    mask = mask.filter(ImageFilter.GaussianBlur(blur_radius))

    return Image.composite(red_pil_img, blur_background, mask)

cam = cv2.VideoCapture(0)
prevTime = 0

if cam.isOpened():
    while True:
        ret,im = cam.read()
        pil_im = Image.fromarray(im)
        
        im_square = pil_im.resize((thumb_width, thumb_height), Image.LANCZOS)
        lens_img = lens_distortion(im_square, 1.8, 1)
        red_pil_img = red_processing(lens_img)
        
        mask = mask_circle(lens_img, red_pil_img, 5)
        arr_mask = np.array(mask)
        img = cv2.resize(arr_mask, dsize=(1920, 1080), interpolation=cv2.INTER_LINEAR) #cv2.INTER_CUBIC + cv2.INTER_NEAREST
        
        curTime = time.time()
        sec = curTime - prevTime
        prevTime = curTime
        fps = 1/(sec)
        str = ('FPS: %0.1f'%fps)

        if ret:
            cv2.putText(img, str, (1, 100), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0))

            cv2.imshow('img',img)
            
            if cv2.waitKey(1) != -1:
                break
        else:
            break

cam.release()
cv2.destroyAllWindows()
