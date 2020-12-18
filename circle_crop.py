import os
import glob
import numpy as np
import cv2

from PIL import Image, ImageDraw, ImageFilter

thumb_width = 150


def crop_center(pil_img, crop_width, crop_height):
    img_width, img_height = pil_img.size
    return pil_img.crop(((img_width - crop_width) // 2,
                         (img_height - crop_height) // 2,
                         (img_width + crop_width) // 2,
                         (img_height + crop_height) // 2))

def crop_max_square(pil_img):
    return crop_center(pil_img, min(pil_img.size), min(pil_img.size))

def mask_circle_solid(pil_img, background_color, blur_radius, offset=0):
    background = Image.new(pil_img.mode, pil_img.size, background_color)

    offset = blur_radius * 2 + offset
    mask = Image.new("L", pil_img.size, 0)
    draw = ImageDraw.Draw(mask)
    draw.ellipse((offset, offset, pil_img.size[0] - offset, pil_img.size[1] - offset), fill=255)
    mask = mask.filter(ImageFilter.GaussianBlur(blur_radius))

    return Image.composite(pil_img, background, mask)

def mask_circle_transparent(pil_img, blur_radius, offset=0):
    offset = blur_radius * 2 + offset
    mask = Image.new("L", pil_img.size, 0)
    draw = ImageDraw.Draw(mask)
    draw.ellipse((offset, offset, pil_img.size[0] - offset, pil_img.size[1] - offset), fill=255)
    mask = mask.filter(ImageFilter.GaussianBlur(blur_radius))

    result = pil_img.copy()
    result.putalpha(mask)

    return result

cam = cv2.VideoCapture(0)

if cam.isOpened():
    while True:
        ret,im = cam.read()
        pil_im = Image.fromarray(im)
        im_square = crop_max_square(pil_im).resize((thumb_width, thumb_width), Image.LANCZOS)
        im_thumb = mask_circle_solid(im_square, (0,0,0), 4)
        img = np.array(im_thumb)

        if ret:
            cv2.imshow('img',img)
            
            if cv2.waitKey(1) != -1:
                break
        else:
            break

cam.release()
cv2.destroyAllWindows()
