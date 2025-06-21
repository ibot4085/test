import cv2
import numpy as np

def remove_objects_with_mask(image_path, mask_path, output_path):
    # خوێندنەوەی وێنە و ماسک
    image = cv2.imread(image_path)
    mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)  # ماسک بە شێوەی ڕەش و سپی
    
    # دروستکردنی ماسکی پێچەوانە (ئەو شتانەی دەتەوێت بیسڕیتەوە)
    inverse_mask = cv2.bitwise_not(mask)
    
    # سڕینەوەی ناوچە ماسکراوەکان بە هاوکێشەی "inpainting"
    result = cv2.inpaint(image, inverse_mask, inpaintRadius=3, flags=cv2.INPAINT_TELEA)
    
    # پاشەکەوتکردنی وێنە
    cv2.imwrite(output_path, result)

# نموونەی بەکارهێنان
remove_objects_with_mask("image.jpg", "mask.png", "output.jpg")