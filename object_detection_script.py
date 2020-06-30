from model import *


def predict(image_path):
    # just give image path 
    img = Image.open(image_path)
#     print(img.size)
    image = letterbox_image(img, (416, 416))
#     print(image.size)
    
    box, score, classes = detect_objects(image)
    
    return show_objects(image, box, score, classes)


# give image path or give name with extension if the image is present in the project directory
predict(r"F:\Computer Vision\Object Detection Yolov3\test.jpg")