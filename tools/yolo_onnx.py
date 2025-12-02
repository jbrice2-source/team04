import onnxruntime as ort
import cv2
import numpy as np

mode_path = "best_new.onnx"
onnx_model = ort.InferenceSession(mode_path)

classes = ["0","45","90","135","180","225","270","315"]
    

def letterbox(img: np.ndarray, new_shape: tuple[int, int] = (640, 640)) -> tuple[np.ndarray, tuple[int, int]]:
    shape = img.shape[:2]  # current shape [height, width]

    # Scale ratio (new / old)
    r = min(new_shape[0] / shape[0], new_shape[1] / shape[1])

    # Compute padding
    new_unpad = round(shape[1] * r), round(shape[0] * r)
    dw, dh = (new_shape[1] - new_unpad[0]) / 2, (new_shape[0] - new_unpad[1]) / 2  # wh padding

    if shape[::-1] != new_unpad:  # resize
        img = cv2.resize(img, new_unpad, interpolation=cv2.INTER_LINEAR)
    top, bottom = round(dh - 0.1), round(dh + 0.1)
    left, right = round(dw - 0.1), round(dw + 0.1)
    img = cv2.copyMakeBorder(img, top, bottom, left, right, cv2.BORDER_CONSTANT, value=(114, 114, 114))

    return img, (top, left)


    
# image = cv2.imread("image.png")
image = cv2.imread("Labelled_dataset/Image/135/image192.jpg")
# image = cv2.imread("IMG_20251121_180015.jpg")
# image = cv2.imread("dataset/fdd56011d91d49f78e09c064e313c4ea.png")


img_height, img_width = image.shape[:2]

# Convert the image color space from BGR to RGB
img = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

img, pad = letterbox(img, (640, 640))

# Normalize the image data by dividing it by 255.0
image_data = np.array(img) / 255.0

# Transpose the image to have the channel dimension as the first dimension
image_data = np.transpose(image_data, (2, 0, 1))  # Channel first

# Expand the dimensions of the image data to match the expected input shape
image_data = np.expand_dims(image_data, axis=0).astype(np.float32)

outputs = onnx_model.run(None, {"images": image_data})

results = outputs[0]
results = results.transpose()

res1 = np.argmax(results[:,4:])//8
result = results[res1]

class_id = np.argmax(result[4:])
conf = np.max(result[4:])

dimentions = np.array([img_width,img_height])
padding = 640-dimentions/dimentions.max()*640

bbox = np.array([[result[0]-result[2]/2-padding[0]/2,result[0]+result[2]/2-padding[0]/2],
                 [result[1]-result[3]/2-padding[1]/2,result[1]+result[3]/2-padding[1]/2]])
bbox = bbox.reshape(-1,2)*np.array([img_width/(640-padding[0]),img_height/(640-padding[1])]).reshape(1,2)
bbox = bbox.astype(int).T.reshape(-1)*640//img_width

image = cv2.resize(image.copy(),(640,360))
cv2.rectangle(image, (bbox[0],bbox[1]),(bbox[2],bbox[3]),(255,0, 0),1)
cv2.putText(image,classes[class_id]+' '+f"{conf:.2f}",(bbox[0],bbox[1]+20),
                cv2.FONT_HERSHEY_SIMPLEX,0.7,(255,0,0),1)

print(conf)

cv2.imshow("Output", image)
cv2.waitKey(0)
