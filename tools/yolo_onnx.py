import onnxruntime as ort
import cv2
import numpy as np

mode_path = "best.onnx"
onnx_model = ort.InferenceSession(mode_path)

classes = ["0","45","90","135","180","225","270","315"]
    

def letterbox(img: np.ndarray, new_shape: tuple[int, int] = (640, 640)) -> tuple[np.ndarray, tuple[int, int]]:
    """Resize and reshape images while maintaining aspect ratio by adding padding.

    Args:
        img (np.ndarray): Input image to be resized.
        new_shape (tuple[int, int]): Target shape (height, width) for the image.

    Returns:
        img (np.ndarray): Resized and padded image.
        pad (tuple[int, int]): Padding values (top, left) applied to the image.
    """
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
# image = cv2.imread("Labelled_dataset/Image/135/image192.jpg")
image = cv2.imread("IMG_20251121_180015.jpg")


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
print(results.shape)


def filter_Detections(results, thresh = 0.001):
    # if model is trained on 1 class only
    if len(results[0]) == 5:
        # filter out the detections with confidence > thresh
        considerable_detections = [detection for detection in results if detection[4] > thresh]
        considerable_detections = np.array(considerable_detections)
        return considerable_detections

    # if model is trained on multiple classes
    else:
        A = []
        for detection in results:

            class_id = detection[4:].argmax()
            
            confidence_score = np.sum(detection[4:])

            if confidence_score > thresh:
                print(detection[4:])


            new_detection = np.append(detection[:4],[class_id,confidence_score])

            A.append(new_detection)

        A = np.array(A)
        # filter out the detections with confidence > thresh
        considerable_detections = [detection for detection in A if detection[-1] > thresh]
        considerable_detections = np.array(considerable_detections)
        # print(considerable_detections[:,-1])
        return considerable_detections
    
results = filter_Detections(results)

print(results.shape)


def NMS(boxes, conf_scores, iou_thresh = 0.55):

    #  boxes [[x1,y1, x2,y2], [x1,y1, x2,y2], ...]

    x1 = boxes[:,0]
    y1 = boxes[:,1]
    x2 = boxes[:,2]
    y2 = boxes[:,3]

    areas = (x2-x1)*(y2-y1)

    order = conf_scores.argsort()

    keep = []
    keep_confidences = []

    while len(order) > 0:
        idx = order[-1]
        A = boxes[idx]
        conf = conf_scores[idx]

        order = order[:-1]

        xx1 = np.take(x1, indices= order)
        yy1 = np.take(y1, indices= order)
        xx2 = np.take(x2, indices= order)
        yy2 = np.take(y2, indices= order)

        keep.append(A)
        keep_confidences.append(conf)

        # iou = inter/union

        xx1 = np.maximum(x1[idx], xx1)
        yy1 = np.maximum(y1[idx], yy1)
        xx2 = np.minimum(x2[idx], xx2)
        yy2 = np.minimum(y2[idx], yy2)

        w = np.maximum(xx2-xx1, 0)
        h = np.maximum(yy2-yy1, 0)

        intersection = w*h

        # union = areaA + other_areas - intesection
        other_areas = np.take(areas, indices= order)
        union = areas[idx] + other_areas - intersection

        iou = intersection/union

        boleans = iou < iou_thresh

        order = order[boleans]

        # order = [2,0,1]  boleans = [True, False, True]
        # order = [2,1]

    return keep, keep_confidences



# function to rescale bounding boxes 
def rescale_back(results,img_w,img_h):
    cx, cy, w, h, class_id, confidence = results[:,0], results[:,1], results[:,2], results[:,3], results[:,4], results[:,-1]
    cx = cx/640.0 * img_w
    cy = cy/640.0 * img_h
    w = w/640.0 * img_w
    h = h/640.0 * img_h
    x1 = cx - w/2
    y1 = cy - h/2
    x2 = cx + w/2
    y2 = cy + h/2

    boxes = np.column_stack((x1, y1, x2, y2, class_id))
    keep, keep_confidences = NMS(boxes,confidence)
    print(np.array(keep).shape)
    return keep, keep_confidences


rescaled_results, confidences = rescale_back(results, img_width, img_height)


for res, conf in zip(rescaled_results, confidences):

    x1,y1,x2,y2, cls_id = res
    cls_id = int(cls_id)
    x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
    conf = "{:.2f}".format(conf)
    # draw the bounding boxes
    cv2.rectangle(image,(int(x1),int(y1)),(int(x2),int(y2)),(255,0, 0),1)
    cv2.putText(image,classes[cls_id]+' '+conf,(x1,y1-17),
                cv2.FONT_HERSHEY_SIMPLEX,0.7,(255,0,0),1)

print(confidences)
# image = cv2.resize(image, (640, 640))

cv2.imshow("Output", image)
cv2.waitKey(0)