import numpy as np
import cv2
img = cv2.imread('pictures/0_0.png')


imgray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
edges = cv2.Canny(imgray,175,200)
ret, thresh = cv2.threshold(edges, 127, 255, 0)
contours, hierarchy = cv2.findContours(thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

cont_list = []
hiere_list = []
for i,n in enumerate(contours):
    if len(n) < 100:
        cont_list.append(n)
        hiere_list.append(hierarchy[0,i])

# print(hierarchy)
# print(hierarchy[0,:,3])

# child_map = {}
# counter = (-1,0)
# for i in hierarchy[0,:,3][::-1]:
#     if i != -1:
#         counter = (i,counter[1]+1)
#     elif counter[0] != -1:
#         child_map.update({counter[0]:counter[1]})
#         counter = (-1,0)
#     else:
#         counter = (-1,0)


# print(child_map)
# item_max = (-1,0)
# for i in child_map.items():
#     if item_max[1] <= i[1]:
#         item_max = i

print(list(map(len,cont_list)))
print(np.max(cont_list))
# print(item_max)
for i in range(0,len(cont_list)):
    img2 = img.copy()
    cv2.drawContours(img2, cont_list[i:], -1, (0,0,255), 1)

    # cv2.drawContours(img, contours, 3, (0,255,0), 3)

    # cnt = contours[4]
    # cv2.drawContours(img, [cnt], 0, (0,255,0), 3)

    cv2.crop()

    cv2.imshow('Contours', img2)
    cv2.waitKey(0)
cv2.destroyAllWindows()
