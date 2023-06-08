import cv2
import numpy as np
def unpickle(file):
    import pickle
    with open(file, 'rb') as fo:
        dict = pickle.load(fo, encoding='bytes')
    return dict




d = unpickle('/Users/xyan/Desktop/RL/DLR_FacialEmotionRecognition-master/cifar-10-batches-py/data_batch_1')
d_data = d[b'data']

# Reshape the data into a 32x32x3 image array
image = d_data.reshape(10000, 32, 32, 3)

print(image.shape)
# Select an image from the array (e.g., first image)
selected_image = image[222]

# Convert the image to uint8 data type
# selected_image = selected_image.astype(np.uint8)

# Display the image using cv.imshow
cv2.imshow("Image", selected_image)
cv2.waitKey(0)
cv2.destroyAllWindows()



# print(len(d[b'data']))
# print(d[b'data'][0])
# print(len(d[b'data'][0]))

#cv.imshow('image',d[b'data'][0])
#cv.waitKey(0)