from keras.models import load_model
import matplotlib.pyplot as plt
import cv2
import numpy as np

# ##################### TESTING THE TRAINED MODEL ######################

# Load the trained generator model for testing
generator = load_model('gen_e_50.h5', compile=False)

# Select random test images for evaluation
lr_test = cv2.imread("data/lr_images/im2.jpg")
hr_test = cv2.imread("data/hr_images/im2.jpg")

lr_test = cv2.cvtColor(lr_test, cv2.COLOR_BGR2RGB)
hr_test = cv2.cvtColor(hr_test, cv2.COLOR_BGR2RGB)

lr_test = lr_test / 255
hr_test = hr_test / 255

# Add batch dimension to the test images
src_image = np.expand_dims(lr_test, axis=0)  # Shape becomes (1, 32, 32, 3)
tar_image = np.expand_dims(hr_test, axis=0)  # Shape becomes (1, 128, 128, 3)

# Generate super-resolution image from the low-resolution input
gen_image = generator.predict(src_image)


# Display the original LR image, generated image, and original HR image side by side
plt.figure(figsize=(16, 8))
plt.subplot(231)
plt.title('LR Image')
plt.imshow(src_image[0, :, :, :])
plt.subplot(232)
plt.title('Superresolution')
plt.imshow(gen_image[0, :, :, :])
plt.subplot(233)
plt.title('Orig. HR image')
plt.imshow(tar_image[0, :, :, :])

plt.show()
