from Data_Preprocessing import X_balanced
import random
from matplotlib import pyplot as plt
ix = random.randint(0, len(X_balanced) )

#------------------
# Variant-1
#------------------
plt.figure(figsize=(3, 3))
plt.subplot(1, 1, 1)
plt.imshow(X_balanced[ix],cmap='gray')
plt.title('Test Image')
plt.axis('off')
plt.show()





#------------------
# Variant-2
#------------------
import numpy as np
from skimage import exposure
import matplotlib.pyplot as plt
idx = np.random.randint(0, len(X_balanced))
original_image = X_balanced[idx].copy()
contrasted_image = exposure.rescale_intensity(original_image, in_range='image')

plt.figure(figsize=(5, 5))

plt.subplot(1, 2, 1)
plt.imshow(original_image.squeeze(), cmap='gray')
plt.title('Original Image (Balanced)')
plt.axis('off')

plt.subplot(1, 2, 2)
plt.imshow(contrasted_image.squeeze(), cmap='gray')
plt.title('Contrast Enhanced Image')
plt.axis('off')

plt.show()



#------------------
# Variant-3
#------------------
import numpy as np
from skimage.filters import gaussian
import matplotlib.pyplot as plt
idx = np.random.randint(0, len(X_balanced))
original_image = X_balanced[idx].copy()
smoothed_image = gaussian(original_image, sigma=0.3, preserve_range=True)

plt.figure(figsize=(8, 5))

plt.subplot(1, 2, 1)
plt.imshow(original_image.squeeze(), cmap='gray')
plt.title('Original Image')
plt.axis('off')

plt.subplot(1, 2, 2)
plt.imshow(smoothed_image.squeeze(), cmap='gray')
plt.title('Gaussian Smoothed Image (sigma=1)')
plt.axis('off')

plt.show()