import os
import sys
import numpy as np
from PIL import Image, ImageChops
from calculate_ssim import *


if len(sys.argv) == 1:
    print('usage: solution.py [-h] --path PATH \n solution.py: the following arguments are required: --path')
    sys.exit()
else:
    command = sys.argv[1]

if command == '--h' or command == '-h':
    print('usage: solution.py [-h] --path PATH\nFirst test task on images similarity.')
    sys.exit()
elif command == '--path':
    directory = sys.argv[-1]

images = []
for filename in os.listdir(directory):
    # Convert to greyscale
    im = Image.open('dev_dataset/' + filename).convert('L', (0.2989, 0.5870, 0.1140, 0))
    images.append(im.resize((128, 128)))


def find_duplicates(images):
    similar = []
    for i in range(len(images)):
        for j in range(i + 1, len(images)):
            diff = ImageChops.subtract(images[i], images[j])
            # Duplicates always have variance of difference 0
            # Modifications typically have SSIM greater than 0.7
            if np.asarray(diff).var() == 0 or compare_ssim(images[i], images[j]) > 0.9:
                similar.append((i, j))

            else:
                # Shifted images are both similar to their sum roughly equally
                # so the threshold 0.03 is set
                # Also, scores tend to be around 0.5
                sum = ImageChops.add(images[i], images[j])
                score0, score1 = compare_ssim(sum, images[i]), compare_ssim(sum, images[j])
                if abs(score0 - score1) < 0.03 and 0.49 < score0 < 0.6:
                    similar.append((i, j))

    return similar


for pair in find_duplicates(images):
    print(os.listdir(directory)[pair[0]], os.listdir(directory)[pair[1]])

