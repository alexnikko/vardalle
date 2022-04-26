from dataset_generator import generate_random_image
from config import generate_params

import numpy as np
import os


if __name__ == '__main__':
    saveroot = 'samples'
    os.makedirs(saveroot, exist_ok=True)
    seed = 42
    np.random.seed(seed)
    n = 5
    for i in range(n):
        image = generate_random_image(**generate_params)
        print(type(image))
        label = str(i).zfill(2)
        image.save(os.path.join(saveroot, f'image_{label}.jpg'))
