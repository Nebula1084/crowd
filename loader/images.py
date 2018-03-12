import pickle
from io import BytesIO

import numpy as np
from PIL import Image


def load_images(path):
    f = open(path, 'rb')
    profile_images = pickle.load(f)
    f.close()
    ids = []
    images = []
    target_width = 48
    target_height = 48

    for uid, image in profile_images.items():
        image = BytesIO(image)
        image = Image.open(image)
        image = image.convert("RGB")
        if image.size != (target_width, target_height):
            left = (image.width - target_width) / 2
            top = (image.height - target_height) / 2
            right = (image.width + target_width) / 2
            bottom = (image.height + target_height) / 2

            image.crop((left, top, right, bottom))
            image.show()

        image_np = np.array(image, dtype=np.float32) / 255
        ids.append(uid)
        images.append(image_np)

    ids = np.array(ids)
    images = np.stack(images)
    return ids, images
