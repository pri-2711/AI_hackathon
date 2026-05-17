from PIL import Image, ImageEnhance, ImageOps
import os


def augment(img):

    return [
        ("rot", img.rotate(10)),
        ("flip", ImageOps.mirror(img)),
        ("bright",
         ImageEnhance.Brightness(img).enhance(1.2)),

        ("zoom",
         img.crop((6,6,122,122)).resize((128,128)))
    ]


for folder in os.listdir("balanced_dataset"):

    os.makedirs(
        f"augmented_dataset/{folder}",
        exist_ok=True
    )


    for file in os.listdir(
        f"balanced_dataset/{folder}"
    ):

        img = Image.open(
            f"balanced_dataset/{folder}/{file}"
        )

        for name, aug in augment(img):

            aug.save(
                f"augmented_dataset/{folder}/{file}_{name}.jpg"
            )


print("Augmentation of images : Done")