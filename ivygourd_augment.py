from PIL import Image, ImageEnhance, ImageOps
import os


SOURCE = "balanced_dataset/ivygourd"
DESTINATION = "augmented_dataset/ivygourd"


os.makedirs(
    DESTINATION,
    exist_ok=True
)


for file in os.listdir(SOURCE):

    img = Image.open(
        f"{SOURCE}/{file}"
    )


    augmented = [

        ("orig", img),

        ("rot",
         img.rotate(10)),

        ("flip",
         ImageOps.mirror(img)),

        ("bright",
         ImageEnhance.Brightness(img).enhance(1.2)),

        ("zoom",
         img.crop((6,6,122,122))
            .resize((128,128))
        )
    ]


    for name, aug in augmented:

        aug.save(
            f"{DESTINATION}/{file}_{name}.jpg"
        )


print("Ivygourd augmented")