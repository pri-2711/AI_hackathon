import os
from PIL import Image


SOURCE = "raw_dataset/ivygourd"
DESTINATION = "balanced_dataset/ivygourd"


os.makedirs(
    DESTINATION,
    exist_ok=True
)


for file in os.listdir(SOURCE):

    try:

        img = Image.open(
            f"{SOURCE}/{file}"
        )

        img = img.convert("RGB")

        img = img.resize((128,128))

        img.save(
            f"{DESTINATION}/{file}"
        )

    except:

        print("Skipped:", file)


print("Ivygourd added to balanced_dataset")