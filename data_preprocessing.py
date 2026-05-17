import os
import random
from PIL import Image


SOURCE = "raw_dataset"
DESTINATION = "balanced_dataset"


if os.path.exists(DESTINATION):

    print("balanced_dataset already exists. Skipping.")


else:

    os.mkdir(DESTINATION)


    for folder in os.listdir(SOURCE):

        images = os.listdir(
            f"{SOURCE}/{folder}"
        )

        random.shuffle(images)


        os.makedirs(
            f"{DESTINATION}/{folder}",
            exist_ok=True
        )


        count = 0


        for file in images:

            if count == 200:
                break


            try:

                img = Image.open(
                    f"{SOURCE}/{folder}/{file}"
                )

                img = img.convert("RGB")

                img = img.resize((128,128))

                img.save(
                    f"{DESTINATION}/{folder}/{file}"
                )

                count += 1


            except:

                print("Skipped:", file)


    print("Balanced + resized dataset created")



# checks still run
print("\nImages per class:")

for folder in os.listdir(DESTINATION):

    print(
        folder,
        len(
            os.listdir(
                f"{DESTINATION}/{folder}"
            )
        )
    )


print("\nImage sizes:")

for folder in os.listdir(DESTINATION):

    img_name = os.listdir(
        f"{DESTINATION}/{folder}"
    )[0]


    img = Image.open(
        f"{DESTINATION}/{folder}/{img_name}"
    )


    print(
        folder,
        img.size
    )