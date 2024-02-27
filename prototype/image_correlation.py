from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
from os import walk
import pandas as pd
from scipy.signal import correlate2d

file_names = []
for _, _, file_names in walk("<Path to Dataset>\\cyberzoo_poles\\20190121-135009"):
    print(file_names)
    break


images = []
for file_name in file_names:
    with Image.open(f"<Path to Dataset>\\cyberzoo_poles\\20190121-135009\\{file_name}") as image_file:
        images.append({'img': np.array(image_file.rotate(90, expand=True)), 'time': float(file_name.removesuffix('.jpg'))/(10**6)})

images = sorted(images, key=lambda x: x['time'])

image_size = images[0]['img'].shape
print(image_size)

with open("<Path to Dataset>\\cyberzoo_poles\\20190121-135121.csv") as csv_file:
    dataframe = pd.read_csv(csv_file)

print(dataframe)

current_image = 300

fig = plt.figure(figsize=(8, 4), layout='constrained')
plt.title(str(images[10]['time']) + " + " + str(images[11]['time']))
plt.imshow(images[current_image]['img'], alpha=0.5)
plt.imshow(images[current_image + 1]['img'], alpha=0.5)
plt.show()

rectangle_size = (20, 20, 1)
rectangle_extend = (int(rectangle_size[0]/2), int(rectangle_size[1]/2), rectangle_size[2])
amount_of_rectangles = np.prod(image_size)
print(amount_of_rectangles)

# EXAMPLE
x = 100
y = 480
c = 0
print(f"EXAMPLE: Correlation for {x}, {y}, {c}")

rectangle = np.copy(images[current_image]['img'][x - rectangle_extend[0]:x + rectangle_extend[0],
                    y - rectangle_extend[1]:y + rectangle_extend[1],
                    c])
rectangle = rectangle.astype('int16') - rectangle.mean().astype('int16')

plt.imshow(rectangle)
plt.show()

likeness_image = correlate2d(
    images[current_image + 1]['img'][:, :, c].astype('int16') - images[current_image + 1]['img'][:, :, c].mean().astype(
        'int16'), rectangle, boundary='symm', mode='same')

match_x, match_y = np.unravel_index(np.argmax(likeness_image), likeness_image.shape)

print(f"x={match_x}, y={match_y}; certainty?: {(likeness_image[match_x, match_y] - likeness_image.mean()) / 100000:3.2f}")

movement = (x - match_x,
            y - match_y)

print(f"Error={movement}")

plt.imshow(likeness_image)
plt.plot(match_y, match_x, 'ro')
plt.plot(y, x, 'rx')
plt.show()


# MOVEMENT FIELD CALC
movement_field_mag = np.zeros(image_size)


for x in range(rectangle_extend[0], image_size[0] - rectangle_extend[0], rectangle_size[0]):
    for y in range(rectangle_extend[1], image_size[1] - rectangle_extend[1], rectangle_size[0]):
        for c in [0, 1, 2]:
            print(f"Correlation for {x}, {y}, {c}")

            rectangle = np.copy(images[current_image]['img'][x - rectangle_extend[0]:x + rectangle_extend[0],
                                y - rectangle_extend[1]:y + rectangle_extend[1],
                                c])
            rectangle = rectangle.astype('int16') - rectangle.mean().astype('int16')

            # plt.imshow(rectangle)
            # plt.show()

            likeness_image = correlate2d(images[current_image + 1]['img'][:, :, c].astype('int16') - images[current_image + 1]['img'][:, :, c].mean().astype('int16'), rectangle, boundary='symm', mode='same')

            match_x, match_y = np.unravel_index(np.argmax(likeness_image), likeness_image.shape)

            print(f"x={match_x}, y={match_y}; certainty?: {(likeness_image[match_x, match_y] - likeness_image.mean()) / 100000:3.2f}")

            movement = (x - match_x,
                        y - match_y)

            print(f"Movement={movement}")

            movement_field_mag[x, y] += (movement[0]**2 + movement[1]**2)**0.5

            # plt.imshow(likeness_image)
            # plt.plot(match_y, match_x, 'ro')
            # plt.plot(y, x, 'rx')
            # plt.show()

movement_field_mag /= movement_field_mag.max()

plt.imshow(movement_field_mag)
plt.show()