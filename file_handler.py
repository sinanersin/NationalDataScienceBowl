from PIL import Image
import PIL
import os
import csv
import numpy as np
import sys

def remove_whitespace(img, axis):
    """
    Returns image without whitespace along given axis.
    """

    del_index = []

    # create numpy boolean array with True if row/col is all white
    whitespace = np.all(img==255, axis=axis)

    # remember index of whitespace starting from begin till non-whitespace
    for each in range(len(whitespace)):
        if whitespace[each]:
            del_index.append(each)
        else:
            break

    # remember index of whitespace starting from end till non-whitespace
    for each in range(len(whitespace) - 1, 0, -1):
        if whitespace[each]:
            del_index.append(each)
        else:
            break

    # delete whitespaces
    output = np.delete(img, del_index, int(not axis))

    return output


def transform_image(image, shape):
    """
    Returns transformed image.

    First, deletes whitespace on the edges of the image to keep maximum
    information in image.
    Second, resize image and add padding to keep aspect ratio and give
    image cube-shape.
    """

    img = Image.open(image)

    # remove whitespace on edges to keep maximal information in image
    # transform to numpy array and back to image for further adjustments
    img = np.asarray(img)
    img = remove_whitespace(img, 1)
    img = remove_whitespace(img, 0)
    img = Image.fromarray(img)

    # resize to max dimension equal to shape and calculate padding to add
    if img.size[0] > img.size[1]:
        wpercent = (shape / float(img.size[0]))
        hsize = max(int((float(img.size[1]) * float(wpercent))), 1)
        img = img.resize((shape, hsize), PIL.Image.ANTIALIAS)
        add = np.empty((img.size[0] - img.size[1], img.size[0]))
        axis = 0

    else:
        wpercent = (shape / float(img.size[1]))
        hsize = max(int((float(img.size[0]) * float(wpercent))), 1)
        img = img.resize((hsize, shape), PIL.Image.ANTIALIAS)
        add = np.empty((img.size[1], img.size[1] - img.size[0]))
        axis = 1

    # add padding to keep aspect ratio and make cube
    add[:] = 255
    data = np.asarray( img, dtype=np.uint8 )
    data = np.append(data, add, axis=axis)

    return data


def load_images(images_path, labels_file=None, num=None, shape=50):
    """
    Returns tuple containing numpy array of images and numpy array of
    labels. Indexes of both numpy arrays correspond to the same image.

    If no labels_file is specified, filenames instead of labels will be
    returned.

    If no number of images (num) specified, all images will be loaded.
    """

    print('Loading data')

    labels = {}
    output = []

    # load labels into dict with image name as key and class as value
    if labels_file:
        with open(labels_file) as inpt:
            reader = csv.reader(inpt, delimiter=',')
            next(reader)
            for line in reader:
                labels[line[0]] = int(line[1])

    # loop for loading images and labels into list
    # by making list of lists of image and label, labels and images
    # are guaranteed to be outputted in same index
    for root, dirs, filenames in os.walk(images_path):
        if not num:
            num = len(filenames)

        # transform images in images_path
        for count, img_name in enumerate(filenames[:num]):
            img_raw = os.path.join(images_path, img_name)
            image = transform_image(img_raw, shape)

            # add images and labels (or image names) together in list
            if labels:
                label = labels[img_name]
            else:
                label = img_name
            output.append([label, image])

            # give user notice of progress
            if (count + 1) % 1000 == 0 or count + 1 == num:
                sys.stdout.write('\r{}/{}'.format(count + 1, num))

    # create numpy arrays for both images and labels
    train_images = np.array([im[1] for im in output])
    train_labels = np.array([im[0] for im in output])

    # standardize image data to floats between 0 and 1
    train_images = train_images.astype('float32') / 255.0

    # reshape arrays to array of arrays
    train_labels = train_labels.reshape((-1, 1))
    train_images = train_images.reshape((-1, 1, shape, shape))

    return (train_labels, train_images)
