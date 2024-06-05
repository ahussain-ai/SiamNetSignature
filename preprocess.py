import tensorflow as tf 


def read_image(path):
    # Read the image from the file
    img = tf.io.read_file(path)

    # Decode the image as a grayscale image
    img = tf.image.decode_image(img, channels=1)

    # Resize the image to the desired size (112, 112)
    img = tf.image.resize(img, [105, 105])

    # Convert image to uint8 (if necessary)
    img = tf.cast(img, tf.uint8) 

    return img / 255

def read_image_pair(images, label) :

    """read image pair one by one"""
    img1 , img2 = images
    img1 = read_image(img1)
    img2 = read_image(img2)

    return (img1, img2), label


if __name__ == '__main__' : 
    pass