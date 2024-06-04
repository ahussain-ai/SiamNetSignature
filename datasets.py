import numpy as np 
import glob 
import random
from itertools import product, combinations
from preprocess import read_image_pair
import tensorflow as tf
from preprocess import read_image_pair


def load_data(dataset_dir) : 

    """Arrange the signatures in two folers, and writer wise """

    forged = {}

    for idx in range(1, 56):
        files = glob.glob(f"{dataset_dir}/full_forg/forgeries_{idx}_*.*")
        forged[f"forged_{idx}"] = files

    originals = {}
    for idx in range(1, 56):
        files = glob.glob(f"{dataset_dir}/full_org/original_{idx}_*.*")
        originals[f"original_{idx}"] = files

    return originals, forged 

def pair_generate(original, forged, num_pairs =  200 ) :

    """generate pair from two list by taking one of each

    Arguments

        original --> list of images containing original signature of the writer
        forged --> list of images containing forged signatures of the writer
        num_pair --> number of pairs for each writer

    returns -->
        anchor --> List of tuples containing two original signature from the writer
        negative --> ............................one original and one forged

    """

    anc_ = random.sample ( list ( combinations(original, 2) ), num_pairs)
    neg_ =  random.sample( list(product(original, forged)), num_pairs)

    return anc_, neg_

def pairs(original, forged) :

    """
    Arguments :

     original --> Dictionary containing  paths to the original signature of each writer ( 55 writer having 24 signatures each)
     forged --> Dictionary contained paths to the forged signature of each writer ( 55 writer having 24 signatures each)

    Returns :
    anchor, anchor_y : original signature pair with label of each [1,1,1,...  ]
    negative, negative_y: forged signature pair, with label [0,0,0,......]

    """

    anchor = []
    negative = []

    for key1, key2 in zip(original, forged) :
        anc, neg =  pair_generate(original[key1], forged[key2])
        anchor.extend( anc )
        negative.extend( neg )

    anchor_y = np.ones(len(anchor))
    negative_y = np.zeros(len(negative))

    return ( anchor, anchor_y ), (negative, negative_y)

def data_pipeline(anchor, anchor_y,negative, negative_y, batch_size=128) : 

    anchor_images_1 = np.array([pair[0] for pair in anchor])
    anchor_images_2 = np.array([pair[1] for pair in anchor])

    negative_images_1 = np.array([pair[0] for pair in negative])
    negative_images_2 = np.array([pair[1] for pair in negative])


    # Combine anchor and negative data
    all_images_1 = np.concatenate([anchor_images_1, negative_images_1], axis=0)
    all_images_2 = np.concatenate([anchor_images_2, negative_images_2], axis=0)
    all_labels = np.concatenate([anchor_y, negative_y], axis=0)

    # Create a TensorFlow dataset
    dataset = tf.data.Dataset.from_tensor_slices(((all_images_1, all_images_2), all_labels))

    # Shuffle, batch, and prefetch the dataset
    dataset = dataset.shuffle(buffer_size=2*len(anchor))
    dataset = dataset.map(read_image_pair)
    dataset = dataset.prefetch(buffer_size=tf.data.experimental.AUTOTUNE)

    #split data into train and set 
    train_size = int(0.8 * len(anchor) * 2)
    test_size = len(anchor)*2 - train_size 

    print("Training data size {train_size}")
    print("Test data size {test_size}")

    train_dataset = dataset.take(train_size)
    test_dataset = dataset.skip(train_size)

    train_dataset = train_dataset.batch(batch_size)
    test_dataset = test_dataset.batch(batch_size)

    return (train_dataset, test_dataset)


if __name__ == "__main__" : 
    pass