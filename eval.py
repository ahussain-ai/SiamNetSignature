
import tensorflow as tf 
import numpy as np
from sklearn.metrics import confusion_matrix


def evaluate(model, test_data):
    """
    This function evaluates a Keras model on a test dataset.

    Args:
        model: A Keras model.
        test_data: A tf.data.Dataset object containing the test data.

    Returns:
        A tuple of (loss, accuracy, precision, recall) values.
    """
    loss = tf.keras.metrics.Mean()
    accuracy = tf.keras.metrics.BinaryAccuracy()
    precision = tf.keras.metrics.Precision()
    recall = tf.keras.metrics.Recall()

    for images, labels in test_data:
        predictions = model(images)
        # Reshape labels to match predictions shape
        labels = tf.reshape(labels, (-1, 1))
        loss(tf.keras.losses.binary_crossentropy(labels, predictions))
        accuracy.update_state(labels, predictions)
        precision.update_state(labels, predictions)
        recall.update_state(labels, predictions)

    return loss.result(), accuracy.result(), precision.result(), recall.result()



def plot_confusion_matrix(model, test_data):
    """
    This function plots the confusion matrix of a Keras model on a test dataset.

    Args:
        model: A Keras model.
        test_data: A tf.data.Dataset object containing the test data.
    """
    # Get predictions and labels
    predictions = []
    labels = []
    for images, label in test_data:

        predictions.extend(model(images).numpy())
        labels.extend(label.numpy())

    # Convert predictions to binary values
    predictions = np.where(np.array(predictions) > 0.5, 1, 0)

    # Create and plot the confusion matrix
    cm = confusion_matrix(labels, predictions)
    print(f"Confusion matrix \n{cm}")
