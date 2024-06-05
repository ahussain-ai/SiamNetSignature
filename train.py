from tensorflow.keras.losses import BinaryCrossentropy
from tensorflow.keras.metrics import BinaryAccuracy, Mean
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam

from datasets import * 
from model import make_siamese_model

class SiameseModel(Model):
    """The Siamese Network model with a custom training and testing loops.

    Computes the triplet loss using the three embeddings produced by the
    Siamese Network.
    """

    def __init__(self, siamese_network):
        super().__init__()
        self.siamese_network = siamese_network
        self.acc_tracker = BinaryAccuracy(name="accuracy")
        self.loss_tracker = Mean(name="loss")
        self.loss_fn = BinaryCrossentropy()

    def compile(self, optimizer):
        super().compile()
        self.optimizer = optimizer

    def call(self, inputs):
        return self.siamese_network(inputs)

    def train_step(self, data):
        (images1, images2), actual = data

        with tf.GradientTape() as tape:
            predicted = self.siamese_network([images1, images2])
            loss = self.loss_fn(actual, predicted)

        gradients = tape.gradient(loss, self.siamese_network.trainable_weights)
        self.optimizer.apply_gradients(
            zip(gradients, self.siamese_network.trainable_weights)
        )

        self.loss_tracker.update_state(loss)
        self.acc_tracker.update_state(actual, predicted)
        return {"loss": self.loss_tracker.result(), "accuracy": self.acc_tracker.result()}

    def test_step(self, batch):
        (images1, images2), actual = batch

        predicted = self.siamese_network([images1, images2])
        loss = self.loss_fn(actual, predicted)

        self.loss_tracker.update_state(loss)
        self.acc_tracker.update_state(actual, predicted)
        return {"loss": self.loss_tracker.result(), "accuracy": self.acc_tracker.result()}

    @property
    def metrics(self):
        return [self.loss_tracker, self.acc_tracker]


def train_siamese_network(dataset_dir) : 

    dataset_dir = dataset_dir

    #Dataset
    original , forged = load_data(dataset_dir)
    (anc,anc_y), (neg, neg_y) = pairs(original, forged)

    train_set, test_set = data_pipeline(anc, anc_y, neg, neg_y)

    #model definition 
    siamese_network = make_siamese_model()
    

    siamese_model = SiameseModel(siamese_network)
    siamese_model.compile(optimizer= Adam())
    history = siamese_model.fit(train_set, epochs=5, validation_data=test_set)
    return history 