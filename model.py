
import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Conv2D, MaxPooling2D, Flatten, Dense

def build_base_network(input_shape):

    input = Input(shape=input_shape)

    #block 1
    x = Conv2D(64, (3, 3),activation='relu')(input)
    x = MaxPooling2D(pool_size=(2, 2),strides = 2)(x)

    #block 2
    x = Conv2D(128, (3, 3),activation='relu')(x)
    x = MaxPooling2D(pool_size=(2, 2),strides = 2)(x)

    #block 3
    x = Conv2D(256, (3, 3),activation='relu')(x)
    x = MaxPooling2D(pool_size=(2, 2),strides = 2)(x)

    #block 3
    x = Conv2D(128, (3, 3),activation='relu')(x)
    x = MaxPooling2D(pool_size=(2, 2),strides = 2)(x)

    #block 4
    x = Conv2D(128, (4, 4))(x)
    x = Flatten()(x)

    X = Dense(4096, activation='relu')(x)

    return Model(input, X)

class L1Dist(tf.keras.layers.Layer) :

  def __init__(self,**kwargs) :
    super(L1Dist, self).__init__(**kwargs)

  def call(self,embedding_a, embedding_b) :
    return tf.math.abs(embedding_a - embedding_b)


def make_siamese_model(inp_shape=(105,105,1)) :

  input_shape = inp_shape 
  base_network = build_base_network(input_shape)
  print(base_network.summary())

  input_a = Input(shape=input_shape)
  input_b = Input(shape=input_shape)

  processed_a = base_network(input_a)
  processed_b = base_network(input_b)

  distance_layer = L1Dist()
  distance = distance_layer(processed_a, processed_b)

  dense = Dense(1, activation ='sigmoid')(distance)

  siamese_network = Model([input_a, input_b], dense)
  return siamese_network

if __name__ == '__main__' : 
  pass