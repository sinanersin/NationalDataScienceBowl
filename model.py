from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation, Flatten, LeakyReLU
from keras.layers import Convolution2D, MaxPooling2D

def create_model(input_shape=(1, 56, 56), relu=0.2,
                 loss_function='categorical_crossentropy',
                 optimizer='adam',
                 conv1_neurons=64, conv1_kernel=(11,11),
                 maxpool1_pool=(3,3),
                 conv2_neurons=64, conv2_kernel=(3,3),
                 maxpool2_pool=(3,3),
                 dropout1_drop=0.3,
                 conv3_neurons=64, conv3_kernel=(3,3),
                 dropout2_drop=0.4,
                 conv4_neurons=256, conv4_kernel=(3,3),
                 maxpool3_pool=(3,3),
                 dense1_neurons=256,
                 dense2_neurons=128):
    """
    Returns the standard model. Takes as input the parameters per layer
    and to compile the model.

    If 'relu>0' the activation function used will be LeakyReLU, else it is
    just ReLu.

    For the optimizer, 'adam' is the standard 
    """
    # instantiate model
    model = Sequential()

    # conv1_layer
    model.add(Convolution2D(conv1_neurons, conv1_kernel, input_shape=input_shape,
                            data_format='channels_first', padding='same'))
    model.add(LeakyReLU(alpha=relu))

    #maxpool1_layer
    model.add(MaxPooling2D(pool_size=(maxpool1_pool)))

    # conv2_layer
    model.add(Convolution2D(conv2_neurons, conv2_kernel,
                            data_format='channels_first', padding='same'))
    model.add(LeakyReLU(alpha=relu))

    # maxpool2_layer
    model.add(MaxPooling2D(pool_size=maxpool2_pool))

    # dropout1_layer
    model.add(Dropout(dropout1_drop))

    # conv3_layer
    model.add(Convolution2D(conv3_neurons, conv3_kernel,
                            data_format='channels_first', padding='same'))
    model.add(LeakyReLU(alpha=relu))

    # dropout2_layer
    model.add(Dropout(dropout2_drop))

    # conv4_layer
    model.add(Convolution2D(conv4_neurons, conv4_kernel,
                            data_format='channels_first', padding='same'))
    model.add(LeakyReLU(alpha=relu))

    # maxpool3_layer
    model.add(MaxPooling2D(pool_size=maxpool3_pool))

    # no parameters to be set for this layer
    model.add(Flatten())

    # dense1_layer
    model.add(Dense(dense1_neurons))
    model.add(LeakyReLU(alpha=relu))

    # dense2_layer
    model.add(Dense(dense2_neurons))
    model.add(LeakyReLU(alpha=relu))

    # output_dense_layer, no parameters to be set
    # 121 neurons for each category 1. Softmax activation to fill 1 neuron only
    model.add(Dense(121, activation='softmax'))

    # compile model
    model.compile(loss=loss_function, optimizer=optimizer, metrics=['acc'])

    return model
