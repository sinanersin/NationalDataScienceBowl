import numpy as np
import random
from keras.utils import np_utils
from model import create_model
from itertools import product, combinations
import json


def cross_validation(parameters, X, Y, folds=1, datagen=None, batch_size=50,
                     epochs=10):
    """
    Returns a list of dictionaries, each dictionary containing the accuracy
    history, the loss history, the accuracy of model evaluation and the loss
    the of model evaluation, all averaged over all folds.

    Parameters is a dict with as key the parameter to change and as value
    a list of all values this parameter needs to be tested against.
    """

    output = []

    # create list of dicts with all possible parameter combinations
    parameters = [dict(zip(parameters, x))
                  for x in product(*parameters.values())]
    x_folds, y_folds = k_folds(X, Y, folds=folds)

    fold_combis = list(combinations(range(folds), folds-1))

    # first iterate over all parameter combinations, then all fold combinations
    for param_count, param in enumerate(parameters):
        mid_hist = []
        mid_accu = []
        mid_loss = []

        for fold_count, combi in enumerate(fold_combis):

            # keep track of progress
            update_message = 'Parameter {}/{} || Fold {}/{}'\
                              .format(param_count+1, len(parameters),
                                      fold_count+1, len(fold_combis))
            print(update_message)

            # seperate combination of folds into test and train data
            x_train, y_train, x_test, y_test = combine_folds(x_folds, y_folds,
                                                             combi)

            # create new model and run with datagen if specified else normal
            model = create_model(input_shape=x_train[0].shape, **param)

            if datagen:
                hist = model.fit_generator(datagen.flow(x_train, y_train,
                                                        batch_size=batch_size),
                                           steps_per_epoch=len(x_train)/batch_size,
                                           epochs=epochs)
            else:
                hist = model.fit(x_train, y_train, batch_size=batch_size,
                                 epochs=epochs, verbose=1)

            score = model.evaluate(x_test, y_test, verbose=1)

        # save information to be returned    
            mid_hist.append(hist)
            mid_accu.append(score[1])
            mid_loss.append(score[0])

        outcome = {}
        outcome['hist_acc'] = [sum(x) /float(len(x)) for x in zip(*[y.history['acc'] for y in mid_hist])]
        outcome['hist_loss'] = [sum(x) /float(len(x)) for x in zip(*[y.history['loss'] for y in mid_hist])]
        outcome['accuracy'] = sum(mid_accu) / len(mid_accu)
        outcome['loss'] = sum(mid_loss) / len(mid_loss)
        outcome['parameters'] = param
        output.append(outcome)

    return output


def k_folds(X, Y, folds=3):
    """
    Return 2 list of folds, for images (x_folds) and the labels (y_folds).
    Each list contains number of lists according to parameter 'sets'.

    All folds are equally distributed containing (almost) equal amount of
    each label.
    """
    x_folds = []
    y_folds = []

    # append list to x_ and y_folds for each set
    for i in range(folds):
        x_folds.append([])
        y_folds.append([])

    unique_labels = np.unique(Y)

    # iterate over unique labels to create equal distributed folds
    for label in unique_labels:
        fold = random.randint(0, folds-1)
        label_indices = np.where(np.any(Y==label, axis=1))[0]
        random.shuffle(label_indices)

        # assign each index to next fold
        for index in label_indices:
            x_folds[fold].append(X[index])
            y_folds[fold].append(Y[index])
            fold = (fold + 1) % folds

    return (x_folds, y_folds)

def combine_folds(x_folds, y_folds, train_indices):
    """
    Return 4 lists; trainingset for x and y, testset for x and y.

    Takes as input the folds for both x and y and a list of indices which
    should be combined as the trainingset. The leftover index will be the
    index for the testset.
    """

    train_x_folds = [x_folds[i] for i in train_indices]
    train_y_folds = [y_folds[i] for i in train_indices]

    current_train_x = np.concatenate(train_x_folds)
    current_train_y = np.concatenate(train_y_folds)

    current_train_y = np_utils.to_categorical(current_train_y, 121)

    current_test_x = np.array([fold for fold in x_folds
                               if fold not in train_x_folds][0])
    current_test_y = [fold for fold in y_folds
                      if fold not in train_y_folds][0]

    current_test_y = np_utils.to_categorical(current_test_y, 121)

    return (current_train_x, current_train_y, current_test_x, current_test_y)
