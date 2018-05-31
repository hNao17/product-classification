import cv2
import numpy as np
import random

from glob import glob

from keras import backend as K

from keras.applications import mobilenet, inception_v3, inception_resnet_v2, resnet50
from keras.layers import Dense, dot, GlobalAveragePooling2D, Input, Lambda
from keras.models import Model
from keras.optimizers import RMSprop


'''*************************************Parameters*******************************************************************'''

# data import options
class_list_filename = "siamese_testSet_classes.txt"
root_dir = "data/product"
n_classes = 11
n_trials = 10

# siamese model parameters
img_shape = (224, 224, 3)
feature_shape = (2048,)
base_id = 1  # 0 = Inception-v3, 1 = MobileNet, 2 = InceptionResNet-v2, 3 = ResNet50

# saved training weights to be loaded
weights_filename = '.h5'

# test loop
acc_array = []

'''******************************************************************************************************************'''

'''*************************************Functions********************************************************************'''

# create a list of training image directory locations,class labels and image file formats
def create_class_list(list_filename):

    with open(list_filename, 'r') as f:
        list = []

        counter = 0
        for line in f.readlines():
            category, newLine = line.strip().split(',')
            list.append(category)
            counter += 1

    return list


# create numpy array of images from a specified class directory
def create_test_set(class_list, dataset_type):

    img_array = []
    labels_array = []

    counter = 0

    for i, class_name in enumerate(class_list):
        folder_name = root_dir + dataset_type + "/" + class_name

        for fn in glob(folder_name + '/*.' + 'png'):

            img_cv = cv2.imread(fn)
            img_cv = cv2.resize(img_cv, (224, 224), img_cv)
            img_cv = cv2.cvtColor(img_cv, cv2.COLOR_BGR2RGB)

            img_array.append(img_cv)
            labels_array.append(i)
            counter += 1

        batch = np.array(img_array)

    return img_array, labels_array


# create positive / negative training example pairs
def create_pairs(x, idcs):

    pairs = []
    labels = []  # 1 = positive pair, 0 = negative pair

    # find the minimum number of examples per class
    n_min = (min([len(idcs[d]) for d in range(n_classes)]) - 1)

    for n in range(n_classes):

        counter_pos = 0
        counter_neg = 0

        # create n_min positive pairs and n_min negative pairs per class
        for i in range(n_min):

            # positive example
            z1, z2 = idcs[n][i], idcs[n][i + 1]
            pairs += [[x[z1], x[z2]]]
            counter_pos += 1

            # negative example
            inc = random.randrange(1, n_classes)
            n_rand = (n + inc) % n_classes
            z1, z2 = idcs[n][i], idcs[n_rand][i]
            pairs += [[x[z1], x[z2]]]
            counter_neg += 1

            labels += [1, 0]

    return np.array(pairs, dtype=np.float32), np.array(labels)


# create a base model that generates a (n,1) feature vector for an input image
def create_base_model(input_shape, id):

    # inception v3
    if base_id is 0:
        base = inception_v3.InceptionV3(input_shape=input_shape, weights='imagenet', include_top=False)
        base_name = 'Inception-V3'

    elif base_id is 1:
        base = mobilenet.MobileNet(input_shape=input_shape, weights='imagenet', include_top=False)
        base_name = 'MobileNet'

    elif base_id is 2:
        base = inception_resnet_v2.InceptionResNetV2(input_shape=input_shape, weights='imagenet', include_top=False)
        base_name = 'InceptionResNet-v2'

    elif base_id is 3:
        base = resnet50.ResNet50(input_shape=input_shape, weights='imagenet', include_top=False)
        base_name = 'ResNet50'

    print("\nBase Network: %s" % base_name)

    top = GlobalAveragePooling2D()(base.output)

    # freeze all layers in the base network
    for layers in base.layers:
        layers.trainable = False

    model = Model(inputs=base.input, outputs=top, name='base_model')

    return model


# calculate l1_norm b/t feature vector outputs from base network
def l1_norm(feat_vects):

    x1, x2 = feat_vects

    result = K.maximum(x=K.sum(x=K.abs(x1-x2), axis=1, keepdims=True), y=K.epsilon())

    return result


# calculate l2_distance b/t feature vector outputs from base network
def l2_distance(feat_vects):

    x1, x2 = feat_vects

    result = K.sqrt(K.sum(x=K.square(x1 - x2), axis=1, keepdims=True))

    return result


# calculate cosine distance b/t feature vector outputs from base network
def cos_distance(feat_vects):

    x1, x2 = feat_vects

    result = dot(inputs=[x1, x2], axes=1, normalize=True)

    return result


# create a siamese model
def create_siamese(base, input_shape):

    input_a = Input(shape=input_shape)
    input_b = Input(shape=input_shape)

    encoding_a = base(input_a)
    encoding_b = base(input_b)

    fc1_a = Dense(units=2048, activation='relu', kernel_regularizer=None, name='fc1_a')(encoding_a)
    fc1_b = Dense(units=2048, activation='relu', kernel_regularizer=None, name='fc1_b')(encoding_b)

    # distance = Lambda(function=l1_distance, name='l1_distance', )([fc1_a, fc1_b])
    #distance = Lambda(function=l2_distance, name='l2_distance', )([fc1_a, fc1_b])
    distance = Lambda(function=cos_distance, name='cos_distance', )([fc1_a, fc1_b])

    prediction = Dense(units=1, activation='sigmoid', name='sigmoid')(distance)

    model = Model(inputs=[input_a, input_b], outputs=prediction, name='siamese_model')

    # load weights for sigmoid layers
    path_to_weights = "weights/" + weights_filename
    model.load_weights(filepath=path_to_weights, by_name=True)
    print('Loading weights from ' + path_to_weights)

    return model

# compute accuracy for siamese model
def compute_accuracy_binary(y_true, y_pred):

    prediction = y_pred.ravel() > 0.5

    return np.mean(prediction == y_true)

'''******************************************************************************************************************'''

'''*************************************Main*************************************************************************'''

if __name__ == ("__main__"):

    K.clear_session()

    class_list = create_class_list(class_list_filename)

    # import original test set
    x_test_orig, y_test_orig = create_test_set(class_list, "test")
    x_test = np.array(x_test_orig, dtype=np.float32)
    y_test = np.array(y_test_orig, dtype=np.float32)
    y_test = np.reshape(y_test_orig, newshape=(y_test.shape[0], 1))
    print("X_test: "+str(x_test.shape))
    print("Y-test: "+str(y_test.shape))

    # find indices in loaded images in x_test corresponding to each class
    idcs_train = [np.where(y_test == i)[0] for i in range(n_classes)]

    # normalize test set for Inception-v3, Mobilenet, Inception-ResNet-v2
    if base_id is 0 or base_id is 1 or base_id is 2:
        x_test /= 255
        print("Normalizing test b/t [0,1]")

    # create base model
    base_network = create_base_model(input_shape=img_shape, id=base_id)

    # create siamese network
    siamese_model = create_siamese(base=base_network, input_shape=img_shape)

    # test model with binary_cross-entropy loss function
    siamese_model.compile(optimizer=RMSprop(), loss='binary_crossentropy', metrics=['accuracy'])

    # find model mean accuracy over n trials
    for i in range(n_trials):
        print("\nTrial %i:" % i)

        # create train & test pairs from original sets
        x_test_pair, y_test_pair = create_pairs(x_test, idcs_train)
        print("# of Pairs: "+str(x_test_pair.shape[0]))

        predictions = siamese_model.predict(x=[x_test_pair[:, 0], x_test_pair[:, 1]], batch_size=1, verbose=1)

        # calculate model accuracy
        test_acc = compute_accuracy_binary(y_true=y_test_pair, y_pred=predictions)
        print("Test Accuracy: %f" % test_acc)

        acc_array.append(test_acc)

    acc_np_array = np.array(acc_array, dtype=np.float32)
    final_acc = np.mean(acc_np_array)

    print("\nFinal Accuracy: "+str(final_acc))


