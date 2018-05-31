
import keras.backend as K

from keras.applications import mobilenet, inception_v3, inception_resnet_v2, resnet50
from keras.callbacks import CSVLogger, EarlyStopping
from keras.layers import Dense, Dropout, GlobalAveragePooling2D, Input
from keras.models import Model
from keras.preprocessing.image import ImageDataGenerator
from keras.optimizers import RMSprop
from keras.regularizers import l2
from keras.utils import np_utils


'''**********************************Parameters**********************************************************************'''

# datagen parameters
img_height = 224
img_width = 224
n_classes = 25

# specify location of freiburg groceries dataset, along with train and val folders
root_dir = "data/product_category"
train_dir = "/train"
val_dir = "/val"

# base network
base_id = 1 # 0 = Inception-v3, 1 = MobileNet, 2 = Inception-ResNet-v2, 3 = ResNet50

# network hyperparameters
n_epochs = 100
n_batches = 32
learning_rate = 0.001
dr = 0.5
wd = 0.0001

# save parameters
date_dir = ' ' # specify save directory for results using YYYY-MM-DD format
n_rounding = 4
parameter_type =' ' # specify parameter (learning_rate, dr, wd) that is being evaluated

enable_model_save = True
weight_decay_test = True
dropout_test = False
learning_rate_test = False
use_previous_weights = False

'''******************************************************************************************************************'''


'''**********************************Functions***********************************************************************'''

# add 2 fully-connected layers to top of base network
def create_dense_model(input_shape,weight_decay, r_dropout):

    input_layer = Input(shape=input_shape)

    top = GlobalAveragePooling2D()(input_layer)
    top = Dense(units=2048, activation='relu', kernel_regularizer=None, name='fc_1')(top)

    predictions = Dense(units=n_classes, activation='softmax', kernel_regularizer=l2(l=weight_decay), name='softmax')(top)

    model_combined = Model(inputs=input_layer, outputs=predictions, name='InceptionV3')

    if use_previous_weights is True:
        print("Loading previous weights")
        model_combined.load_weights(filepath='weights/2018_02_16/inception_wd=0.0001.h5', by_name=True)

    return model_combined


# generate file names for csv and weight files
def save_filenames(base_network_name, test_descriptor, param, precision):

    descriptor = '/' + base_network_name + '_' + test_descriptor + '='

    name_csv = 'results/' + date_dir + descriptor + str(round(param, precision)) + '.csv'
    name_weights = 'weights/' + date_dir + descriptor + str(round(param, precision)) + '.h5'

    return name_csv, name_weights

'''******************************************************************************************************************'''

'''**********************************Main****************************************************************************'''

if __name__==("__main__"):

    K.clear_session()

    # create train/val generators
    datagen_train = ImageDataGenerator(rescale=1./255, shear_range=0.2, rotation_range=10, horizontal_flip=True)
    datagen_val_test = ImageDataGenerator(rescale=1./255)

    train_generator = datagen_train.flow_from_directory(directory=root_dir+train_dir,target_size=(224,224),
                                                        batch_size=n_batches,shuffle=False)

    val_generator = datagen_val_test.flow_from_directory(directory=root_dir+val_dir, target_size=(224, 224),
                                                         batch_size=n_batches, shuffle=False)

    # find y_train/y_val/y_test
    y_train = train_generator.classes
    y_val = val_generator.classes


    # convert output to one-hot encoding
    y_train = np_utils.to_categorical(y=y_train,num_classes=n_classes)
    y_val = np_utils.to_categorical(y=y_val, num_classes=n_classes)
    n_train = y_train.shape[0]
    n_val = y_val.shape[0]


    # base network = Inception-v3
    if base_id is 0:
        base = inception_v3.InceptionV3(input_shape=(224, 224, 3), include_top=False, weights='imagenet')
        base_output_shape = (5,5,2048)
        base_name = 'Inception_v3'

    # base network = MobileNet
    elif base_id is 1:
        base = mobilenet.MobileNet(input_shape=(224, 224, 3), include_top=False, weights='imagenet')
        base_output_shape = (7,7,1024)
        base_name = 'MobileNet'

    # base network = Inception-ResNet-v2
    elif base_id is 2:
        base = inception_resnet_v2.InceptionResNetV2(input_shape=(224, 224, 3), include_top=False, weights='imagenet')
        base_output_shape = (5,5,1536)
        base_name = 'Inception_ResNet_v2'

    # base network = ResNet50
    else:
        base = resnet50.ResNet50(input_shape=(224, 224, 3), include_top=False, weights='imagenet')
        base_output_shape = (1,1,2048)
        base_name = 'ResNet50'


    # generate features
    x_train_features = base.predict_generator(generator=train_generator, steps=n_train/n_batches, verbose=1)
    x_val_features = base.predict_generator(generator=val_generator, steps=n_val/n_batches, verbose=1)


    # correct y_train / y_val / y-test lengths
    l_train = x_train_features.shape[0]
    l_val = x_val_features.shape[0]
    y_train = y_train[0:l_train]
    y_val = y_val[0:l_val]


    # display hyperparameter summary
    print("Dataset Summary:")
    print ("\tX-Train: "+str(x_train_features.shape[0]))
    print ("\tX-Validation: "+str(x_val_features.shape[0]))
    print ("\tY-Train: "+str(y_train.shape[0]))
    print ("\tY-Validation: "+str(y_val.shape[0]))

    # display hyperparameter summary
    print("Hyperparameter Summary:")
    print ("\tWeight decay: "+str(wd))
    print ("\tDropout: "+str(dr))
    print ("\tLearning rate: "+str(learning_rate))
    print ("\t# of epochs: "+str(n_epochs))
    print ("\tBatch size: "+str(n_batches))


    # create dense model on top of base network
    model = create_dense_model(input_shape=base_output_shape, weight_decay=wd, r_dropout=dr)


    # compile model
    model.compile(optimizer=RMSprop(lr=learning_rate), loss='categorical_crossentropy', metrics=['accuracy'])


    # create save filenames
    if weight_decay_test is True:
        parameter_type = 'wd'
    elif dropout_test is True:
        parameter_type = 'dr'
    elif learning_rate_test is True:
        parameter_type = 'lr'

    file_csv, file_trained_weights = save_filenames(base_name, parameter_type, wd, n_rounding)


    # train model
    csv_logger = CSVLogger(filename=file_csv, append=True, separator=',')
    early_stopper = EarlyStopping(monitor='val_loss', patience=20)
    training_history = model.fit(x=x_train_features,
                                 y=y_train,
                                 batch_size=n_batches,
                                 epochs=n_epochs,
                                 validation_data=(x_val_features, y_val),
                                 steps_per_epoch=None,
                                 shuffle=True,
                                 callbacks=[csv_logger])


    # save results
    if enable_model_save == True:
        model.save_weights(filepath=file_trained_weights)




'''******************************************************************************************************************'''