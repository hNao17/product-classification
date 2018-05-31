import collections
import itertools
import numpy as np
import matplotlib.pyplot as plt


from keras.applications import mobilenet, inception_v3, resnet50, inception_resnet_v2
from keras.layers import Dense, GlobalAveragePooling2D
from keras.models import Model
from keras.preprocessing.image import ImageDataGenerator
from keras.optimizers import RMSprop
from keras.regularizers import l2

from sklearn.metrics import confusion_matrix

'''*****************************Parameters***************************************************************************'''

# specify location of freiburg groceries data and test folders
root_dir = 'data/product_brand'
test_dir ='/test'

img_height = 224
img_width = 224

# saved training weights file to be loaded
weights_filename = '.h5'

# model output
n_classes = 25

# hyperparameters
wd = 0.0001
learning_rate = 0.001

# confusion matrix labels
model_name = 'MobileNet'
dataset_name = 'Freiburg Groceries'

'''******************************************************************************************************************'''


'''*****************************Functions****************************************************************************'''

# create custom model with trained FC layer weights
def create_model(img_size,model_type,base_name):

    if model_type == 0:
        print("Creating MobileNet model")
        base = mobilenet.MobileNet(input_shape=img_size,include_top=False,weights='imagenet')

    elif model_type == 1:
        print("Creating InceptionV3 model")
        base = inception_v3.InceptionV3(input_shape=img_size,include_top=False,weights='imagenet')

    elif model_type == 2:
        print("Creating Resnet50 model")
        base = resnet50.ResNet50(input_shape=img_size,include_top=False,weights='imagenet')

    elif model_type == 3:
        print("Creating InceptionResNet-V2 model")
        base = inception_resnet_v2.InceptionResNetV2(input_shape=img_size,include_top=False,weights='imagenet')

    top = base.output
    top = GlobalAveragePooling2D()(top)

    top = Dense(units=2048,activation='relu',kernel_regularizer=None,name='fc_1')(top)
    predictions = Dense(units=n_classes,activation='softmax',kernel_regularizer=l2(l=wd),name='softmax')(top)

    model_combined = Model(inputs=base.input, outputs=predictions, name=base_name)

    path_to_weights = 'weights/'+weights_filename
    model_combined.load_weights(filepath=path_to_weights,by_name=True)
    print('Loading weights from ' + path_to_weights)

    return model_combined


# creates confusion matrix comparing predicted and ground truth results
def plot_confusion_matrix(cm, classes, normalize=False, title='Confusion matrix', cmap=plt.cm.Blues):

    if normalize:
        cm = 100*cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        np.savetxt(fname='normalized_cf.txt',X=cm,fmt='%1.1f')
        print("Normalized confusion matrix")
    else:
        print('Confusion matrix, without normalization')

    print(cm)

    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title, fontweight='bold', fontsize=14)
    plt.colorbar()
    tick_marks = np.arange(len(classes))

    plt.xticks(tick_marks, classes, fontsize=8, rotation=90)
    plt.yticks(tick_marks, classes, fontsize=8)

    fmt = '.1f' if normalize else 'd'
    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, format(cm[i, j], fmt), horizontalalignment="center",
                 color="white" if cm[i, j] > thresh else "black", fontsize=7)

    plt.tight_layout()
    plt.ylabel('True label', fontweight='bold', fontsize=12)
    plt.xlabel('Predicted label', fontweight='bold', fontsize=12)


'''*************************************Main*************************************************************************'''

if __name__==("__main__"):

    # import test images
    import_location = root_dir+test_dir
    image_datagen = ImageDataGenerator(rescale=1./255)
    test_generator = image_datagen.flow_from_directory(directory=import_location,
                                                       target_size=(224,224),
                                                       batch_size=1,
                                                       shuffle=False)
    print(test_generator.class_indices)

    od = collections.OrderedDict(sorted(test_generator.class_indices.items(), key=lambda t: t[1]))

    # create list of class names
    class_names=[]
    for name in od:
        class_names.append(name)

    # ground truth output
    y_test = test_generator.classes


    # create CNN
    model = create_model(img_size=(224,224,3),model_type=0,base_name=model_name)

    # compile CNN
    model.compile(optimizer=RMSprop(lr=learning_rate), loss='categorical_crossentropy', metrics=['accuracy'])

    # evaluate test set
    loss, accuracy = model.evaluate_generator(generator=test_generator,
                                              steps=len(test_generator),
                                              max_queue_size=100)

    print("Test loss:"+str(loss))
    print("Test accuracy"+str(accuracy))


    # predict classes
    prediction = model.predict_generator(generator=test_generator,
                                         steps=len(test_generator),
                                         max_queue_size=100,
                                         verbose=1)

    np.set_printoptions(precision=3,suppress=True)
    prediction = (prediction > 0.5)
    prediction_idxs = np.argmax(prediction,axis=1)
    print("Predictions-Index Form:"+str(prediction_idxs))

    # generate confusion matrix data
    cm = confusion_matrix(y_true=y_test,y_pred=prediction_idxs)
    print(cm)

    graph_title = 'Product Category Accuracy - ' + dataset_name

    # plot normalized confusion matrix
    plt.figure()
    plot_confusion_matrix(cm, classes=class_names, normalize=True, title=graph_title)

    plt.show()

'''******************************************************************************************************************'''