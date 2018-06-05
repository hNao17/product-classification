import cv2
import numpy as np

from glob import glob

from keras.preprocessing.image import ImageDataGenerator


'''**********************************Parameters**********************************************************************'''

# specify locations of freiburg groceries and augmented image folders
root_dir = "data/product_category/original/"

# text file containing all class names in freiburg groceries
class_list_filename = "freiburg_groceries_classList.txt"

# target images per class
imgs_per_class = 400

'''******************************************************************************************************************'''

'''**********************************Functions***********************************************************************'''

# create a list of training image directory locations,class labels and image file formats
def create_class_list(list_filename):

    with open(list_filename, 'r') as f:
        list = []

        counter = 0
        for line in f.readlines():
            category, newLine = line.strip().split(',')
            list.append(category)
            counter+=1

    return list

# create numpy array of images from a specified class directory
def create_img_batch(folder_name):

    img_array=[]

    counter=0
    for fn in glob(folder_name+'/*.'+'png'):

        img_cv = cv2.imread(fn)
        img_cv = cv2.resize(img_cv,(224,224),img_cv)
        img_cv = cv2.cvtColor(img_cv, cv2.COLOR_BGR2RGB)

        img_array.append(img_cv)
        counter+=1

    batch = np.array(img_array)

    return batch
'''******************************************************************************************************************'''


'''**********************************Main****************************************************************************'''

# retrieve list of available class names & quantity of images / class
class_list = create_class_list(class_list_filename)

# data generator
datagen = ImageDataGenerator(width_shift_range=0.1, height_shift_range= 0.1, shear_range=0.05)

# open each class directory
for category in class_list:

    class_dir = root_dir + category
    x_batch = create_img_batch(class_dir)

    new_img_counter = 0
    n_img_max = imgs_per_class-x_batch.shape[0]

    print("\nClass: %s" %category)
    print("# of Images: %i " %x_batch.shape[0])

    # generate augmented images until n_old + n_aug = 400
    for batch in datagen.flow(x=x_batch,batch_size=1,save_to_dir=class_dir,save_prefix=category):
        new_img_counter+=1

        if(new_img_counter >= n_img_max):
            break

    print("Created %i new images " %new_img_counter)

'''******************************************************************************************************************'''