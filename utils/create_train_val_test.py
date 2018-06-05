import cv2

from glob import glob
from sklearn.model_selection import train_test_split


'''**********************************Parameters**********************************************************************'''

# dataset locations
src_dir = "data/product_category/original/"
root_dir = "data/product_category/"

class_list_filename = "freiburg_groceries_classList.txt"


# dataset split ratios
p_train = 0.7
p_val = 0.15
p_test = 0.15

'''******************************************************************************************************************'''


'''**********************************Functions***********************************************************************'''

# create a list of dataset classes
def create_class_list(list_filename):

    with open(list_filename, 'r') as f:

        list = []
        counter = 0

        for line in f.readlines():

            category, newLine = line.strip().split(',')
            list.append(category)
            counter+=1

    return list

# create array of images from a specified class directory
def create_trainval_batch(class_name):

    img_array=[]

    print("\nClass: %s" % class_name)

    img_orig_folder_name = src_dir+class_name
    img_counter = 0

    # collect images belonging to specific class from source folder
    for fn in glob(img_orig_folder_name+'/*.'+'png'):

        img_cv = cv2.imread(fn)
        #img_cv = cv2.cvtColor(img_cv, cv2.COLOR_BGR2RGB)

        img_array.append(img_cv)
        img_counter+=1

    print("%s: %i images" %(img_orig_folder_name, img_counter))


    '''*********************************Train/Val/Test split*******************************************'''

    # split into train/val/test sets
    x_train,x_remain  = train_test_split(img_array, shuffle=True, test_size=p_val+p_test, random_state=0)
    x_val, x_test = train_test_split(x_remain, shuffle=True, test_size=p_test/(p_val+p_test) , random_state=0)

    print("x_train:"+str(len(x_train)))
    print("x_val:"+str(len(x_val)))
    print("x_test:"+str(len(x_test)))

    # train/val/test save locations
    save_train = root_dir + 'train/' + class_name + '/'
    save_val = root_dir + 'val/' + class_name + '/'
    save_test = root_dir + 'test/' + class_name + '/'

    # save images to train/val/test folders
    save_images(save_train,class_name,x_train)
    save_images(save_val, class_name, x_val)
    save_images(save_test, class_name, x_test)

    '''************************************************************************************************'''


# save images to specified train/val/test folder
def save_images(save_dir, class_name, x_set):

    for i,img_set in enumerate(x_set):

        if(i < 10):
            padding = '000'

        elif(i >=10 and i < 100):
            padding = '00'

        else:
            padding = '0'

        filename_save = class_name + padding + str(i) + '.png'
        cv2.imwrite(filename=save_dir+filename_save,img=img_set)

    print("Saving %i images to %s " %(len(x_set),save_dir))


'''******************************************************************************************************************'''


'''**********************************Main****************************************************************************'''

if __name__=="__main__":

    # retrieve list of available classes
    class_list = create_class_list(class_list_filename)

    for category in class_list:
        create_trainval_batch(category)

'''******************************************************************************************************************'''