import os
from keras.utils import to_categorical
import simplejson
import numpy as np
from PIL import Image
import glob
import tensorflow as tf

class DataLoader:
    def __init__(self, path, extension, classes, img_size, limit, out_dir, imgs_per_class):
        self.path = path
        self.extension = str(extension)
        self.classes = classes
        self.img_size = img_size
        self.limit = limit
        self.datasetInfo = '_' + str(self.img_size) + '_limit_' + str(self.limit)
        self.splitDatasetsDir = 'splitDatasets' + str(img_size)
        self.modelsDir = 'models'
        self.resultsDir = out_dir
        self.imgs_per_class = imgs_per_class

    def random_augment(self, image, img_size):
        def aug1(img):
            img = tf.cast(image, tf.float32)
            img = img/255.0
            return img
        transfms = [aug1(image), tf.image.random_crop(image, size=[img_size,img_size, 3]),
         tf.image.random_brightness(image, max_delta=0.5),tf.image.random_flip_up_down(image),
         tf.image.random_flip_left_right(image), tf.image.random_contrast(image, lower=.2, upper=.8),
         tf.image.random_hue(image, max_delta = .5), tf.image.random_saturation(image, lower=.2, upper=.8)]
        indices = np.arange(0, len(transfms))
        distr = [1/len(transfms) for i in range(0,len(transfms))]
        idx = np.random.choice(indices, p=distr)
        image = transfms[idx]
        return image
    
    def load(self):
        images = []
        labels = []
        idx = 0
        i = 0
        curr_classes = 0
        for class_dir in glob.glob(os.path.join(self.path, "*/")):
            if curr_classes >= len(self.classes):
                break
            curr_classes += 1
            curr_imgs = 0
            for image in glob.glob(os.path.join(class_dir, "*" + self.extension)):
                if curr_imgs < self.imgs_per_class:
                    curr_imgs += 1
                    img = Image.open(image).resize((self.img_size, self.img_size))
                    
                    # removing the 4th dim which is transparency and rescaling to 0-1 range
                    im = np.array(img)[..., :3]
                    im_aug = self.random_augment(im, self.img_size)
                    images.append(im)
                    images.append(im_aug)
                    labels.append(idx)
                    labels.append(idx)
                    i += 1
                else: break
            idx += 1
        return np.array(images), np.array(labels)

    def demo_load(self):
        path1 = os.path.join(self.path, "data")
        path2 = os.path.join(self.path, "data")
        img1 = np.array(Image.open(path1).resize((self.img_size, self.img_size)))[..., :3]
        img2 = np.array(Image.open(path2).resize((self.img_size, self.img_size)))[..., :3]
        return np.array([img1, img2]), np.array([0, 0])

    def save_train_test_split(self, X_train, X_test, y_train, y_test):
        np.save(os.path.join('X_train' + '.npy'), X_train)
        np.save(os.path.join('X_test' + '.npy'), X_test)
        np.save(os.path.join('y_train' + '.npy'), y_train)
        np.save(os.path.join('y_test' + '.npy'), y_test)

    def load_train_test_split(self):
        X_train = np.load(os.path.join(self.splitDatasetsDir, 'X_train_size' + self.datasetInfo + '.npy'))
        X_test = np.load(os.path.join(self.splitDatasetsDir, 'X_test_size' + self.datasetInfo + '.npy'))
        y_train = np.load(os.path.join(self.splitDatasetsDir, 'y_train_size' + self.datasetInfo + '.npy'))
        y_test = np.load(os.path.join(self.splitDatasetsDir, 'y_test_size' + self.datasetInfo + '.npy'))
        return X_train, X_test, y_train, y_test

    def toOneHot(self, yData):
        return to_categorical(yData, num_classes=len(self.classes))

    def save_training_history(self, history):
        np.save(os.path.join('training_history' + '.npy'), history)

    def load_training_history(self):
        return np.load(os.path.join('training_history' + '.npy'), allow_pickle=True).item()

    def save_model(self, networkName, model):
        model_json = model.to_json()
        with open(os.path.join(self.modelsDir, 'model' + networkName + ".json"), "w") as json_file:
            json_file.write(simplejson.dumps(simplejson.loads(model_json), indent=4))
        model.save_weights(os.path.join(self.modelsDir, 'model' + networkName + self.datasetInfo + '.h5'))
        print("Saved model to disk")

    def load_model_weights(self, networkName, model):
        model.load_weights(os.path.join(self.modelsDir, 'best' + networkName + self.datasetInfo + '.hdf5'))

    def save_details(self, stats, networkName, fileName):
        with open(os.path.join(os.path.dirname(os.path.realpath(__file__)), self.resultsDir, 'details' + fileName + ".txt"),
                  "w") as f:
            f.write("recall: " + str(stats.recall) + '\n')
            f.write("precision: " + str(stats.precision) + '\n')
            f.write("F1 score: " + str(stats.f1Score) + '\n')
            f.write("report: " + str(stats.report) + '\n')
            f.write("accuracy: " + str(stats.accuracy) + '\n')
            if fileName[:2] == "RL":
                f.write("RL execution time: " + str(stats.RL_time) + '\n')
                f.write("Misclassifications [NoRL, RL]: " + str(stats.misclassifications))
