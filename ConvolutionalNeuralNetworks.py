import os

from keras import callbacks, optimizers
from keras.models import Model
from keras.applications import ResNet50, InceptionV3, MobileNetV2, NASNetMobile
from keras.layers import GlobalAveragePooling2D, Dense, Flatten, Dropout
from tensorflow.python.keras.callbacks import ModelCheckpoint

from ImageHelper import NumpyImg2Tensor

import ssl

ssl._create_default_https_context = ssl._create_unverified_context


class ConvolutionalNeuralNetworks:
    def __init__(self, networkName, datasetInfo, num_classes, lrn_rate):
        self.datasetInfo = datasetInfo
        self.networkName = networkName
        self.model = None
        self.last_base_layer_idx = 0
        self.num_classes = num_classes
        self.lrn_rate = lrn_rate
        # self.callbacks = [
        #     # callbacks.EarlyStopping(monitor='val_accuracy', patience=5),
        #     callbacks.EarlyStopping(patience=5, verbose=1),
        #     ModelCheckpoint(os.path.join('models', 'best' + self.networkName + self.datasetInfo + '.hdf5'),
        #                     monitor='val_accuracy')]

    def __get_layer_idx_by_name(self, layerName):
        index = None
        for idx, layer in enumerate(self.model.layers):
            if layer.name == layerName:
                index = idx
                break
        return index

    def create_model_architecture(self, shape=(64, 64, 3)):
        if self.networkName == "ResNet":
            self.model = ResNet50(include_top=False, weights="imagenet", input_shape=shape)
            self.last_base_layer_idx = self.__get_layer_idx_by_name(self.model.layers[-1].name)
            for layer in self.model.layers[:-18]:
                layer.trainable = False
            gavp = GlobalAveragePooling2D()(self.model.output)
            d1 = Dense(1024, 'relu')(gavp)
            d2 = Dense(1024, 'relu')(d1)
            d3 = Dense(1024, 'relu')(d2)
            d4 = Dense(512, 'relu')(d3)
            d5 = Dense(self.num_classes, 'softmax')(d4) #edited
            self.model = Model(inputs=self.model.input, outputs=d5)

        if self.networkName == "Inception":
            self.model = InceptionV3(include_top=False, weights="imagenet", input_shape=shape)
            self.last_base_layer_idx = self.__get_layer_idx_by_name(self.model.layers[-1].name)
            for layer in self.model.layers[:-4]:
                layer.trainable = False
            f = Flatten()(self.model.output)
            d1 = Dense(1024, 'relu')(f)
            do1 = Dropout(0.20)(d1)
            d2 = Dense(self.num_classes, 'softmax')(do1)
            self.model = Model(inputs=self.model.input, outputs=d2)

        if self.networkName == 'MobileNet':
            self.model = MobileNetV2(include_top=False, input_shape=shape)
            self.last_base_layer_idx = self.__get_layer_idx_by_name(self.model.layers[-1].name)
            for layer in self.model.layers[:-4]:
                layer.trainable = False
            gavp = GlobalAveragePooling2D()(self.model.output)
            dense = Dense(self.num_classes, 'softmax')(gavp)
            self.model = Model(inputs=self.model.input, outputs=dense)
        #self.model.compile(loss='categorical_crossentropy', optimizer=optimizers.Adam(learning_rate=self.lrn_rate), metrics=['accuracy'])

        if self.networkName == "NASNetMobile":
            self.model = NASNetMobile(include_top=False, input_shape = shape)
            self.last_base_layer_idx = self.__get_layer_idx_by_name(self.model.layers[-1].name)
            for layer in self.model.layers[:-4]:
                layer.trainable = False
            gavp = GlobalAveragePooling2D()(self.model.output)
            dense = Dense(self.num_classes, 'softmax')(gavp)
            self.model = Model(inputs=self.model.input, outputs=dense)
            
        self.model.compile(loss='categorical_crossentropy', optimizer=optimizers.Adam(learning_rate=self.lrn_rate), metrics=['accuracy'])

    def get_output_base_model(self, img):
        feature_extractor = Model(inputs=self.model.inputs,
                                  outputs=[layer.output for layer in self.model.layers])
        features = feature_extractor(NumpyImg2Tensor(img))
        return features[self.last_base_layer_idx]
