from ConvolutionalNeuralNetworks import ConvolutionalNeuralNetworks
from DataLoader import DataLoader
from Plotter import *
from ImageHelper import NumpyImg2Tensor, ShowNumpyImg
from QLearningModel import QLearningModel
from MC import MonteCarlo
from DQN import DeepQNetwork
from sklearn.model_selection import train_test_split
import time
from StatisticsController import StatisticsController
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, classification_report, \
    confusion_matrix
import numpy as np

#edit parameter variables
#-----Model parameters------
# set to true if algorithm is launched for the first time
LOAD_DATA = True
TRAIN_NETWORK = True
LIMIT = 10
ACTION_NAMES = ['rotate +90', 'rotate +180', 'diagonal translation']
networkName = "ResNet" #Resnet, Inception, MobileNet, NASNetMobile
epoch = 7
batch_s = 16
learning_rate = .00001
#RL algs: Q_learning, Monte_Carlo, DeepQ_N
Q_learning = False
Monte_Carlo = False
DeepQ_N = True
# ----------Data Load-----------------
t1 = time.time()
#-----data parameters--------
IMG_SIZE = 75 #75
imgs_dir = "pedestrian"
train_dir = "train"
img_ext = ".jpg"
output_dir = networkName + "_" + imgs_dir + "_out"
imgs_per_class = 500
#classes = {"airplane": 0, "automobile": 1, "bird": 2, "cat": 3, "deer": 4, "dog":5, "frog":6, "horse": 7, "ship": 8, "truck": 9} #cifar10
#classes = {"daisy": 0, "dandelion": 1, "rose": 2, "sunflower": 3, "tulip": 4} #flower
#classes = {"cloudy": 0, "desert": 1, "green_area": 2, "water": 3} #remote sensing
classes = {"no pedestrian": 0, "pedestrian": 1} #pedestrian detection

#prevent file doesnt exist errors
if not os.path.exists(os.path.join(os.path.dirname(os.path.realpath(__file__)), output_dir)):
    os.mkdir(os.path.join(os.path.dirname(os.path.realpath(__file__)), output_dir))

dl = DataLoader(os.path.join(os.path.dirname(os.path.realpath(__file__)), imgs_dir, train_dir), #modified1
                img_ext,
                classes,
                IMG_SIZE,
                LIMIT, output_dir, imgs_per_class)
if LOAD_DATA:
    images, labels = dl.load()
    print(len(images))
    print(len(labels))
    X_train, X_test, y_train, y_test = train_test_split(images, labels, test_size=0.2, random_state=1)
    print(X_train.shape)
    dl.save_train_test_split(X_train, X_test, y_train, y_test)
else:
    X_train, X_test, y_train, y_test = dl.load_train_test_split()
print("Data Load time: " + str(time.time() - t1))

# ---------CNN training---------------
t2 = time.time()
cnn = ConvolutionalNeuralNetworks(networkName, dl.datasetInfo, len(classes), learning_rate)
cnn.create_model_architecture(X_train[0].shape)
statControllerNoRl = StatisticsController(classes)

if TRAIN_NETWORK:
    print
    statControllerNoRl.trainingHistory = cnn.model.fit(X_train, dl.toOneHot(y_train), batch_size=batch_s, epochs=epoch,
                                                        validation_split=0.2).history #callbacks=cnn.callbacks
    dl.save_training_history(statControllerNoRl.trainingHistory)
    dl.save_model(cnn.networkName, cnn.model)
else:
    dl.load_model_weights(networkName, cnn.model)
    statControllerNoRl.trainingHistory = dl.load_training_history()
print("CNN training time: " + str(time.time() - t2))

# ----------RL execution--------------
print('executing RL:')
def Qlearning():
    t = time.time()
    q = QLearningModel()
    statControllerRl = StatisticsController(classes, len(ACTION_NAMES))
    statnoRL = StatisticsController(classes)
    misclassif_stats = [0, 0] #noRL, RL
    for img, label in zip(X_test, y_test):
        no_lr_probabilities_vector = cnn.model.predict(NumpyImg2Tensor(img))
        predictedLabel = np.argmax(no_lr_probabilities_vector)
        statnoRL.predictedLabels.append(predictedLabel)    
        if predictedLabel != label:
            misclassif_stats[0] += 1
            q.perform_iterative_Q_learning(cnn, img, statControllerRl)
            optimal_action = q.choose_optimal_action()
            statControllerRl.updateOptimalActionsStats(optimal_action)
            corrected_img = q.apply_action(optimal_action, img)
            probabilities_vector = cnn.model.predict(NumpyImg2Tensor(corrected_img))
            predictedlabel_rl = np.argmax(probabilities_vector)
            statControllerRl.predictedLabels.append(predictedlabel_rl)
            if predictedlabel_rl != label:
                misclassif_stats[1] += 1
        else:
            statControllerRl.predictedLabels.append(predictedLabel)
    RL_time = time.time() - t
    print("RL execution time (Q-learning): " + str(RL_time))
    statControllerRl.updateRL_time(RL_time)
    statControllerRl.updateMisclassif(misclassif_stats)
    return statnoRL, statControllerRl

def MClearning():
    t = time.time()
    mc = MonteCarlo()
    statcontrRL = StatisticsController(classes, len(ACTION_NAMES))
    statcontrNoRL = StatisticsController(classes)
    misclassif_stats = [0, 0] #noRL, RL
    for img, label in zip(X_test, y_test):
        no_lr_probabilities_vector = cnn.model.predict(NumpyImg2Tensor(img))
        predictedLabel = np.argmax(no_lr_probabilities_vector)
        statcontrNoRL.predictedLabels.append(predictedLabel)     
        if predictedLabel != label:
            misclassif_stats[0] += 1
            mc.iterative_MC_learning(cnn, img, statcontrRL)
            optimal_action = mc.choose_optimal_action()
            statcontrRL.updateOptimalActionsStats(optimal_action)
            corrected_img = mc.apply_action(optimal_action, img)
            probabilities_vector = cnn.model.predict(NumpyImg2Tensor(corrected_img))
            predictedlabel_rl = np.argmax(probabilities_vector)
            statcontrRL.predictedLabels.append(predictedlabel_rl)
            if predictedlabel_rl != label:
                misclassif_stats[1] += 1
        else:
            statcontrRL.predictedLabels.append(predictedLabel)
    RL_time = time.time() - t
    print("Execution time (MC): " + str(RL_time))
    statcontrRL.updateMisclassif(misclassif_stats)
    statcontrRL.updateRL_time(RL_time)
    return statcontrNoRL, statcontrRL

def DQNlearning():
    t = time.time()
    dqn = DeepQNetwork()
    statcontrRL = StatisticsController(classes, len(ACTION_NAMES))
    statcontrNoRL = StatisticsController(classes)
    misclassif_stats = [0, 0] #noRL, RL
    for img, label in zip(X_test, y_test):
        no_lr_probabilities_vector = cnn.model.predict(NumpyImg2Tensor(img))
        predictedLabel = np.argmax(no_lr_probabilities_vector)
        statcontrNoRL.predictedLabels.append(predictedLabel)     
        if predictedLabel != label:
            misclassif_stats[0] += 1
            dqn.train(cnn, img, statcontrRL)
            optimal_action = dqn.choose_optimal_action()
            statcontrRL.updateOptimalActionsStats(optimal_action)
            corrected_img = dqn.apply_action(optimal_action, img)
            probabilities_vector = cnn.model.predict(NumpyImg2Tensor(corrected_img))
            predictedlabel_rl = np.argmax(probabilities_vector)
            statcontrRL.predictedLabels.append(predictedlabel_rl)
            if predictedlabel_rl != label:
                misclassif_stats[1] += 1
        else:
            statcontrRL.predictedLabels.append(predictedLabel)
    RL_time = time.time() - t
    print("Execution time (DQN): " + str(RL_time))
    statcontrRL.updateMisclassif(misclassif_stats)
    statcontrRL.updateRL_time(RL_time)
    return statcontrNoRL, statcontrRL

if Q_learning:
    statno, statrl = Qlearning()
    plot_actions_stats(dl, networkName, ACTION_NAMES, statrl.allActionsStats, "allActionsQ")
    plot_actions_stats(dl, networkName, ACTION_NAMES, statrl.optimalActionsStats, "optimalActionsQ")
    conf_matrix_no_RL = confusion_matrix(y_test, statno.predictedLabels)
    conf_matrix_RL = confusion_matrix(y_test, statrl.predictedLabels)
    plot_conf_matrix(dl, networkName, conf_matrix_no_RL, classes, "NoRLQ")
    plot_conf_matrix(dl, networkName, conf_matrix_RL, classes, "RLQ")
    plot_misclassif(dl, statrl.misclassifications, "Q learning")
    statrl.f1Score = f1_score(y_test, statrl.predictedLabels, average="macro")
    statrl.precision = precision_score(y_test, statrl.predictedLabels, average="macro")
    statrl.recall = recall_score(y_test, statrl.predictedLabels, average="macro")
    statrl.report = classification_report(y_test, statrl.predictedLabels)
    statrl.accuracy = accuracy_score(y_test, statrl.predictedLabels)
    dl.save_details(statrl, networkName, "RLQ")

if Monte_Carlo:
    statno, statrl = MClearning()
    plot_actions_stats(dl, networkName, ACTION_NAMES, statrl.allActionsStats, "allActionsMC")
    plot_actions_stats(dl, networkName, ACTION_NAMES, statrl.optimalActionsStats, "optimalActionsMC")
    conf_matrix_no_RL = confusion_matrix(y_test, statno.predictedLabels)
    conf_matrix_RL = confusion_matrix(y_test, statrl.predictedLabels)
    plot_conf_matrix(dl, networkName, conf_matrix_no_RL, classes, "NoRLMC")
    plot_conf_matrix(dl, networkName, conf_matrix_RL, classes, "RLMC")
    plot_misclassif(dl, statrl.misclassifications, "Monte Carlo")
    statrl.f1Score = f1_score(y_test, statrl.predictedLabels, average="macro")
    statrl.precision = precision_score(y_test, statrl.predictedLabels, average="macro")
    statrl.recall = recall_score(y_test, statrl.predictedLabels, average="macro")
    statrl.report = classification_report(y_test, statrl.predictedLabels)
    statrl.accuracy = accuracy_score(y_test, statrl.predictedLabels)
    dl.save_details(statrl, networkName, "RLMC")

if DeepQ_N:
    statno, statrl = DQNlearning()
    plot_actions_stats(dl, networkName, ACTION_NAMES, statrl.allActionsStats, "allActionsDQN")
    plot_actions_stats(dl, networkName, ACTION_NAMES, statrl.optimalActionsStats, "optimalActionsDQN")
    conf_matrix_no_RL = confusion_matrix(y_test, statno.predictedLabels)
    conf_matrix_RL = confusion_matrix(y_test, statrl.predictedLabels)
    plot_conf_matrix(dl, networkName, conf_matrix_no_RL, classes, "NoRLDQN")
    plot_conf_matrix(dl, networkName, conf_matrix_RL, classes, "RLDQN")
    plot_misclassif(dl, statrl.misclassifications, "Deep Q Network")
    statrl.f1Score = f1_score(y_test, statrl.predictedLabels, average="macro")
    statrl.precision = precision_score(y_test, statrl.predictedLabels, average="macro")
    statrl.recall = recall_score(y_test, statrl.predictedLabels, average="macro")
    statrl.report = classification_report(y_test, statrl.predictedLabels)
    statrl.accuracy = accuracy_score(y_test, statrl.predictedLabels)
    dl.save_details(statrl, networkName, "RLDQN")

plot_history(dl, networkName, statControllerNoRl.trainingHistory)
statno.f1Score = f1_score(y_test, statno.predictedLabels, average="macro")
statno.precision = precision_score(y_test, statno.predictedLabels, average="macro")
statno.recall = recall_score(y_test, statno.predictedLabels, average="macro")
statno.report = classification_report(y_test, statno.predictedLabels)
statno.accuracy = accuracy_score(y_test, statno.predictedLabels)

# print_classification_details(statControllerNoRl)
# print_classification_details(statControllerRl)
# print("Misclassifications [No RL, RL]:", misclassif_stats)
dl.save_details(statno, networkName, "NoRL")


