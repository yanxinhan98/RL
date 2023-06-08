import random
import numpy as np
from collections import deque
from tensorflow.python.keras.layers import Input, Dense
from tensorflow.python.keras.optimizers import rmsprop_v2
from tensorflow.python.keras.optimizers import adam_v2
from tensorflow.python.keras.models import Model, load_model
from PIL import Image, ImageOps
from random import randrange
from scipy.ndimage.interpolation import rotate

def OurModel(input_shape, action_space):
    X_input = Input(input_shape)
    X = Dense(512, input_shape=input_shape, activation="relu", kernel_initializer='he_uniform')(X_input)
    X = Dense(256, activation="relu", kernel_initializer='he_uniform')(X)
    X = Dense(64, activation="relu", kernel_initializer='he_uniform')(X)
    X = Dense(action_space, activation="linear", kernel_initializer='he_uniform')(X)
    model = Model(inputs = X_input, outputs = X, name='dqn_model')
    optimizer = rmsprop_v2.RMSProp(learning_rate=0.00025, rho=0.95, epsilon=0.01)
    model.compile(loss="mse", optimizer=optimizer, metrics=["accuracy"])
    return model

class DeepQNetwork:
    def __init__(self):
        self.memory = deque(maxlen=1000)
        self.actions = dict([(0, self.action_rotate_1), (1, self.action_rotate_2), (2, self.diagonal_translation)])
        self.states = [0, 1]
        self.gamma = 0.4
        self.epsilon = 1.0  # exploration rate
        self.epsilon_min = 0.001
        self.epsilon_decay = 0.999
        self.batch_size = 16
        self.train_start = 16
        self.episodes = len(self.actions)
        self.steps = 16
        self.Qtable = np.zeros((len(self.states), len(self.actions)))
        self.model = OurModel(input_shape=(1,), action_space = len(self.actions))
        self.target_model = OurModel(input_shape=(1,), action_space = len(self.actions))
        self.angle1 = 90
        self.angle2 = 180
        self.angle3 = 45
        self.angle4 = -45

    def action_rotate_1(self, picture):
        return rotate(picture, self.angle1, reshape=False)

    def action_rotate_2(self, picture):
        return rotate(picture, self.angle2, reshape=False)

    def action_rotate_3(self, picture):
        return rotate(picture, self.angle3, reshape=False)

    def action_rotate_4(self, picture):
        return rotate(picture, self.angle4, reshape=False)

    def action_invariant(self, picture):
        return picture

    def diagonal_translation(self, picture):
        img = Image.fromarray(picture.astype('uint8'), 'RGB')
        w = int(img.size[0] * 0.75)
        h = int(img.size[1] * 0.75)
        border = (15, 15, img.size[0] - w - 15, img.size[1] - h - 15)
        img = img.resize((w, h), Image.ANTIALIAS)
        translated = ImageOps.expand(img, border=border, fill='black')
        return np.array(translated)

    def selectAction(self):
        return randrange(len(self.actions))

    def apply_action(self, action, img):
        return self.actions[action](img)

    def get_features_metric(self, features):
        return np.std(features)

    def get_reward(self, m1, m2):
        return np.sign(m2-m1)

    def define_state(self, reward): #if reward is 1, state 0. #if reward is 0 or -1, state 1
        return 0 if reward > 0 else 1
    
    def update_target_m(self):
        self.target_model.set_weights(self.model.get_weights())

    def remember(self, state, action, reward, next_state):
        self.memory.append((state, action, reward, next_state))
        if len(self.memory) > self.train_start:
            if self.epsilon > self.epsilon_min:
                self.epsilon *= self.epsilon_decay

    def act(self, state):
        model_pred = self.model.predict(state)[0]
        pred_a = np.argmax(model_pred)
        random_a = self.selectAction()
        a = [pred_a, random_a]
        return np.random.choice(a, p=[1-self.epsilon, self.epsilon])

    def replay(self):
        if len(self.memory) < self.train_start:
            return
        minibatch = random.sample(self.memory, min(len(self.memory), self.batch_size))
        state = np.zeros((self.batch_size, 1))
        next_state = np.zeros((self.batch_size, 1))
        action, reward = [], []

        for i in range(self.batch_size):
            state[i] = minibatch[i][0]
            action.append(minibatch[i][1])
            reward.append(minibatch[i][2])
            next_state[i] = minibatch[i][3]

        target = self.model.predict(state)
        target_next = self.target_model.predict(next_state)
        for i in range(self.batch_size):
            target_val = reward[i] + self.gamma * np.max(target_next[i])
            target[i][action[i]] = target_val                
        self.model.fit(state, target, batch_size=self.batch_size, verbose=0)

    def random_state(self):
        distr = [1/len(self.states) for i in range(0, len(self.states))]
        return np.random.choice(self.states, p=distr)
    
    def train(self, cnn, img, statscontroller):
        img_features = cnn.get_output_base_model(img)
        m1 = self.get_features_metric(img_features)
        for e in range(self.episodes):
            state = np.array([self.random_state()])
            for s in range(0, self.steps):
                action = self.act(state)
                statscontroller.updateAllActionStats(action)
                modified_img = self.apply_action(action, img)
                modified_img_features = cnn.get_output_base_model(modified_img)
                m2 = self.get_features_metric(modified_img_features)
                reward = self.get_reward(m1, m2)
                next_state = np.array([self.define_state(reward)])
                self.remember(state, action, reward, next_state)
                state = next_state
            self.update_target_m()
            self.replay()
        self.updateQtable()
        
    def updateQtable(self):
        for s in self.states:
            self.Qtable[s] = self.model.predict(np.array([s]))[0]

    def choose_optimal_action(self):
        return np.where(self.Qtable == np.amax(self.Qtable))[1][0]

