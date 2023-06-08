import numpy as np
from PIL import Image, ImageOps
from random import randrange
from scipy.ndimage.interpolation import rotate

class MonteCarlo:
    def __init__(self):
        self.alpha = 0.4
        self.gamma = 0.3
        self.angle1 = 90
        self.angle2 = 180
        self.angle3 = 45
        self.angle4 = -45
        self.actions = dict([(0, self.action_rotate_1), (1, self.action_rotate_2), (2, self.diagonal_translation)])
        self.states = [0, 1]
        self.episodes = len(self.actions) * 20
        self.Qtable = np.zeros((len(self.states), len(self.actions)))
        self.ep_steps = 10

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

    def action_policy(self, s):
        s_all_actions = self.Qtable[s, :]
        return np.argmax(s_all_actions)

    def iterative_MC_learning(self, cnn, img, statscontroller):
        Returns = np.zeros((len(self.states), len(self.actions)))
        img_features = cnn.get_output_base_model(img)
        m1 = self.get_features_metric(img_features)
        num_of_qa_visits = np.zeros((len(self.states), len(self.actions)))
        for i in range(0, self.episodes):
            distr = [1/len(self.states) for i in range(0, len(self.states))]
            s = np.random.choice(self.states, p=distr)
            G = 0
            seen = [[False for i in range(0, len(self.actions))] for i in range(0, len(self.states))]
            for j in range(0, self.ep_steps):
                a = self.selectAction()
                statscontroller.updateAllActionStats(a)
                modified_img = self.apply_action(a, img)
                modified_img_features = cnn.get_output_base_model(modified_img)
                m2 = self.get_features_metric(modified_img_features)
                R = self.get_reward(m1, m2)
                G = self.gamma*G + R
                if not seen[s][a]:
                    seen[s][a] = True
                    num_of_qa_visits[s][a] += 1
                    Returns[s][a] += G
                    self.Qtable[s][a] = Returns[s][a]/num_of_qa_visits[s][a]
                s = self.define_state(R)

    def choose_optimal_action(self):
        return np.where(self.Qtable == np.amax(self.Qtable))[1][0]