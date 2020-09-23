from environment import MountainCar
import sys
import numpy as np

class RLModel:
    def __init__(self, mode, weight_out, returns_out, episodes, max_itrs, epsilon, gamma, learn_rate):
        self.mode = mode

        self.weight_out = weight_out
        self.returns_out = returns_out

        self.episodes = episodes
        self.max_itrs = max_itrs
        self.epsilon = epsilon
        self.gamma = gamma
        self.learn_rate = learn_rate

        self.car = MountainCar(self.mode)
        self.num_actions, self.num_states = 3, self.getNumStates()

        self.weights = np.zeros((self.num_states, self.num_actions))
        self.bias = 0

        self.done = False
        self.state_dict = {}
        self.q_val = 0


    def getNumStates(self):
        if self.mode == "tile":
            return 2048
        return 2


    def findQ(self, s, w, b):
        sum = 0
        for key in s:
            sum += w[key]*s[key]
        return sum + b

    def findAction(self, q):
        rand_val = np.random.random()
        if rand_val <= 1 - self.epsilon:
            return np.argmax(q)
        return np.random.choice([0, 1, 2])


    def learnModel(self):
        all_r = []
        weights = np.zeros((self.num_states, self.num_actions))
        bias = 0
        for i in range(self.episodes):
            self.done = False
            state = self.car.reset()
            sum_reward = 0
            itr = 0
            while (not self.done) and (itr < self.max_itrs): ######
                q = self.findQ(state, weights, bias)

                action = self.findAction(q)

                state_p, reward, self.done = self.car.step(action)

                q_p = self.findQ(state_p, weights, bias)

                sum_reward += reward


                d_q = np.zeros((self.num_states, self.num_actions))
                for key in state:
                    d_q[int(key)][action] = state[key]

                q_pi = reward + self.gamma * np.max(q_p)


                weights -= self.learn_rate * (q[action] - q_pi) * d_q
                bias -= self.learn_rate * (q[action] - q_pi)

                state = state_p

                itr += 1
                # if self.done:
                #     print("DONEEEE")
                # if itr >= self.max_itrs:
                #     print("ITERRRRR")
            all_r.append(sum_reward)
        self.weights = weights
        self.bias = bias

        print(self.bias)
        print(self.weights)
        print(all_r)

        # for r in all_r:
        #     print(r)
        return all_r



    def outputAll(self):
        rewards = self.learnModel()
        ret_out = open(self.returns_out, 'w')
        for i in range(len(rewards)):
            ret_out.write("%f\n" %rewards[i])
        ret_out.close()

        wei_out = open(self.weight_out, 'w')
        wei_out.write("%f\n" %self.bias)
        for i in range(self.weights.shape[0]):
            for j in range(self.weights.shape[1]):
             wei_out.write("%f\n" %self.weights[i][j])
        wei_out.close()





def main(args):
    mode = args[1]
    weight_out = args[2]
    returns_out = args[3]
    episodes = args[4]
    max_itrs = args[5]
    epsilon = args[6]
    gamma = args[7]
    learn_rate = args[8]

    model = RLModel(mode, weight_out, returns_out, int(episodes), int(max_itrs),
                    float(epsilon), float(gamma), float(learn_rate))
    model.outputAll()




if __name__ == "__main__":
    main(sys.argv)