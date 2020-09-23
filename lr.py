import sys
import numpy as np

class LogReg():
    def __init__(self, form_train_in, form_val_in, form_test_in, dict_in, train_out, test_out, metrics_out, num_epoch):
        self.train_in = form_train_in
        self.val_in = form_val_in
        self.test_in = form_test_in
        self.dict_in = dict_in
        #self.dict = self.makeDict()

        self.train_out = train_out
        self.test_out = test_out
        self.metrics_out = metrics_out
        self.epochs = int(num_epoch)

        self.learn_rate = 0.1

        self.train_labels, self.train_features = self.readFile(self.train_in)
        self.val_labels, self.val_features = self.readFile(self.val_in)
        self.test_labels, self.test_features = self.readFile(self.test_in)
        self.weights = self.trainNew(self.train_features, self.train_labels, self.epochs)

    def readFile(self, file_in): # reads file, outputs label, feature lists as ints
        file = open(file_in)
        lines = file.readlines()
        file.close()
        features = []
        labels = []
        for line in lines:
            feature = []
            data = line.split("\t")
            labels.append(int(data[0]))
            feature_list = data[1:-1]
            last = data[-1]
            for item in feature_list:
                feat = item[:-2]
                feature.append(int(feat))
            last_feat = last[:-3] # the last feature value include the newline char
            feature.append(int(last_feat))
            features.append(feature)
        return labels, features


    def sigmoid(self, x):
        #if x < (-700): assert(0)
        return 1 / (1 + np.exp(-x))


    def makeSparseFeat(self, feature): # input is list of features, output is dict of feature keys, 1 as val
        feat = dict()
        for word in feature:
            feat[word] = 1
        return feat

    def listDictDot(self, list, dict):
        sum = 0
        for ind in dict.keys():
            sum += dict[ind] * list[ind]
        return sum

    def calcLoss(self, features, labels, weights):
        loss = 0
        for i in range(len(labels)):
            feature = self.makeSparseFeat(features[i])  # sparse feature
            feature[39176] = 1  # bias
            dot = self.listDictDot(weights, feature)
            loss += (-labels[i] * dot + np.log(1 + np.exp(dot)))
        return loss


    def listDictAdd(self, list, dict):
        for key in dict.keys():
            list[key] += dict[key]
        return list


    def trainNew(self, features, labels, epochs):
        weights = np.zeros(39177) # 39176 features (words) + 1 bias
        for epoch in range(epochs):
            for i in range(len(labels)):
                feature = self.makeSparseFeat(features[i]) # sparse feature
                feature[39176] = 1 # bias

                val = self.listDictDot(weights, feature)
                exp_val = np.exp(val)
                theta_j = dict()
                for j in feature.keys():
                    add = self.learn_rate * feature[j] * ( labels[i] - (exp_val/(1 + exp_val)))
                    theta_j[j] = add
                weights = self.listDictAdd(weights, theta_j)
            #print(self.calcLoss(self.val_features, self.val_labels, weights))
        return weights


    def predictOutput(self, labels, features, weights, file_out): # writes file, returns error rate
        total = 0
        right = 0
        file = open(file_out, 'w')
        for i in range(len(labels)):
            feature = self.makeSparseFeat(features[i])
            feature[39176] = 1
            dot = self.listDictDot(weights, feature)
            val = self.sigmoid(dot)
            if val >= 0.5:
                prediction = 1
            else: prediction = 0
            if prediction == labels[i]:
                right += 1
            total += 1
            file.write("%d\n"%(prediction))
        file.close()
        return 1 - right/total

    def outputAll(self):
        train_error = self.predictOutput(self.train_labels, self.train_features, self.weights, self.train_out)
        test_error = self.predictOutput(self.test_labels, self.test_features, self.weights, self.test_out)
        file = open(self.metrics_out, 'w')
        file.write("error(train): %f\n" %(train_error))
        file.write("error(test): %f\n" %(test_error))
        file.close()


def main():
    form_train_in = sys.argv[1]
    form_val_in = sys.argv[2]
    form_test_in = sys.argv[3]
    dict_in = sys.argv[4]
    train_out = sys.argv[5]
    test_out = sys.argv[6]
    metrics_out = sys.argv[7]
    num_epoch = sys.argv[8]

    lr = LogReg(form_train_in, form_val_in, form_test_in, dict_in, train_out, test_out, metrics_out, num_epoch)
    lr.outputAll()


if __name__ == '__main__':
    main()