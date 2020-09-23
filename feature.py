import sys
import numpy as np


class FeatureModel():
    def __init__(self, train_in, val_in, test_in, dict_in, train_out, val_out, test_out, feature_flag):
        self.train_in = train_in
        self.val_in = val_in
        self.test_in = test_in
        self.train_out = train_out
        self.val_out = val_out
        self.test_out = test_out

        self.dict_in = dict_in
        self.dict = self.makeDict()
        self.flag = int(feature_flag)
        self.threshold = 4


    def extractFeatures(self, file_in, file_out):
        file = open(file_in)
        lines = file.readlines()
        file.close()
        features = []
        labels = []
        for line in lines:
            label, review = line.split("\t")
            labels.append(label)
            words = review.split()
            feature = dict()
            for word in words:
                if word in self.dict.keys():
                    if self.dict[word] not in feature:
                        feature[self.dict[word]] = 1
                    elif self.dict[word] in feature:
                        feature[self.dict[word]] += 1
            features.append(feature)
        self.writeFile(labels, features, file_out)


    def makeDict(self):
        dic = dict()
        file = open(self.dict_in)
        lines = file.readlines()
        file.close()
        for line in lines:
            key, val = line.split()
            dic[key] = val
        return dic


    def writeFile(self, labels, features, file_out):
        file = open(file_out, 'w')
        for i in range(len(labels)):
            output = ""
            output += "%s" % (labels[i])
            curr_dict = features[i]
            for word in curr_dict.keys():
                if self.flag == 1:
                    output += "\t%s:1" % (word)
                elif self.flag == 2:
                    if curr_dict[word] < self.threshold:
                        output += "\t%s:1" % (word)
            output += "\n"
            file.write(output)
        file.close()



    def runAll(self):
        self.extractFeatures(self.train_in, self.train_out)
        self.extractFeatures(self.val_in, self.val_out)
        self.extractFeatures(self.test_in, self.test_out)



def main():
    train_in = sys.argv[1]
    val_in = sys.argv[2]
    test_in = sys.argv[3]
    dict_in = sys.argv[4]
    form_train_out = sys.argv[5]
    form_val_out = sys.argv[6]
    form_test_out = sys.argv[7]
    feature_flag = sys.argv[8]

    model = FeatureModel(train_in, val_in, test_in, dict_in, form_train_out, form_val_out, form_test_out, feature_flag)
    model.runAll()



if __name__ == '__main__':
    main()