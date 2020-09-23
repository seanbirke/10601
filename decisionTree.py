import sys
import numpy as np

class Node():
    def __init__(self, train_in, attributes, prev_impurity, depth, max_depth, labels, name):
        self.data = np.array(train_in)
        self.prev_impurity = prev_impurity # previous impurity, root impurity if root
        self.attributes = attributes # only unused attributes, still valid to split on
        self.splittingAttribute = None # index in self.attributes of the split, NOT index of overall attribute list
        self.max_depth = max_depth
        self.depth = depth
        self.labels = labels # the two label choices
        self.all_labels = list(self.data[1:,-1])
        self.label = self.findMajority()
        self.left = None
        self.right = None
        self.counts = []
        self.name = name
        self.left_info = None
        self.right_info = None



    def findSplit(self):
        giniGains = []
        for i in range(len(self.attributes)):
            giniGains.append(self.findGain(i)) # attribite given as index
        high_gain = max(giniGains)
        split = giniGains.index(high_gain)
        self.splittingAttribute = split
        self.counts = self.counts[split]


    def findMajority(self):
        first = self.all_labels.count(self.labels[0])
        second = self.all_labels.count(self.labels[1])
        if first > second:
            return self.labels[0]
        elif second > first:
            return self.labels[1]
        else:   # evenly split, so last in lexographical order wins
            if self.labels[0] > self.labels[1]:
                return self.labels[0]
            else:
                return self.labels[1]


    def train(self):
        if len(self.attributes) == 0:
            return
        if self.depth < self.max_depth:
            self.findSplit()
            if self.splittingAttribute == None:
                return
            attribute = self.attributes[self.splittingAttribute]
            split_main_ind = np.where(self.data[0,:] == attribute)

            split_values = np.unique(self.data[1:,split_main_ind])
            if len(split_values) == 1:
                return
            data_0 = self.partitionData(self.splittingAttribute, split_values[0])
            data_1 = self.partitionData(self.splittingAttribute, split_values[1])
            giniImp = self.findImpurity(self.splittingAttribute)

            new_attributes = np.delete(self.attributes, self.splittingAttribute)
            name_data_0 = self.printHelper(data_0)
            name_data_1 = self.printHelper(data_1)
            self.left = Node(data_0, new_attributes, giniImp, self.depth + 1, self.max_depth,
                             self.labels, "%s = %s: "%(attribute,split_values[0]) + name_data_0)
            self.left_info = [attribute,split_values[0]]
            self.right = Node(data_1, new_attributes, giniImp, self.depth + 1, self.max_depth,
                             self.labels, "%s = %s: "%(attribute,split_values[1]) + name_data_1)
            self.right_info = [attribute,split_values[1]]
            if self.left != None:
                self.left.train()
            if self.right != None:
                self.right.train()
            return
        return

    def printHelper(self, data):
        label_0 = self.labels[0]
        label_1 = self.labels[1]
        count_0 = list(data[1:,-1]).count(label_0)
        count_1 = list(data[1:,-1]).count(label_1)
        string = "[%d %s /%d %s]"%(count_0, label_0, count_1, label_1)
        return string

    def partitionData(self, attribute, value): # returns new rows, also removes col of used attribute, keeps header
        new_dataset = []
        for i in range(len(self.data)):
            if i == 0:
                new_dataset.append(self.data[i])
            elif self.data[i][attribute] == value:
                new_dataset.append(self.data[i])
        remove_att = np.delete(new_dataset, attribute, 1)
        return remove_att


    def findImpurity(self, attribute): # attribute given as an index
        all_values = self.data[1:,attribute] # indexed values of the attribute
        values = list(set(all_values)) # list of the two values
        counts = [[0,0],[0,0]] # [(Y,N),(Y,N)]
        sizes = []
        impurities = []
        all_labels = self.data[1:,-1]

        for i in range(len(all_values)):
            if all_values[i] == values[0]:
                if all_labels[i] == self.labels[0]:
                    counts[0][0] += 1
                else: counts[0][1] += 1
            else:
                if all_labels[i] == self.labels[0]:
                    counts[1][0] += 1
                else: counts[1][1] += 1

        yv1, nv1 = counts[0][0], counts[0][1]
        yv2, nv2 = counts[1][0], counts[1][1]
        sizes.append(yv1 + nv1)
        sizes.append(yv2 + nv2)
        total = sizes[0] + sizes[1]
        self.counts.append(counts)
        if sizes[0] != 0:
            impurities.append( (yv1/sizes[0])*(1-(yv1/sizes[0])) + (nv1/sizes[0])*(1-(nv1/sizes[0])) )
        else: impurities.append(0)
        if sizes[1] != 0:
            impurities.append( (yv2/sizes[1])*(1-(yv2/sizes[1])) + (nv2/sizes[1])*(1-(nv2/sizes[1])) )
        else: impurities.append(0)
        giniImp = (sizes[0]/total)*impurities[0] + (sizes[1]/total)*impurities[1]
        return giniImp


    def findGain(self, attribute):
        impurity = self.findImpurity(attribute)
        giniGain = self.prev_impurity - impurity
        if impurity == 0:
            return 0
        return giniGain

    def prettyPrint(self):
        attribute = self.attributes[self.splittingAttribute]
        print("| " * (self.depth) + self.name)
        if self.left != None:
            self.left.prettyPrint()
        if self.right != None:
            self.right.prettyPrint()
        return



class Tree():
    def __init__(self, maxDepth, trainIn):
        self.maxDepth = int(maxDepth)
        self.data = trainIn
        self.attributes = (self.data[0])[:-1]
        self.all_labels = list(self.data[1:,-1])
        self.root_impurity = self.getRootImpurity()
        self.labels = list(set(self.data[1:, -1]))
        self.name = self.printHelper()
        self.root = Node(trainIn, self.attributes, self.root_impurity, 0, self.maxDepth, self.labels, self.name)
        self.train_predict = []
        self.test_predict = []
        self.train_error = 0
        self.test_error = 0

    def predict(self, data, filename, predictions):
        file = open(filename, 'w')
        for i in range(len(data)):
            if i == 0: continue
            prediction = self.predictHelper(data, i, self.root)

            predictions.append(prediction)
            file.write("%s\n" % prediction)
        file.close()

    def predictHelper(self, data, row, node):
        if node.left == None:
            return node.label
        else:
            attribute = node.left_info[0]
            split_main_ind = np.where(data[0, :] == attribute)
            value = data[row][split_main_ind]
            if value == node.left_info[1]:
                return self.predictHelper(data, row, node.left)
            else:
                return self.predictHelper(data, row, node.right)


    def getRootImpurity(self):
        labels = list(set(self.all_labels))
        counts = []
        counts.append(self.all_labels.count(labels[0]))
        counts.append(self.all_labels.count(labels[1]))
        total = counts[0] + counts[1]
        giniImp = (counts[0]/total)*(1-(counts[0]/total)) + (counts[1]/total)*(1-(counts[1]/total))
        return giniImp


    def printHelper(self):
        label_0 = self.labels[0]
        label_1 = self.labels[1]
        count_0 = self.all_labels.count(label_0)
        count_1 = self.all_labels.count(label_1)
        string = "[%d %s /%d %s]"%(count_0, label_0, count_1, label_1)
        return string

    def findError(self, data, predictions):
        total = 0
        wrong = 0
        for i in range(len(predictions)):
            if predictions[i] != data[i+1][-1]:
                wrong += 1
            total += 1
        error = float(wrong)/total
        return error

    def outputErrors(self, filename):
        file = open(filename, 'w')
        file.write("error(train): %f\n" % self.train_error)
        file.write("error(test): %f\n" % self.test_error)
        print("error(train): %f\n" % self.train_error)
        print("error(test): %f\n" % self.test_error)
        file.close()




def main():
    train_in = sys.argv[1]
    test_in = sys.argv[2]
    max_depth = sys.argv[3]
    train_out = sys.argv[4]
    test_out = sys.argv[5]
    metrics_out = sys.argv[6]

    trainIn = np.loadtxt(train_in, dtype=str, delimiter='\t')
    testIn = np.loadtxt(test_in, dtype=str, delimiter='\t')

    decisionTree = Tree(max_depth, trainIn)
    decisionTree.root.train()

    print("\n")
    decisionTree.root.prettyPrint()
    print("\n")

    decisionTree.predict(trainIn, "%s" % train_out, decisionTree.train_predict)
    decisionTree.predict(testIn, "%s" % test_out, decisionTree.test_predict)

    decisionTree.train_error = decisionTree.findError(trainIn, decisionTree.train_predict)
    decisionTree.test_error = decisionTree.findError(testIn, decisionTree.test_predict)
    decisionTree.outputErrors("%s" % metrics_out)



if __name__ == '__main__':
    main()