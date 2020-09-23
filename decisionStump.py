import sys
import numpy as np


class Node():
    def __init__(self, train_in, test_in, split_ind):
        self.data = np.loadtxt(train_in, dtype=str, delimiter='\t') # train_in
        self.test_data = np.loadtxt(test_in, dtype=str, delimiter='\t')# test_in
        self.split_ind = int(split_ind)
        self.values = [] # (2) values of given attribute, index 0 = a, index 1 = b
        self.votes = [] # (2) values of votes (ex. demo/repub)
        self.a = [] # list of votes from one of the split sides
        self.b = []
        self.a_vote = None
        self.b_vote = None
        self.train_predict = [] # prediction of labels
        self.test_predict = []
        self.train_error = 0 # error as float (wrong/total)
        self.test_error = 0


    def train(self): # maps self.values[0] to self.a_vote, and same for b
        print(self.data[0])
        for i in range(len(self.data)): # split data into 'a' and 'b' groups
            if i == 0: continue
            val = self.data[i][self.split_ind]
            vote = self.data[i][-1]
            if val in self.values:

                if self.values.index(val) == 0:
                    self.a.append(vote)
                    if vote not in self.votes:
                        self.votes.append(vote)
                else:
                    self.b.append(vote)
                    if vote not in self.votes:
                        self.votes.append(vote)
            if len(self.values) == 0:
                self.values.append(val)
                self.a.append(vote)
            else:
                self.values.append(val)
                self.b.append(self.data[i][-1])
        self.a_vote = self.findMajority(self.a)
        if self.a_vote == self.votes[0]: # second vote is just the remaining option, not another majority
            self.b_vote = self.votes[1]
        else:
            self.b_vote = self.votes[0]
        #self.b_vote = self.findMajority(self.b)

    def findMajority(self, data): # returns the majority vote given a dataset
        first = data.count(self.votes[0])
        second = data.count(self.votes[1])
        if first > second:
            return self.votes[0]
        else:
            return self.votes[1] # if votes are equal, defaults to the second choice

    def test(self, data, filename, predictions):
        file = open(filename, 'w')
        for i in range(len(data)):
            if i == 0: continue
            val = data[i][self.split_ind]
            if val == self.values[0]:
                vote = self.a_vote
            else:
                vote = self.b_vote
            predictions.append(vote)
            file.write("%s\n" %vote)
        file.close()

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
        file.close()


def main():
    train_in = sys.argv[1]
    test_in = sys.argv[2]
    split_ind = sys.argv[3]
    train_out = sys.argv[4]
    test_out = sys.argv[5]
    metrics_out = sys.argv[6]

    Stump = Node(train_in, test_in, split_ind)
    Stump.train()
    Stump.test(Stump.data, "%s" % train_out, Stump.train_predict)
    Stump.test(Stump.test_data, "%s" % test_out, Stump.test_predict)

    Stump.train_error = Stump.findError(Stump.data, Stump.train_predict)
    Stump.test_error = Stump.findError(Stump.test_data, Stump.test_predict)
    Stump.outputErrors("%s" % metrics_out)


if __name__ == '__main__':
    main()


