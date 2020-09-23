import sys
import numpy as np

class HMM():
    def __init__(self, train_in, ind_to_word, ind_to_tag, prior_out, emit_out, trans_out):
        self.train_in = train_in
        self.word_dict, self.num_words = self.makeDict(ind_to_word)
        self.tag_dict, self.num_tags = self.makeDict(ind_to_tag)
        self.prior_out = prior_out
        self.emit_out = emit_out
        self.trans_out = trans_out

        self.words, self.tags = self.processInput(train_in) # lists of sentences, tags


    def outputAll(self):
        prior = self.getPrior()
        trans = self.getTrans()
        emit = self.getEmit()
        self.output(prior, self.prior_out)
        self.output(trans, self.trans_out)
        self.output(emit, self.emit_out)

    def output(self, array, file_out):
        file = open(file_out, 'w')
        if array.ndim == 1:
            for i in range(len(array)):
                file.write("%f " % array[i])
        else:
            for i in range(array.shape[0]):
                for j in range(array.shape[1]):
                    file.write("%f " %array[i][j])
                file.write("\n")
        file.close()

    def getEmit(self):
        emit = np.zeros((self.num_tags, self.num_words))
        for i in range(len(self.words)):
            for j in range(len(self.words[i])):
                tag_ind = self.tag_dict[self.tags[i][j]]
                word_ind = self.word_dict[self.words[i][j]]
                emit[tag_ind][word_ind] += 1
        emit += 1
        for i in range(self.num_tags):
            emit[i] /= np.sum(emit[i])
        return emit


    def getTrans(self):
        trans = np.zeros((self.num_tags, self.num_tags))
        for sent in self.tags:
            for i in range(len(sent)-1):
                first_ind = self.tag_dict[sent[i]]
                next_ind = self.tag_dict[sent[i+1]]
                trans[first_ind][next_ind] += 1
        trans += 1
        for i in range(self.num_tags):
            trans[i] /= np.sum(trans[i])
        return trans # 2d np.array of tags x tags across all sentences


    def getPrior(self):
        pi = []
        first = [i[0] for i in self.tags]
        sum = 0
        for tag in self.tag_dict:
            val = first.count(tag) + 1
            sum += val
            pi.append([val])
        pi = np.array(pi) / sum
        return pi # vertical np.array of priors


    def makeDict(self, file_in):
        file = open(file_in)
        lines = file.readlines()
        file.close()
        dict = {}
        i = 0           # 0 or 1??
        for line in lines:
            line = line.strip() # remove trailing newline char
            dict[line] = i
            i += 1
        return dict, i

    def processInput(self, file_in):
        file = open(file_in)
        lines = file.readlines()
        file.close()
        all_words = []
        all_tags = []
        for line in lines:
            words = []
            tags = []
            pairs = line.split(' ')
            for pair in pairs:
                word = pair.split('_')[0]
                tag = pair.split('_')[1]
                tag = tag.strip() # remove trailing newline char
                words.append(word)
                tags.append(tag)
            all_words.append(words)
            all_tags.append(tags)
        return all_words, all_tags # numpy array of numpy arrays if not same inner lengths



def main():
    train_in = sys.argv[1]
    ind_to_word = sys.argv[2]
    ind_to_tag = sys.argv[3]
    prior_out = sys.argv[4]
    emit_out = sys.argv[5]
    trans_out = sys.argv[6]

    model = HMM(train_in, ind_to_word, ind_to_tag, prior_out, emit_out, trans_out)
    model.outputAll()



if __name__ == '__main__':
    main()