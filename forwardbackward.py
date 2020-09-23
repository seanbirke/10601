import sys
import numpy as np

class HMM():
    def __init__(self, test_in, ind_to_word, ind_to_tag, prior_in, emit_in, trans_in, pred_out, metrics_out):
        self.test_in = test_in
        self.word_dict, self.num_words = self.makeDict(ind_to_word)
        self.tag_dict, self.num_tags = self.makeIndDict(ind_to_tag)

        self.prior = self.initProbs(prior_in)
        self.emit = self.initProbs(emit_in)
        self.trans = self.initProbs(trans_in)

        self.pred_out = pred_out
        self.metrics_out = metrics_out

        self.words, self.tags = self.processInput(test_in)  # lists of sentences, tags

    def forwardBackward(self, sent):
        alpha, beta = np.zeros((len(sent), self.num_tags)), np.zeros((len(sent), self.num_tags))
        for i in range(len(sent)):
            for j in range(self.num_tags):
                if i == 0:
                    alpha[i][j] = float(self.prior[j][i]) * float(self.emit[j][self.word_dict[sent[i]]])
                else:
                    sum = 0
                    for k in range(self.num_tags):
                        sum += float(alpha[i-1][k]) * float(self.trans[k][j])
                    alpha[i][j] = sum * float(self.emit[j][self.word_dict[sent[i]]])
            if i == len(sent)-1:
                continue
            alpha[i] /= np.sum(alpha[i])

        final_sum = np.sum(alpha[-1])
        loss = np.log(final_sum)

        for row in range(len(alpha)):
            alpha[row] /= np.sum(alpha[row])

        beta[-1] += 1
        for i in range(len(sent)-2, -1, -1):
            for j in range(self.num_tags):
                for k in range(self.num_tags):
                    beta[i][j] += float(beta[i+1][k]) * float(self.emit[k][self.word_dict[sent[i+1]]]) * float(self.trans[j][k])

        for row in range(len(beta)):
            beta[row] /= np.sum(beta[row])

        final = alpha * beta
        return loss, np.argmax(final, axis=1)

    def predictOutput(self):
        loss = 0
        correct = 0
        total = 0
        prediction = []
        for i in range(len(self.words)):
            mini_loss, tag_inds = self.forwardBackward(self.words[i])
            loss += mini_loss
            tags = []
            for j in range(len(tag_inds)):
                tags.append(self.tag_dict[tag_inds[j]])
            prediction.append(tags)
            for k in range(len(self.tags[i])):
                if self.tags[i][k] == tags[k]:
                    correct += 1
                total += 1
        return prediction, loss, correct/total

    def outputRes(self):
        prediction, loss, error = self.predictOutput()
        print(loss)
        metrics_file = open(self.metrics_out,'w')
        metrics_file.write("Average Log-Likelihood: %f\n" %loss)
        metrics_file.write("Accuracy: %f" %error)
        metrics_file.close()
        pred_file = open(self.pred_out,'w')
        for i in range(len(self.words)):
            for j in range(len(self.words[i])):
                if j == 0:
                    pred_file.write("%s_%s" %(self.words[i][j], prediction[i][j]))
                else:
                    pred_file.write(" %s_%s" % (self.words[i][j], prediction[i][j]))
            pred_file.write("\n")
        pred_file.close()


    def initProbs(self, file_in):
        data = np.loadtxt(file_in)
        if data.ndim == 1:
            data = data[np.newaxis].T
        return data
        # file = open(file_in)
        # lines = file.readlines()
        # file.close()
        # arr_out = []
        # for line in lines:
        #     data = line.split(' ')
        #     row = []
        #     if len(data) > 1:
        #         for val in data:
        #             val = val.strip()
        #             print(val)
        #             row.append(float(val))
        #         arr_out.append(row)
        #     else:
        #         val = data[0].strip()
        #         arr_out.append([float(val)])
        # return arr_out


    def makeDict(self, file_in):
        file = open(file_in)
        lines = file.readlines()
        file.close()
        dict = {}
        i = 0  # 0 or 1??
        for line in lines:
            line = line.strip()  # remove trailing newline char
            dict[line] = i
            i += 1
        return dict, i

    def makeIndDict(self, file_in):
        file = open(file_in)
        lines = file.readlines()
        file.close()
        list = []
        for line in lines:
            line = line.strip()  # remove trailing newline char
            list.append(line)
        return list, len(list)


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
                tag = tag.strip()  # remove trailing newline char
                words.append(word)
                tags.append(tag)
            all_words.append(words)
            all_tags.append(tags)
        return all_words, all_tags  # numpy array of numpy arrays if not same inner lengths





def main():
    test_in = sys.argv[1]
    ind_to_word = sys.argv[2]
    ind_to_tag = sys.argv[3]
    prior_in = sys.argv[4]
    emit_in = sys.argv[5]
    trans_in = sys.argv[6]
    pred_out = sys.argv[7]
    metrics_out = sys.argv[8]

    model = HMM(test_in, ind_to_word, ind_to_tag, prior_in, emit_in, trans_in, pred_out, metrics_out)
    model.outputRes()



if __name__ == '__main__':
    main()