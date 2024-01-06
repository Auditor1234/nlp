import nltk
import math
import os


# start of sentence, end of sentence, unknow word
SOS = '<s> '
EOS = '</s>'
UNK = '<UNK>'

def load_data(data_path):
    """
    load data file
    """
    train_path = os.path.join(data_path, 'train.txt')
    test_path = os.path.join(data_path, 'test.txt')
    a = os.path.abspath(train_path)
    print(a)

    with open(train_path, 'r') as f:
        train = [line.strip() for line in f.readlines()]

    with open(test_path, 'r') as f:
        test = [line.strip() for line in f.readlines()]
    return train, test

def preprocess(sentences, n):
    """
    pad sentences with sos and eos
    """
    sos = SOS * max(1, n - 1)
    sentences = ['{}{} {}'.format(sos, sentence, EOS) for sentence in sentences]
    tokens = ' '.join(sentences).split(' ')
    vocab = nltk.FreqDist(tokens)
    tokens = [token if vocab[token] > 1 else UNK for token in tokens]
    return tokens

class LanguageModel():
    """
    n-gram model
    """
    def __init__(self, train_data, n) -> None:
        self.n = n
        self.tokens = preprocess(train_data, n)
        self.vocab = nltk.FreqDist(self.tokens)
        self.model = self._create_model()
    
    def _create_model(self):
        """
        create n-gram model
        """
        num_tokens = len(self.tokens)
        if self.n == 1:
            return { (unigram,): count / num_tokens for unigram, count in self.vocab.items() }
        else:
            n_grams = nltk.ngrams(self.tokens, self.n)
            n_vocab = nltk.FreqDist(n_grams)

            m_grams = nltk.ngrams(self.tokens, self.n - 1)
            m_vocab = nltk.FreqDist(m_grams)

            return { n_gram: count / m_vocab[n_gram[:-1]] for n_gram, count in n_vocab.items() }
    
    def _best_candidate(self, prev, i, without=[]):
        """
        generate next token
        """
        blacklist = [UNK] + without
        candidates = [(n_gram[-1], prob) for n_gram, prob in self.model.items() if n_gram[:-1] == prev]
        candidates = filter(lambda candidate: candidate[0] not in blacklist, candidates)
        candidates = sorted(candidates, key=lambda candidate: candidate[1], reverse=True)
        if len(candidates) == 0:
            return (EOS, 1)
        else:
            return candidates[0 if prev != () and prev[-1] != '<s>' else i]

    def generate_sentences(self, num, min_len=12, max_len=24):
        """
        generate sentence
        """
        for i in range(num):
            sent, prob = ['<s>'] * max(1, self.n - 1), 1
            while sent[-1] != '</s>':
                prev = () if self.n == 1 else tuple(sent[-(self.n - 1):])
                blacklist = sent + (['</s>'] if len(sent) < min_len else [])
                next_token, next_prob = self._best_candidate(prev, i, without=blacklist)
                sent.append(next_token)
                prob *= next_prob

                if len(sent) >= max_len:
                    sent.append('</s>')
            
            yield ' '.join(sent), 0 if prob == 0 else -1 / math.log(prob)
    
    def sentence_prob(self, sentence):
        """
        claculate sentence probability
        """
        sentence = '{}{} {}'.format(SOS * (self.n - 1),  sentence, EOS).split(' ')
        sen_len = len(sentence)
        prob = 1
        for i in range(self.n - 1, sen_len):
            key = tuple(sentence[i - self.n + 1 : i + 1])
            if key not in self.model.keys():
                return 0
            prob *= self.model[key]
        
        return 0 if prob == 0 else -1 / math.log(prob)
        

if __name__ == '__main__':
    # set n and data directory
    n = 2
    data_path = 'data'
    train, test = load_data(data_path)
    print('Loadint {}-gram model...'.format(n))
    lm = LanguageModel(train, n)

    # generate 10 sentences
    for sentence, prob in lm.generate_sentences(10):
        print('{} ({:.5f})'.format(sentence, prob))

    # calculate sentence probability, get negative log probability
    input_sentence = 'we are being accused of not implementing this agreement'
    prob = lm.sentence_prob(input_sentence)
    print('input sentence probability = %.5f' % prob)