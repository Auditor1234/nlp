import numpy as np
import csv
import jieba


def load_data(data_file):
    """
    load csv data
    """
    label, data = [], []

    with open(data_file, encoding='utf-8') as f:
        reader = csv.reader(f)
        for row in reader:
            if 'label' in row[0]:
                continue
            label.append(eval(row[0]))
            data.append(row[1])
    return data, label

def drop_stopwords(sentence, stopwords):
    """
    drop useless characters
    """
    return [l for l in jieba.lcut(str(sentence)) if l not in stopwords]

def load_txt(file):
    """
    load txt files
    """
    with  open(file, encoding='utf-8', errors='ignore') as fp:
        lines = fp.readlines()
        lines = [l.strip() for l in lines]
    return lines

def get_sentence_feature(sentence, vocabulary, stopwords):
    """
    vectorize input sentence
    """
    words = drop_stopwords(sentence, stopwords)
    return [int(words.count(w)) for w in vocabulary]

def train(data, label):
    """
    train the modle
    """
    print('training start...')
    data_len = len(data)
    word_len = len(data[0])

    positive_cls = sum(label) / data_len
    positive_vec = np.ones(word_len)
    negative_vec = np.ones(word_len)

    for i in range(data_len):
        if label[i] == 1:
            positive_vec += data[i]
        else:
            negative_vec += data[i]
    
    positive_prob = np.log(positive_vec / sum(positive_vec))
    negative_prob = np.log(negative_vec / sum(negative_vec))

    print('training done.')

    return positive_prob, negative_prob, positive_cls

def predict(sent_vec, pos_prob, neg_prob, pos_cls):
    """
    predict sentence's category
    """
    p1 = sum(sent_vec * pos_prob) + np.log(pos_cls)
    p2 = sum(sent_vec * neg_prob) + np.log(1- pos_cls)

    if p1 > p2:
        print('positive')
    else:
        print('negative')

def validate(data, label, pos_prob, neg_prob, pos_cls):
    """
    get data's classification
    """
    p1 = np.sum(data * pos_prob, axis=-1) + np.log(pos_cls)
    p2 = np.sum(data * neg_prob, axis=-1) + np.log(1- pos_cls)

    prediction = np.where(p1 > p2, 1, 0)
    label = np.array(label)

    TP = sum((prediction == 1) * (label == 1))
    TN = sum((prediction == 0) * (label == 0))
    FP = sum((prediction == 1) * (label == 0))
    FN = sum((prediction == 0) * (label == 1))
    
    return TP, TN, FP, FN

def metrics(TP, TN, FP, FN, eps=1e-6):
    """
    calculate accuracy, precision, recall and f1.
    set eps=1e-6 to avoid dividing zero
    """
    # accuracy
    acc = (TP + TN) / (TP + TN + FP + FN)

    # precision
    pre = TP / (TP + FP + eps)

    # recall
    recall = TP / (TP + FN + eps)

    # f1
    f1 = 2 * pre * recall / (pre + recall + eps)

    return acc, pre, recall, f1

def cross_validation(data_vec, label):
    """
    ten fold validation
    """
    fold = 10
    data_len = len(data_vec)
    batch_size = int(1 / fold * data_len)
    TP, TN, FP, FN = np.empty(fold), np.empty(fold), np.empty(fold), np.empty(fold)
    for i in range(fold):
        train_data = np.concatenate((data_vec[0 : i * batch_size], data_vec[(i + 1) * batch_size : ]), axis=0)
        train_label = np.concatenate((label[0 : i * batch_size], label[(i + 1) * batch_size : ]), axis=0)
        val_data = data_vec[i * batch_size : (i + 1) * batch_size]
        val_label = label[i * batch_size : (i + 1) * batch_size]
        pos_prob, neg_prob, pos_cls = train(train_data, train_label)
        tp, tn, fp, fn = validate(val_data, val_label, pos_prob, neg_prob, pos_cls)
        TP[i] = tp
        TN[i] = tn
        FP[i] = fp
        FN[i] = fn
    
    acc, pre, recall, f1 = metrics(TP, TN, FP, FN)
    macro_acc, macro_pre, macro_recall, macro_f1 = acc.mean(), pre.mean(), recall.mean(), f1.mean()
    micro_acc, micro_pre, micro_recall, micro_f1 = metrics(sum(TP), sum(TN), sum(FP), sum(FN))
    return (macro_acc, macro_pre, macro_recall, macro_f1), (micro_acc, micro_pre, micro_recall, micro_f1)


if __name__ == '__main__':

    # set parameters
    data_file = 'data/ChnSentiCorp_htl_all.csv'
    vocab_file = 'data/vocabulary_pearson_40000.txt'
    stopwords_file = 'data/stopwords.txt'
    feature_len = 2000
    data, label = load_data(data_file)
    vocabulary = [str(w.replace('\n', '')) for w in load_txt(vocab_file)][:feature_len]
    stopwords = set(load_txt(stopwords_file))
    
    # encode input data
    data_vec = np.empty((len(label), feature_len))
    label = np.array(label)
    for i in range(len(data)):
        data_vec[i] = get_sentence_feature(data[i], vocabulary, stopwords)

    # shuffle data
    N = np.random.permutation(len(label))
    data_vec = data_vec[N]
    label = label[N]

    # ten fold cross validation, get macro and micro metrics
    (macro_acc, macro_pre, macro_recall, macro_f1), (micro_acc, micro_pre, micro_recall, micro_f1) = cross_validation(data_vec, label)
    print('ten fold cross validation')
    print('macro accuracy = %.6f' % macro_acc)
    print('macro precision = %.6f' % macro_pre)
    print('macro recall = %.6f' % macro_recall)
    print('macro f1 = %.6f' % macro_f1)
    print('micro accuracy = %.6f' % micro_acc)
    print('micro precision = %.6f' % micro_pre)
    print('micro recall = %.6f' % micro_recall)
    print('micro f1 = %.6f' % micro_f1)

    # train all data
    pos_prob, neg_prob, pos_cls = train(data_vec, label)

    # prediction
    sentence = '这家酒店很差'
    sent_vec = get_sentence_feature(sentence, vocabulary, stopwords)
    predict(sent_vec, pos_prob, neg_prob, pos_cls)
