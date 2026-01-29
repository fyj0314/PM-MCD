import numpy as np

class AverageMeter(object):
    """Computes and stores the average and current value"""

    def __init__(self):
        self.initialized = False
        self.val = None
        self.avg = None
        self.sum = None
        self.count = None

    def initialize(self, val, count, weight):
        self.val = val
        self.avg = val
        self.count = count
        self.sum = val * weight
        self.initialized = True

    def update(self, val, count=1, weight=1):
        if not self.initialized:
            self.initialize(val, count, weight)
        else:
            self.add(val, count, weight)

    def add(self, val, count, weight):
        self.val = val
        self.count += count
        self.sum += val * weight
        self.avg = self.sum / self.count

    def value(self):
        return self.val

    def average(self):
        return self.avg

def accuracy(pred, label, ignore_zero=False):
    valid = (label >= 0)
    if ignore_zero: valid = (label > 0)
    acc_sum = (valid * (pred == label)).sum()
    valid_sum = valid.sum()
    acc = float(acc_sum) / (valid_sum + 1e-10)
    return acc, valid_sum

def fast_hist(a, b, n):
    k = (a >= 0) & (a < n)
    return np.bincount(n * a[k].astype(int) + b[k], minlength=n ** 2).reshape(n, n)

def get_hist(image, label, num_class):
    hist = np.zeros((num_class, num_class))
    hist += fast_hist(image.flatten(), label.flatten(), num_class)
    return hist

def SCDD_eval_all(preds, labels, num_class):
    hist = np.zeros((num_class, num_class))
    for pred, label in zip(preds, labels):
        infer_array = np.array(pred)
        unique_set = set(np.unique(infer_array))
        assert unique_set.issubset(set([x for x in range(num_class)])), "unrecognized label number"
        label_array = np.array(label)
        assert infer_array.shape == label_array.shape, "The size of prediction and target must be the same"
        hist += get_hist(infer_array, label_array, num_class)

    hist_fg = hist[1:, 1:]
    c2hist = np.zeros((2, 2))
    c2hist[0][0] = hist[0][0]
    c2hist[0][1] = hist.sum(1)[0] - hist[0][0]
    c2hist[1][0] = hist.sum(0)[0] - hist[0][0]
    c2hist[1][1] = hist_fg.sum()
    hist_n0 = hist.copy()
    hist_n0[0][0] = 0

    # 计算每个类别
    TP = np.zeros(num_class)
    FP = np.zeros(num_class)
    FN = np.zeros(num_class)
    TN = np.zeros(num_class)
    for i in range(num_class):
        TP[i] = hist[i, i]
        FP[i] = hist.sum(1)[i] - hist[i][i]
        FN[i] = hist.sum(0)[i] - hist[i][i]
        TN[i]= np.trace(hist) - hist[i, i]
    Ious = np.diag(hist) / (hist.sum(1) + hist.sum(0) - np.diag(hist))

    # 计算precision, recall, F1
    Pr = [TP[i] / (TP[i] + FP[i] + 1e-7) for i in range(num_class)]
    Re = [TP[i] / (TP[i] + FN[i] + 1e-7) for i in range(num_class)]
    F1 = [2 * (Pr[i] * Re[i]) / (Pr[i] + Re[i] + 1e-7) for i in range(num_class)]

    return Pr, Re, Ious, F1