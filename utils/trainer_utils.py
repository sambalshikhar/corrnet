
import numpy as np
import time
import math

class AverageMeter(object):
    """Computes and stores the average and current value"""
    def __init__(self, name, fmt=':f'):
        self.name = name
        self.fmt = fmt
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

    def __str__(self):
        fmtstr = '{name} {val' + self.fmt + '} ({avg' + self.fmt + '})'
        return fmtstr.format(**self.__dict__)

def retreival_acc(emb_1,emb_2,instance_labels):
    emb_1= emb_1.detach().cpu().numpy()
    emb_2= emb_2.detach().cpu().numpy()
    distance_matrix = cdist(emb_1,emb_2)
    #print(distance_matrix.shape)
    sorted_distance_matrix = np.argsort(distance_matrix)
    new_sorted_distance_matrix = np.zeros(sorted_distance_matrix.shape)
    labels_y = instance_labels.detach().cpu().numpy()
    #print(labels_y.shape)

    x = sorted_distance_matrix.shape[0]
    y = sorted_distance_matrix.shape[1]

    for i in range(x):
        for j in range(y):
            new_sorted_distance_matrix[i,j] =  labels_y[sorted_distance_matrix[i,j]]
    index = []
    for i,j in enumerate(new_sorted_distance_matrix[:,:]):
        index.append(np.where(j==labels_y[i])[0][0])

    return index    

class ProgressMeter(object):
    def __init__(self, num_batches, meters, prefix=""):
        self.batch_fmtstr = self._get_batch_fmtstr(num_batches)
        self.meters = meters
        self.prefix = prefix

    def display(self, batch):
        entries = [self.prefix + self.batch_fmtstr.format(batch)]
        entries += [str(meter) for meter in self.meters]
        print('\t'.join(entries))

    def _get_batch_fmtstr(self, num_batches):
        num_digits = len(str(num_batches // 1))
        fmt = '{:' + str(num_digits) + 'd}'
        return '[' + fmt + '/' + fmt.format(num_batches) + ']'
