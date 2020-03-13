import sys
sys.path.append('..')
from my_liblinear.python.liblinearutil import *
import numpy as np

train_view = 'Lab'
val_corruption = 'jpeg_compression'

tr_feat = np.load('./results/feat_from_model/tr_%s.npy' %(train_view))
tr_label = np.load('./results/feat_from_model/tr_label.npy')

te_feat = np.load('./results/feat_from_model/val_%s_%s.npy' %(train_view, val_corruption))
te_label = np.load('./results/feat_from_model/val_label.npy')

m = train(tr_label, tr_feat, '-c 4')

p_label, p_acc, p_val = predict(te_label, te_feat, m)