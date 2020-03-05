import numpy as np
from sklearn.neighbors import KNeighborsClassifier


common_corruptions = ['gaussian_noise', 'shot_noise', 'impulse_noise', 'defocus_blur', 'glass_blur',
				'motion_blur', 'zoom_blur', 'snow', 'frost', 'fog',
				'brightness', 'contrast', 'elastic_transform', 'pixelate', 'jpeg_compression', 'scale']



train_view = 'Lab'

tr_feat = np.load('./results/feat_from_model/tr_%s.npy' %(train_view))
tr_label = np.load('./results/feat_from_model/tr_label.npy')

neigh = KNeighborsClassifier(n_neighbors=5)
neigh.fit(tr_feat, tr_label)

for corruption in common_corruptions:
    val_corruption = corruption

    te_feat = np.load('./results/feat_from_model/val_%s_%s.npy' %(train_view, val_corruption))
    te_label = np.load('./results/feat_from_model/val_label.npy')


    predict = neigh.predict(te_feat)
    correct = np.sum( predict == te_label )

    accuracy = correct / predict.shape[0]
    print( 'The accuracy of %s is %s' %(val_corruption, str(accuracy*100)) )