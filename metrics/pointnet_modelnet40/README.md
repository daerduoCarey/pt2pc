Fill in the form, download `pn_ckpt.zip` and unzip here.


# If you want to train the PointNet on ModelNet40


## ModelNet40

Download data from `https://shapenet.cs.stanford.edu/media/modelnet40_ply_hdf5_2048.zip`.

## PointNet Training

Use data augmentation during training: random along-y-axis rotation + jittering. 
This is necessary to make sure the feature is not sensitive to the along-y rotation.
Otherwise, chairs and sofas (all classified as chair in PartNet) will face to different orientations while being trained in ModelNet40 classification task.

Code adapted from https://github.com/WangYueFt/dgcnn

