# PT2PC: Learning to Generate 3D Point Cloud Shapes from Part Tree Conditions

![Overview](https://github.com/daerduoCarey/pt2pc/blob/master/images/teaser.png)

**Figure 1.** We formulate the problem of Part-Tree-to-Point-Cloud (PT2PC) as a conditional generation task which takes in a symbolic part tree as condition and generates multiple point clouds with shape variations satisfy the structure defined by the part tree.

## Introduction
This paper investigates the novel problem of generating 3D shape point cloud geometry from a symbolic part tree representation. In order to learn such a conditional shape generation procedure in an end-to-end fashion, we propose a conditional GAN "part tree"-to-"point cloud" model (PT2PC) that disentangles the structural and geometric factors.

## About the paper

Our team: 
[Kaichun Mo](https://cs.stanford.edu/~kaichun),
[He Wang](http://ai.stanford.edu/~hewang/),
[Xinchen Yan](https://sites.google.com/site/skywalkeryxc/),
and [Leonidas J. Guibas](https://geometry.stanford.edu/member/guibas/) 
from 
Stanford University

Arxiv Version: https://arxiv.org/abs/2003.08624

Project Page: https://cs.stanford.edu/~kaichun/pt2pc/

## Citations


    @article{mo2020pt2pc,
        title={{PT2PC}: Learning to Generate 3D Point Cloud Shapes from Part Tree Conditions},
        author={Mo, Kaichun and Wang, He and Yan, Xinchen and Guibas, Leonidas},
        journal={arXiv preprint arXiv:2003.08624},
        year={2020}
    }

## About this repository

This repository provides data and code as follows.


```
    data/                       # contains PartNet data
        Chair_hier/             # contains PartNet part-tree JSON files
        Chair_geo/              # contains PartNet per-part point cloud geometry

    stats/                      # contains helper meta-info
        part_trees/             # contains part-tree templates
            info.txt            # each line lists all PartNet shapes sharing the same part-tree template
            pt-0/
                template.json   # the part-tree template JSON file
                xxx.txt         # for each shape satisfying the template, this file stores the mapping from 
                                # the canonical part ids in template.json to the part ids for shape xxx
                                # in files data/Chair_hier/xxx.json and data/Chair_geo/xxx.npz
        part_semantics/
            Chair.txt           # stores PartNet semantic part hierarchy
        semantics_colors/
            Chair.txt           # stores the palette used for visualization

    log/                        # store training logs

    metrics/                    # store metric-code for the FPD score
        fid.py                  # the main script that computes FPD scores
        gt_stats/               # stores the ground-truth statistics for mean and covariance
        pointnet_modelnet40/    # code to train PointNet on ModelNet40

    train.py                    # the main training script
    trainer.py                  # the WGAN-gp trainer
    model_gen.py                # model definition for the generator
    model_dis.py                # model definition for the discriminator
    data.py                     # the data loader
    utils.py                    # contain utility functions
```

This code has been tested on Ubuntu 16.04 with Cuda 9.0, GCC 5.4.0, Python 3.6.5 and PyTorch 1.1.0. 

Please fill in [this form](https://docs.google.com/forms/d/e/1FAIpQLScEnRD_b4elKVUHAgWomfmadw6-30caNJ5xJ4ahsu-tkTdXBg/viewform?usp=sf_link) to download the necessary data.

```
    data.zip            # put under the root folder and unzip
    part_trees.zip      # put under stats/ and unzip
    gt_stats.zip        # put under metrics/ and unzip
    pn_ckpt.zip         # put under metrics/pointnet_modelnet40/ and unzip
```

## Dependencies

Please run
    
        pip3 install -r requirements.txt

to install the dependencies.

Then, install https://github.com/rusty1s/pytorch_scatter by running

        pip3 install torch-scatter

Please also install

        cd sampling
        python setup.py install
        cd ..
        cp sampling/build/lib.linux-x86_64-3.6/sampling_cuda.cpython-36m-x86_64-linux-gnu.so .

## To train the model

Simply run

        python ./train.py --category Chair

## HierInsSeg Scores

Check the README in `hierinsseg`. This is our proposed structure reconstruction metric.

## Questions

Please post issues for questions and more helps on this Github repo page. We encourage using Github issues instead of sending us emails since your questions may benefit others.

## License

MIT License

## Updates

* [March 27, 2020] Preliminary Data and Code released.
* [May 17, 2020] Release the proposed HierInsSeg score for evaluating shape structure reconstruction.

## TODOs

* Release evaluation code.
* Release pretrained models.
* Release baseline code.

Please request in Github Issue for more code to release.

