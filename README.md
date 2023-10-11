<div align="center">
  <img src="resources/mmrotate-logo.png" width="450"/>
  <div>&nbsp;</div>
  <div align="center">
    <b><font size="5">OpenMMLab website</font></b>
    <sup>
      <a href="https://openmmlab.com">
        <i><font size="4">HOT</font></i>
      </a>
    </sup>
    &nbsp;&nbsp;&nbsp;&nbsp;
    <b><font size="5">OpenMMLab platform</font></b>
    <sup>
      <a href="https://platform.openmmlab.com">
        <i><font size="4">TRY IT OUT</font></i>
      </a>
    </sup>
  </div>
  <div>&nbsp;</div>

[![PyPI](https://img.shields.io/pypi/v/mmrotate)](https://pypi.org/project/mmrotate)
[![docs](https://img.shields.io/badge/docs-latest-blue)](https://mmrotate.readthedocs.io/en/latest/)
[![badge](https://github.com/open-mmlab/mmrotate/workflows/build/badge.svg)](https://github.com/open-mmlab/mmrotate/actions)
[![codecov](https://codecov.io/gh/open-mmlab/mmrotate/branch/main/graph/badge.svg)](https://codecov.io/gh/open-mmlab/mmrotate)
[![license](https://img.shields.io/github/license/open-mmlab/mmrotate.svg)](https://github.com/open-mmlab/mmrotate/blob/main/LICENSE)
[![open issues](https://isitmaintained.com/badge/open/open-mmlab/mmrotate.svg)](https://github.com/open-mmlab/mmrotate/issues)
[![issue resolution](https://isitmaintained.com/badge/resolution/open-mmlab/mmrotate.svg)](https://github.com/open-mmlab/mmrotate/issues)

[📘Documentation](https://mmrotate.readthedocs.io/en/latest/) |
[🛠️Installation](https://mmrotate.readthedocs.io/en/latest/install.html) |
[👀Model Zoo](https://mmrotate.readthedocs.io/en/latest/model_zoo.html) |
[🆕Update News](https://mmrotate.readthedocs.io/en/latest/changelog.html) |
[🚀Ongoing Projects](https://github.com/open-mmlab/mmrotate/projects) |
[🤔Reporting Issues](https://github.com/open-mmlab/mmrotate/issues/new/choose)

</div>

<!--中/英 文档切换-->

<div align="center">

English | [简体中文](README_zh-CN.md)

</div>

## Introduction
The original project is mmrotate. I forked a copy for my personal study. Let's hope I did it the correct way. 

I might need to put this to private but I don't know how for the moment. 

- The 1st step, or maybe the 0th step, is to use the "vanilla" version properly, get with the program. 
- [x] through this process, I might want to keep track of commands I use to "set up shops" and automate.
  - seems a bit hard to do. bad docs. search error msg can solve most things.
- The current step is to study a mod version of the project, supposedly. 
- [x] to understand the changes, I might want to update folder by folder. This could also be a backup when I cannot fully understand or capture all the changes.
  - didn't do this part. I probably don't need to. 

daily todo here
- [x] anaylyze tensorboard from yesterday
- [x] o-rep on 2 dataset train more + 200
- [x] r-frcnn implementation


start a todo list here
- [ ] get the result on speed
- [ ] think and search about overfitting
  - [ ] L1 or combination of L1/L2
  - [ ] dropout `true`
  - [ ] increase `iou_thres` in `batch_non_max_suppression`
  - [ ] adam
  - [ ] apparently, lr policy also helps
- [x] change class weights to "balance" weak/strong classes
  - by assign class gain in config file, roi_head, bbox_head, loss
  - [ ] do quick test to confirm we actually have the "correct" understanding of class weight
- [x] reset lr policy to be constant/fixed (by default, it's step)
- [ ] step: change when overfitting (val loss up instead of down)
  - read a bit more, step is not for solving overfitting it seems
  - warm up: "ResNet models, which use a low learning rate that increases linearly to the base learning rate in the first five epochs, and then decreases by a factor of 10 every 30 epochs.". our back bone is resnet. 
  - current steping is not working properly, maybe need to give up
  - already applied to some of the model/set
  - can also use step outside of max epoch to "disable" step
- [x] mentioned [focal loss](https://mmrotate.readthedocs.io/en/stable/_modules/mmrotate/models/losses/smooth_focal_loss.html), alpha, beta
  - model 2 and 4 use this, tuned alpha and gamma
- [ ] even balance tasks: class/regress
- [x] try [confusion matrix](https://mmrotate.readthedocs.io/en/stable/useful_tools.html#confusion-matrix) for visualization
  - worote a .sh to automate gen this for all 8 cases
- [x] run train back to back with sript
- [x] fix cannot resume or load
  - [x] found [this issue](https://github.com/open-mmlab/mmdetection/issues/10438#issuecomment-1633894504) and try it out
  - probably not the version in use, not really useful
  - [x] confirm hard to get resume working. load works. 
- seems that we have everything from the guy, redo that?
  - [x] PRIORITY: get the code working with our structure of files

start a held list here
- [ ] fine tune idea: looks like by default, stage 1 is frozen at all times. maybe cannot unfreeze. might run into CUDA mem error.
- [x] balanced set: the built-in one seems unsupported with numpy. maybe write my own as custom. 
  - current env: 

| Package Name            | Version  |
|-------------------------|----------|
| mmcv-full               | 1.7.1    |
| mmdet                   | 2.28.2   |
| mmengine                | 0.8.4    |
| mmrotate                | 0.3.4    |
| numpy                   | 1.24.3   |

  - somehow, balanced set is using `numpy.int`, which is removed in 1.20.0
  - if we go `numpy'<1.20.0'`, we will have 1.19.5, which doesn't support `matplotlib`
  - to get around it and mod the file locally, we narrow down to write our own custom dataset and wrapper
  - it's confirmed that we don't have that error anymore but whether it helps with an unbalanced set still needs to be tested
  - these mods should not affect other part of the code
- [x] finally stumbled upon this: [PolyRandomRotate](https://mmrotate.readthedocs.io/en/stable/api.html#mmrotate.datasets.pipelines.PolyRandomRotate). seems like it only rotate certain classes, might help with unbalanced. however, I still feel like we should sample this class more, or have more in this class, rather than rotate
  - applied this to SOME current "test" versions of config, maybe helped with overfitting
  - [ ] maybe do this to all cases
  - [ ] explore `rect_class`, currently use `None`

tensorboard reading notes here:
- SRS o_rcnn
  - "best"
    - has a "nice" gain we like to follow
    - suspect things happened in the beginning and later of the train -> step policy
  - constant lr "sweep"
    - 5e-2, too big?
    - 5e-3, what we used, OK, something happened before 1/4 or 1/5
    - 5e-4, maybe small, maybe need longer, something happened around 1/4
- CAS o_rcnn
  - max_epoch = 60, something happened right in the middle
  - 150 new best
    - consider step before 1/4, best at 11k/43k, 0.777
  - 50CA1e2
    - also something happened around 12k, coincidence? 0.75. 1e-2 might to large
- o_rep
  - ~max_epoch = 80, hasn't reach overfitting yet -> run longer, load from~
  - CA, could train longer, almost stable, maybe small lr, something good at 5k/22k
    - to 150, increased a little, consider step
  - SR, definitely train longer, 80 at 0.4, meaning 160 to see if 0.6, start with larger lr?
    - increased indeed, consider maybe even longer, 200. 150 is now 
- rfrcnn
  - SR, 100, best at 60, seems flat, consider step

note for given
- seems range(0,1) is the same as 0 (1 gpu train)
- all have specified work_dir, good, run for SR
- need to change data_root or move data
- need to change sample_per_gpu and worker_per_gpu
  - see [this page](https://github.com/Adamdad/ConsistentTeacher/issues/19) here, we will need to try
  - CA reached 8, 8. (rfrcnn) but may not help
- use --auto-resume flag in cmd to get true and seems to be working
  - resume not working but auto fine...

MMRotate is an open-source toolbox for rotated object detection based on PyTorch.
It is a part of the [OpenMMLab project](https://github.com/open-mmlab).

The master branch works with **PyTorch 1.6+**.

https://user-images.githubusercontent.com/10410257/154433305-416d129b-60c8-44c7-9ebb-5ba106d3e9d5.MP4

<details open>
<summary><b>Major Features</b></summary>

- **Support multiple angle representations**

  MMRotate provides three mainstream angle representations to meet different paper settings.

- **Modular Design**

  We decompose the rotated object detection framework into different components,
  which makes it much easy and flexible to build a new model by combining different modules.

- **Strong baseline and State of the art**

  The toolbox provides strong baselines and state-of-the-art methods in rotated object detection.

</details>

## What's New

### Highlight

We are excited to announce our latest work on real-time object recognition tasks, **RTMDet**, a family of fully convolutional single-stage detectors. RTMDet not only achieves the best parameter-accuracy trade-off on object detection from tiny to extra-large model sizes but also obtains new state-of-the-art performance on instance segmentation and rotated object detection tasks. Details can be found in the [technical report](https://arxiv.org/abs/2212.07784). Pre-trained models are [here](https://github.com/open-mmlab/mmrotate/tree/1.x/configs/rotated_rtmdet).

[![PWC](https://img.shields.io/endpoint.svg?url=https://paperswithcode.com/badge/rtmdet-an-empirical-study-of-designing-real/real-time-instance-segmentation-on-mscoco)](https://paperswithcode.com/sota/real-time-instance-segmentation-on-mscoco?p=rtmdet-an-empirical-study-of-designing-real)
[![PWC](https://img.shields.io/endpoint.svg?url=https://paperswithcode.com/badge/rtmdet-an-empirical-study-of-designing-real/object-detection-in-aerial-images-on-dota-1)](https://paperswithcode.com/sota/object-detection-in-aerial-images-on-dota-1?p=rtmdet-an-empirical-study-of-designing-real)
[![PWC](https://img.shields.io/endpoint.svg?url=https://paperswithcode.com/badge/rtmdet-an-empirical-study-of-designing-real/object-detection-in-aerial-images-on-hrsc2016)](https://paperswithcode.com/sota/object-detection-in-aerial-images-on-hrsc2016?p=rtmdet-an-empirical-study-of-designing-real)

| Task                     | Dataset | AP                                   | FPS(TRT FP16 BS1 3090) |
| ------------------------ | ------- | ------------------------------------ | ---------------------- |
| Object Detection         | COCO    | 52.8                                 | 322                    |
| Instance Segmentation    | COCO    | 44.6                                 | 188                    |
| Rotated Object Detection | DOTA    | 78.9(single-scale)/81.3(multi-scale) | 121                    |

<div align=center>
<img src="https://user-images.githubusercontent.com/12907710/208044554-1e8de6b5-48d8-44e4-a7b5-75076c7ebb71.png"/>
</div>

**0.3.4** was released in 01/02/2023:

- Fix compatibility with numpy, scikit-learn, and e2cnn.
- Support empty patch in Rotate Transform
- use iof for RRandomCrop validation

Please refer to [changelog.md](docs/en/changelog.md) for details and release history.

## Installation

MMRotate depends on [PyTorch](https://pytorch.org/), [MMCV](https://github.com/open-mmlab/mmcv) and [MMDetection](https://github.com/open-mmlab/mmdetection).
Below are quick steps for installation.
Please refer to [Install Guide](https://mmrotate.readthedocs.io/en/latest/install.html) for more detailed instruction.

dd: this seems to be outdated. the installation steps in the [Install Guide](https://mmrotate.readthedocs.io/en/latest/install.html) have a more updated version. 

dd: although the one in the docs also has some problem. once reached verify, step 1 will already give you some error. the inference demo is even worse. 

dd: lesson of the day 20230923: why did I try to do this on windows? at least start from WSL

dd: change to WSL still has trouble downloading o_rcnn.py. but there is already a version of it in configs folder, using that one for the moment. somehow dd managed to run the download line. do comparison later. 

```shell
conda create -n open-mmlab python=3.7 pytorch==1.7.0 cudatoolkit=10.1 torchvision -c pytorch -y
conda activate open-mmlab
pip install openmim
mim install mmcv-full
mim install mmdet
# dd: will stuck here with "OSError: CUDA_HOME environment variable is not set. 
#     Please set it to your CUDA install root."
git clone https://github.com/open-mmlab/mmrotate.git
cd mmrotate
pip install -r requirements/build.txt
pip install -v -e .
```

## Get Started

Please see [get_started.md](docs/en/get_started.md) for the basic usage of MMRotate.
We provide [colab tutorial](demo/MMRotate_Tutorial.ipynb), and other tutorials for:

- [learn the basics](docs/en/intro.md)
- [learn the config](docs/en/tutorials/customize_config.md)
- [customize dataset](docs/en/tutorials/customize_dataset.md)
- [customize model](docs/en/tutorials/customize_models.md)
- [useful tools](docs/en/tutorials/useful_tools.md)

## Model Zoo

Results and models are available in the *README.md* of each method's config directory.
A summary can be found in the [Model Zoo](docs/en/model_zoo.md) page.

<details open>
<summary><b>Supported algorithms:</b></summary>

- [x] [Rotated RetinaNet-OBB/HBB](configs/rotated_retinanet/README.md) (ICCV'2017)
- [x] [Rotated FasterRCNN-OBB](configs/rotated_faster_rcnn/README.md) (TPAMI'2017)
- [x] [Rotated RepPoints-OBB](configs/rotated_reppoints/README.md) (ICCV'2019)
- [x] [Rotated FCOS](configs/rotated_fcos/README.md) (ICCV'2019)
- [x] [RoI Transformer](configs/roi_trans/README.md) (CVPR'2019)
- [x] [Gliding Vertex](configs/gliding_vertex/README.md) (TPAMI'2020)
- [x] [Rotated ATSS-OBB](configs/rotated_atss/README.md) (CVPR'2020)
- [x] [CSL](configs/csl/README.md) (ECCV'2020)
- [x] [R<sup>3</sup>Det](configs/r3det/README.md) (AAAI'2021)
- [x] [S<sup>2</sup>A-Net](configs/s2anet/README.md) (TGRS'2021)
- [x] [ReDet](configs/redet/README.md) (CVPR'2021)
- [x] [Beyond Bounding-Box](configs/cfa/README.md) (CVPR'2021)
- [x] [Oriented R-CNN](configs/oriented_rcnn/README.md) (ICCV'2021)
- [x] [GWD](configs/gwd/README.md) (ICML'2021)
- [x] [KLD](configs/kld/README.md) (NeurIPS'2021)
- [x] [SASM](configs/sasm_reppoints/README.md) (AAAI'2022)
- [x] [Oriented RepPoints](configs/oriented_reppoints/README.md) (CVPR'2022)
- [x] [KFIoU](configs/kfiou/README.md) (arXiv)
- [x] [G-Rep](configs/g_reppoints/README.md) (stay tuned)

</details>

## Data Preparation

Please refer to [data_preparation.md](tools/data/README.md) to prepare the data.

## FAQ

Please refer to [FAQ](docs/en/faq.md) for frequently asked questions.

## Contributing

We appreciate all contributions to improve MMRotate. Please refer to [CONTRIBUTING.md](.github/CONTRIBUTING.md) for the contributing guideline.

## Acknowledgement

MMRotate is an open source project that is contributed by researchers and engineers from various colleges and companies. We appreciate all the contributors who implement their methods or add new features, as well as users who give valuable feedbacks. We wish that the toolbox and benchmark could serve the growing research community by providing a flexible toolkit to reimplement existing methods and develop their own new methods.

## Citation

If you use this toolbox or benchmark in your research, please cite this project.

```bibtex
@inproceedings{zhou2022mmrotate,
  title   = {MMRotate: A Rotated Object Detection Benchmark using PyTorch},
  author  = {Zhou, Yue and Yang, Xue and Zhang, Gefan and Wang, Jiabao and Liu, Yanyi and
             Hou, Liping and Jiang, Xue and Liu, Xingzhao and Yan, Junchi and Lyu, Chengqi and
             Zhang, Wenwei and Chen, Kai},
  booktitle={Proceedings of the 30th ACM International Conference on Multimedia},
  year={2022}
}
```

## License

This project is released under the [Apache 2.0 license](LICENSE).

## Projects in OpenMMLab

- [MMCV](https://github.com/open-mmlab/mmcv): OpenMMLab foundational library for computer vision.
- [MIM](https://github.com/open-mmlab/mim): MIM installs OpenMMLab packages.
- [MMClassification](https://github.com/open-mmlab/mmclassification): OpenMMLab image classification toolbox and benchmark.
- [MMDetection](https://github.com/open-mmlab/mmdetection): OpenMMLab detection toolbox and benchmark.
- [MMDetection3D](https://github.com/open-mmlab/mmdetection3d): OpenMMLab's next-generation platform for general 3D object detection.
- [MMRotate](https://github.com/open-mmlab/mmrotate): OpenMMLab rotated object detection toolbox and benchmark.
- [MMSegmentation](https://github.com/open-mmlab/mmsegmentation): OpenMMLab semantic segmentation toolbox and benchmark.
- [MMOCR](https://github.com/open-mmlab/mmocr): OpenMMLab text detection, recognition, and understanding toolbox.
- [MMPose](https://github.com/open-mmlab/mmpose): OpenMMLab pose estimation toolbox and benchmark.
- [MMHuman3D](https://github.com/open-mmlab/mmhuman3d): OpenMMLab 3D human parametric model toolbox and benchmark.
- [MMSelfSup](https://github.com/open-mmlab/mmselfsup): OpenMMLab self-supervised learning toolbox and benchmark.
- [MMRazor](https://github.com/open-mmlab/mmrazor): OpenMMLab model compression toolbox and benchmark.
- [MMFewShot](https://github.com/open-mmlab/mmfewshot): OpenMMLab fewshot learning toolbox and benchmark.
- [MMAction2](https://github.com/open-mmlab/mmaction2): OpenMMLab's next-generation action understanding toolbox and benchmark.
- [MMTracking](https://github.com/open-mmlab/mmtracking): OpenMMLab video perception toolbox and benchmark.
- [MMFlow](https://github.com/open-mmlab/mmflow): OpenMMLab optical flow toolbox and benchmark.
- [MMEditing](https://github.com/open-mmlab/mmediting): OpenMMLab image and video editing toolbox.
- [MMGeneration](https://github.com/open-mmlab/mmgeneration): OpenMMLab image and video generative models toolbox.
- [MMDeploy](https://github.com/open-mmlab/mmdeploy): OpenMMLab model deployment framework.
