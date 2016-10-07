# Hand Rotation Estimation Tutorial
This repository presents a tutorial that will guide you through the various step needed to implement a complete pipeline for hand rotation estimation. The details about the estimation algorithms, as well as the articles this work has been inspired by, are described in this [technical report](http://www.cnr.it/peoplepublic/peoplepublic/index/downloadprodotto/i/117721). The tutorial leverages various tools and libraries developed by the [Engineering for Health and Wellbeing group](http://www.ehw.ieiit.cnr.it/?q=computervision) at the [Institute of Electronics, Computer and Telecommunication Engineering](http://www.ieiit.cnr.it) and freely available on [github](https://github.com/mUogoro/). 

The tutorials consists in the following steps:

- generation of the synthetic datasets;
- training of the classifiers for hand segmentation and rotation estimation;
- run-time acquisition and hand segmentation;
- final hand rotation estimation and filtering.

## Background
The following work has been mainly inspired by the research carried out by Microsoft Research in [real-time estimation of human pose using the Kinect sensor](https://www.microsoft.com/en-us/research/project/human-pose-estimation-for-kinect/). The following publication provides the theoretical background about the tracking approach:

Shotton, Jamie, et al. "Efficient human pose estimation from single depth images." IEEE Transactions on Pattern Analysis and Machine Intelligence 35.12 (2013): 2821-2840.

Keskin, Cem, et al. "Real time hand pose estimation using depth sensors." 2011 IEEE International Conference on Computer Vision Workshops (ICCV Workshops).

Thalmann, Daniel, Hui Liang, and Junsong Yuan. "First-Person Palm Pose Tracking and Gesture Recognition in Augmented Reality." International Joint Conference on Computer Vision, Imaging and Computer Graphics. Springer International Publishing, 2015.

Basically, the Random Forest (RF) algorithm is trained to perform either per-part *classification* of human body parts or *regression* of skeletal joint positions using as input the stream of *depthmaps* acquired from the sensor. We use a slightly modified version of these algorithms on the problem of real-time estimation of the 3D orientation of the human hand. More details can be found in this [technical report](http://www.cnr.it/peoplepublic/peoplepublic/index/downloadprodotto/i/117721).

## Software requirements
The code provided with this tutorial depends on the following libraries:
- [RGBD Grabber Library](https://github.com/mUogoro/rgbd_grabber)
- [RGBB Ground Truth Generator](https://github.com/mUogoro/rgbd_ground_truth_generator)
- [Padenti](https://github.com/mUogoro/padenti)
The code provided together with this tutorial has been test on an Ubuntu Linux system. With slight modifications it should also work on Windows.

## Citing
If you use this material of this tutorial in a scientific publication, please cite the following articles

Daniele Pianu
**Hand orientation prediction using Random Forests: preliminary results**
*Technical Report* August 30, 2016 [link](http://www.cnr.it/peoplepublic/peoplepublic/index/downloadprodotto/i/117721)

Daniele Pianu, Roberto Nerino, Claudia Ferraris, and Antonio Chimienti 
**A novel approach to train random forests on GPU for computer vision applications using local features**
*International Journal of High Performance Computing Applications*
December 29, 2015 doi:10.1177/1094342015622672
[abstract](http://hpc.sagepub.com/content/early/2015/12/29/1094342015622672.abstract) [bib](http://hpc.sagepub.com/citmgr?type=bibtex&gca=sphpc%3B1094342015622672v1)

## Generating the synthetic datasets
An RGBD dataset consists of a (potentially large) set of image pairs, where each pair contains:
- a depthmap, typically a single-channel 16 bit image, where each pixel stores the distance of the object from the camera;
- a labels image, where the pixel colour represents the class of the portion of the object the pixel belongs to (e.g. different anatomic parts of the human body) or the class of the object as a whole (e.g. different objects within a room).
We need such a dataset to train a RF model to segment hand pixels from the background in each depthmap. The depthmap-labels pairs can be generated either by manually labelling each depthmap (by hand or using semiautomatic tools) or by resorting to computer graphics tools, i.e. generating synthetic datasets using 3D rendering techniques. A dataset generated using the first approach can be found [here](http://cims.nyu.edu/~tompson/NYU_Hand_Pose_Dataset.htm#overview). Alternatively, we released the [RGBD Ground Truth Generator](https://github.com/mUogoro/rgbd_ground_truth_generator) library to easy the generation of synthetic RGBD datasets.

As far as the 3D hand orientation is concerned, we need once again a training dataset but, since we are now performing regression, no labelling is required and only the orientation information (e.g. Euler angles triple) must be stored together with the depthmap. As in the previous case we can resort to either real or synthetic datasets. Within the script folder of this repository you can find a generate_dataset.m MATLAB script which uses the RGBD Ground Truth Generator to create such a dataset. It starts from a small set of *base rotations* and a set of *base poses*: the former contains representative orientation of the hand, whereas the latter contains a set of hand poses (i.e. fingers' configuration). The dataset is created by choosing a base rotation, a base pose and perturbing both the rotation and the pose in order to obtain a rich and representative dataset. Look at the code comments for more details about the dataset generation process. Whitin this code, the user has to set the following parameters:
- rendering parameters
- number of perturbed poses and std for fingers, wrist and global orientation perturbations
- path of the hand model
- dataset output path

## Training the RF models
We use two different types of RF model:
- one for hand segmentation from the background;
- one for hand rotation regression.
To train both models we use the [Padenti](https://github.com/mUogoro/padenti) library, which provides an optimized implementation of the training algorithm which runs on GPUs.

For the first task (hand segmentation) you can follow the [tutorial](http://muogoro.github.io/padenti/tutorial.html) on the library documentation page.

While the code in the master does not support regression, this feature has been added in the *hand_rotation* [branch](https://github.com/mUogoro/padenti/tree/hand_rotation). The code needed to perform training for regression is very similar to the one used in the previous step for classification and is shown in the *test\_rtree\_trainer.cpp* sample file within the test directory. Look a the inline comments for details about how to launch the training and the needed input data.

Once the training is performed we end up with an xml file representing the trained RF model. One final step consists in adapting the regression RF from a single hypothesis model to a multi-hypothesis one. As described in the technical report, multi-hypothesis models are better at tackling the multimodality of the hand rotation distribution. The 
code for modifying the model is included in the *retrofit_rtree.cpp* file within the test directory; once compiled, pass as input the path of the single-hypothesis input RF and multiple-hypothesis output RF xml files.

Some pre-trained models are also provided within the data directory.

## Run-time hand orientation estimation
The different steps involved in the estimation of hand orientation are the following:
- depthmap acquisition
- hand segmentation
- 3D position tracking
- rotation estimation
- rotation filtering

For the acquisition of the depthmap stream we use the [RGBD Grabber Library](https://github.com/mUogoro/rgbd_grabber), whereas segmentation is performed by the means of the RF classification model trained in the previous phase. Tracking is performed by the Camshift algorithm (we resort to the [OpenCV implementation](http://docs.opencv.org/2.4/modules/video/doc/motion_analysis_and_object_tracking.html#camshift)). Rotation estimation is computed using the RF regression model previously trained. Rotation filtering is required for filtering most of outliers typically returned by the per-frame orientation estimation. We provide a Particle Filter implementation within the src/particle\_filter folder. The whole pipeline is implemented in the live\_test.cpp example within the test directory.

## TODOs
This tutorial is still a work in progress. Many details (especially implementation ones) have been left out for brevity.
