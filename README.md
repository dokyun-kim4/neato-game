<p align="center">
  <img src="img/project-logo.png" />
</p>

### Authors: Dokyun Kim & [Dominic Salmieri](https://github.com/joloujo)
<br>

# Introduction
Using OpenCV and a Neato, we will recreate the **Red Light Green Light** game from the Netflix show *Squid Game*. The Neato will periodically rotate towards the players, and whoever is still moving gets eliminated from the game.

# Methodology
This program uses 2 popular computer vision algorithms (YOLO and SORT) to track people's movement.

Tracking algorithms usually consist of 4 main steps. 
1. Identification
2. Extract features
3. Calculate distance
4. Match pairs

**Step 1** is done using YOLO (You Only Look Once) while **steps 2, 3 and 4** will be done using SORT (Simple Online Realtime Tracking). 
An explanation of each algorithm will be provided in the sections below.

## Using YOLO (You Only Look Once)

Unlike other detection methods such as HOG (Histogram of Gradients), RCNN, or CNN, YOLO significantly outperforms them in speed. YOLO v1, released in 2016, processed 45 frames per second on a Titan X GPU. YOLO locates and classifies an object at the same time in a one-step process, hence the name 'You Only Look Once.'

The description below is based on the structure of YOLO v1. This program uses the latest YOLO v8, but the governing concepts behind them are similar.

YOLO divides a given image into a S x S grid, represented with the yellow lines in the image below. The red boxes are objects identified by the algorithm. 

<p>
  <img src="img/yolo.png" />
</p>

*Fig 1: YOLO example*


Each grid cell is represented as a multidimensional vector. The first 5 values are $[x_1,y_1,w_1,h_1,c_1]$, where $(x_1,y_1), w_1, h_1$ are the position, width, and height of the bounding box, and $c_1$ is the confidence level (0~1). The next 5 values are $[x_2,y_2,w_2,h_2,c_2]$, as each grid cell can handle up to 2 bounding boxes. The remaining values are $[p_1...p_80]$ where each value represents what object the box belongs to in the train dataset. This example uses the COCO dataset, which has 80 objects. For example, if the 3rd object in the dataset was a person, the values would look like $[0,0,1,....0]$. The final output of the neural network ends up being a S x S x 90 tensor.

Compared to neural networks of RCNN-type algorithms, YOLO's neural network is much simpler. As shown below, YOLO's neural network consists of 24 convolutional layers, 4 max-pooling layers, and 2 fully-connected layers. 

<p>
  <img src="img/neuralnet.png" />
</p>

*Fig 2: YOLO's neural network structure*


## Using SORT (Simple Online Realtime Tracking)

<!-- SORT INTRODUCTION -->

Let $B_{detection}$ represent all the bounding boxes containing people that are identified by YOLO. During the feature extraction process, SORT uses the target's size and past movement from time $t-1$ (equation 1). $(x,y)$ is the target's center, $s$ is size, $r$ is the height to width ratio (fixed). $\dot{x}, \dot{y}, \dot{s}$ represent the previous movement performed by the target. SORT then uses this information to predict the target's location at time $t$. The predictions are stored in $B_{prediction}$.

$b=(x,y,s,r, \dot{x}, \dot{y}, \dot{s})$ *Equation 1*

Then, the IOU (Intersection Over Union) of $B_{detection}$ and $B_{prediction}$ is calculated and converted to distance by subtracting it from 1. These distances are then stored in a matrix. If the number of boxes in $B_{detection}$ and $B_{prediction}$ are different, placeholder boxes are added to the smaller set to ensure both matrices are square. The placeholder boxes have a large distance value to prevent matching with real boxes. A distance matrix example is shown below.

<p>
  <img src="img/distance-mat.png" style="width: 50%;" />
</p>

*Fig 3: Example distance matrix calculated using* $B_{detection} = [1,2,3,4]$ *and* $B_{predicted} = [a,b,c]$. *A placeholder box* $d$ *has been added to* $B_{predicted}$ *to form a square matrix.*

With the distance matrix, SORT now applies the [Hungarian Algorithm](https://en.wikipedia.org/wiki/Hungarian_algorithm) to find the best-matching pairs. The Hungarian Algorithm is an optimization algorithm that assigns 'tasks' to 'workers' to minimize the 'cost.' For example, when assigning tasks (a-c) to workers (1-3) given the cost matrix below (Fig 4), the pairs 1-c, 2-a, 3-b would minimize the cost.

<p>
  <img src="img/hungarian-example.png" style="width: 50%;" />
</p>

*Fig 4: Example of the Hungarian Algorithm, the highlighted cells represent the pairings that minimize the cost.*

With this in mind, finding best-matching pairs using the distance matrix (Fig 3) becomes an optimization problem for assigning boxes from $B_{predicted}$ (task) to $B_{detected}$ (workers) to minimize the distances (cost).

<p>
  <img src="img/hungarian-img-example.png" style="width: 50%;" />
</p>

*Fig 5: Result of applying the Hungarian Algorithm to the distance matrix. The best-matching pairs are 1-b, 2-d, 3-a, and 4-c. Since column d was added for the sake of making the matrix square, the 2-d pairing gets thrown out.*

Once SORT finishes all the steps described above, it moves onto the next frame after postprocessing. During postprocessing, the $b$ values (equation 1) of each target in $B_{predicted}$ gets updated depending on if they have been matched. For targets with a match, their $x,y,s,r$ is replaced with those of their matching pairs, and their $\dot{x}, \dot{y}, \dot{s}$ is updated using a [Kalman Filter](https://en.wikipedia.org/wiki/Kalman_filter). For targets without a match, their $x,y,s$ is updated by adding $\dot{x}, \dot{y}, \dot{s}$. For boxes in $B_{detected}$ without a match, they are considered new objects and their *b* values are initialized with $\dot{x}=0, \dot{y}=0, \dot{s}=0$. The new boxes are added to $B_{predicted}$.

<!-- IMAGE FROM ACTUAL PROGRAM WITH BBOX IDS HERE -->

## Image Substraction

LOREM IPSUM

# Works Cited

Bewley, Alex, et al. "Simple online and realtime tracking." 2016 IEEE international conference on image processing (ICIP). IEEE, 2016. 

Oh, Il-Seok. Pattern Recognition, Computer Vision, and Machine Learning. Hanbit Academy, 2023.  

Redmon, Joseph, et al. "You only look once: Unified, real-time object detection." Proceedings of the IEEE conference on computer vision and pattern recognition. 2016.
