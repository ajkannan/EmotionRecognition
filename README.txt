Testing our Project

The following steps explain how to run our code. 

1. Obtaining the Dataset

The dataset of facial images we used is the Cohn_Kanade AU-Coded Expression Database.  We cannot include the images in the submission of the project as per the rules of distribution of the dataset, but the dataset can be downloaded after a quick agreement from the following URL: http://www.consortium.ri.cmu.edu/ckagree/

For more information about the dataset, see the link provided: http://www.pitt.edu/~emotion/ck-spread.htm

2. Compiling Images into Matlab Data File

The script label_parser.py has been provided to compile images into a data file to be used in our algorithms.  The script ignores pictures that do not have associated labels, and it pulls only the last image in the image sequence for each subject's emotion, since the goal of the project was to classify emotion based on still images.

3. Obtaining the Cropped Face Data

The cropped faces were used to the isolate eyes and mouth later in the algorithm.  Run the MATLAB script load data.m.  This should output a datafile named imagedata_*.mat, where * may be filled be a number of possible extensions reflecting the created representation of the images.  Ensure that this .mat file is placed in the same folder as the script extractmouth.m


4. Isolating the Eyes and Mouth

Because the anthill cluster does not include the computer vision toolbox, it is necessary to run the following command before opening the MATLAB environment:

export LM_LICENSE_FILE=27000@anthill.cs.dartmouth.edu:1711@rclserv1.dartmouth.edu

To obtain the results providing the best results on our data set, run the code "loaddata_phog.m". This uses the off-the-shelf Viola-Jones algorithm to isolate the mouth and eyes. We also implement an eye detector using cross-correlation. To obtain the image representations calculated by using the cross-correlation eyes, run the scripts of the form "loaddata_ahog.m". 

5. Running the Random Forest

Run the script rfemotion.sh on the cluster for fast results. You may submit the job via "qsub rfemotion.sh". Ensure that the supporting files (which can be identified with imagedata*.mat) are included in the directory. In particular, to run the Random Forest with all representations we tried, edit the file "rfemotion.m" and on the first line write: "data = dir('Path/To/Data/imagedata*.mat');"


External Software

We used the Cohn-Kanade AU-Coded Expression Database, copyright Jeffrey Cohn.  It can be downloaded from the following link: http://www.consortium.ri.cmu.edu/ckagree/

In our final algorithm, we used RUSBoost Random Forest, which can be found at: http://www.mathworks.com/help/stats/ensemble-methods.html

For testing and comparison purposes, we used the following external software
- Viola-Jones by Mayasuki Tanaka: http://www.mathworks.com/matlabcentral/fileexchange/36855-face-parts-detection
- SIFT by David Lowe: http://www.cs.ubc.ca/~lowe/keypoints/
- Deep Learning Toolbox by Rasmus Berg Palm - http://mathworks.com/matlabcentral/fileexchange/38310-deep-learning-toolbox

We adapted and improved the implementation of HOG features by Ludwig Oswaldo, which can be found at: http://www.mathworks.com/matlabcentral/fileexchange/28689-hog-descriptor-for-matlab
