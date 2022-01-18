# EEG motor imagery project

The goal of this project is to predict imagined movements (termed 'motor imageries') from EEG recordings. Data is taken from Kaya et al., 2018

There are three paradigms for this BCI task:

* CLA - Three class classification between imagined left hand movements, imagined right hand movements, and a passive state.
* HaLT - Six class classification. Same as CLA but with additional classes for imagined left foot, right foot, and tongue movements.
* 5F - Five class classification for imagined movements for each of the fingers on one hand.

The first method I tried is the winner of the BCI Competition IV Datasets [2a](http://www.bbci.de/competition/iv/results/#dataset2a) and [2b](http://www.bbci.de/competition/iv/results/#dataset2b)

## Filter Bank Common Spatial Pattern (FBCSP)

![FBCSP](https://github.com/AsaBarthMaron/asabarthmaron.github.io/blob/master/files/FBCSP.png)
(Chin et al., 2009)

* **Filter Bank**. Raw EEG data is band-pass filtered, expanding the feature set.

* **Common Spatial Pattern**. Perform spatial filtering (dimensionality reduction) using CSP (Koles et al., 1990), multiclass extension with OVR.

* **Mutual Information-based feature selection (MIBIF)** for feature selection. 

* **Naive Bayes Classifier** using kernel density estimation / parzen window (NBPW).

## Results
Below are the accuracies from my FBCSP implementation compared to performance of an SVM from Kaya et al., 2018
![Accuracy comparison](https://github.com/AsaBarthMaron/asabarthmaron.github.io/blob/master/files/fbcsp_results.png)




Chin, Z., Ang, K., Wang, C., Guan, C., and Zhang, H. (2009). 
    Multi-class Filter Bank Common Spatial Pattern for Four-Class Motor 
    Imagery BCI.
    
Kaya, M., Binli, M.K., Ozbay, E., Yanar, H., and Mishchenko, Y. (2018). 
    A large electroencephalographic motor imagery dataset for 
    electroencephalographic brain computer interfaces. 
    Scientific Data 5, 180211.

Koles, Z.J., Lazar, M.S., and Zhou, S.Z. (1990). Spatial patterns 
    underlying population differences in the background EEG. Brain 
    Topogr 2, 275–284.
