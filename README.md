[![Awesome](https://cdn.rawgit.com/sindresorhus/awesome/d7305f38d29fed78fa85652e3a63e154dd8e8829/media/badge.svg)](https://github.com/sindresorhus/awesome)

List of awesome code, papers, and resources for AI/deep learning/machine learning/neural networks applied to oncology.

Open access: all rights granted for use and re-use of any kind, by anyone, at no cost, under your choice of either the free MIT License or Creative Commons CC-BY International Public License.

© 2022 Craig Bailes ([@cbailes](https://github.com/cbailes) | [contact@craigbailes.com](mailto:contact@craigbailes.com))

## Index
* [Code](#code)
  + [Challenges](#challenges)
  + [Repositories](#repositories)
    - [Brain](#brain)
    - [Breast](#breast)
    - [Esophageal](#esophageal)
    - [Lung](#lung)
    - [Oral](#oral)
    - [Prostate](#prostate)
  + [Datasets](#datasets)
    - [Brain](#brain-1)
    - [Breast](#breast-1)
    - [Pancreatic](#pancreatic)
    - [Prostate](#prostate-1)
* [Papers](#papers)
  + [Meta reviews](#meta-reviews)
  + [Detection and diagnosis](#detection-and-diagnosis)
    - [Techniques](#techniques)
      - [Hyperspectral imaging](#hyperspectral-imaging)
      - [Proton radiography](#proton-radiography)
    - [Brain](#brain-2)
    - [Breast](#breast-2)
    - [Cervical](#cervical)
    - [Esophageal](#esophageal-1)
    - [Liver](#liver)
    - [Lung](#lung-1)
    - [Mesothelioma](#mesothelioma)
    - [Nasopharyngeal](#nasopharyngeal)
    - [Oral](#oral-1)
    - [Ovarian](#ovarian)
    - [Pancreatic](#pancreatic-1)
    - [Prostate](#prostate-2)
    - [Skin](#skin)
    - [Stomach](#stomach)
  + [Prognostics](#prognostics)
    - [Breast](#breast-3)
    - [Colorectal](#colorectal)
    - [Liver](#liver-1)
    - [Pancreatic](#pancreatic-2)
    - [Prostate](#prostate-3)
  + [Treatment](#treatment)
    - [Drug design](#drug-design)
    - [Personalized chemotherapy](#personalized-chemotherapy)
    - [Radiation therapy](#radiation-therapy)
      - [Cervical](#cervical-1)
      - [Head and neck](#head-and-neck)
      - [Pelvic](#pelvic)
      - [Prostate](#prostate-4)
    - [Surgery](#surgery)
* [Presentations](#presentations)
* [Other and meta resources](#other-and-meta-resources)

## Code
* [Cancer](https://paperswithcode.com/task/cancer) @ Papers With Code
* [Google Colab](https://colab.research.google.com) - A powerful, free Jupyter notebook environment for researchers that runs on Tesla K80 with support for Keras, Tensorflow, PyTorch, and other neural network frameworks

### Challenges
* [CAMELYON17](https://camelyon17.grand-challenge.org) - Automated detection and classification of breast cancer metastases in whole-slide images of histological lymph node sections (2017)
* [Histopathologic Cancer Detection](https://www.kaggle.com/c/histopathologic-cancer-detection) (2018-2019)
* [ICIAR Grand Challenge on BreAst Cancer Histology images](https://iciar2018-challenge.grand-challenge.org) (2018)
* [Multimodal Brain Tumor Segmentation Challenge](https://www.med.upenn.edu/sbia/brats2018.html) - UPenn (2018)
* [Pancreatic Cancer Survival Prediction](http://miccai.cloudapp.net/competitions/84) - MICCAI (2018)
* [Screening and Diagnosis of esophageal cancer from in-vivo microscopy images](https://challengedata.ens.fr/challenges/11) - Mauna Kea Technologies (2019-2020)
* [Safe Scan](https://github.com/safe-scan-ai/cancer-ai) - Machine learning model competitions (Skin cancer) (2024)

### Repositories
* [zizhaozhang/nmi-wsi-diagnosis](https://github.com/zizhaozhang/nmi-wsi-diagnosis) - Pathologist-level interpretable whole-slide cancer diagnosis with deep learning
* [mark-watson/cancer-deep-learning-model](https://github.com/mark-watson/cancer-deep-learning-model) - Keras deep Learning neural network model for University of Wisconsin Cancer data that uses the Integrated Variants library to explain predictions made by a trained model with 97% accuracy
* [prasadseemakurthi/Deep-Neural-Networks-HealthCare](https://github.com/prasadseemakurthi/Deep-Neural-Networks-HealthCare/tree/master/Project%202%20--%20SurvNet%20--%20%20Cancer%20Clinical%20Outcomes%20Predictor) - Practical deep learning repository for predicting cancer clinical outcomes
* [MIC-DKFZ/nnunet](https://github.com/MIC-DKFZ/nnunet) - [[Paper](https://arxiv.org/abs/1904.08128)] - A framework designed for medical image segmentation that, when given a new dataset that includes training cases, will automatically take care of the entire experimental pipeline using a U-Net architecture
* [pfjaeger/medicaldetectiontoolkit](https://github.com/pfjaeger/medicaldetectiontoolkit) - [[Paper](https://arxiv.org/abs/1811.08661)] - Contains 2D + 3D implementations of prevalent object detectors such as Mask R-CNN, Retina Net, Retina U-Net, as well as a training and inference framework focused on dealing with medical images

#### Brain
* [jenspetersen/probabilistic-unet](https://github.com/jenspetersen/probabilistic-unet) - [[Paper](https://arxiv.org/abs/1907.04064v1)] - A PyTorch implementation of the Probabilistic U-Net, applied to probabilistic glioma growth
* [Surgical aid visualization system for glioblastoma tumor identification based on deep learning and in-vivo hyperspectral images of human patients](https://doi.org/10.1117/12.2512569) - Himar Fabelo, Martin Halicek, Samuel Ortega, Adam Szolna, Jesus Morera, Roberto Sarmiento, Gustavo M. Callico, Baowei Fei (2019)

#### Breast
* [AFAgarap/wisconsin-breast-cancer](https://github.com/AFAgarap/wisconsin-breast-cancer) - Codebase for *On Breast Cancer Detection: An Application of Machine Learning Algorithms on the Wisconsin Diagnostic Dataset [ICMLSC 2018 / arXiv 1711.07831]*
* [akshaybahadur21/BreastCancer_Classification](https://github.com/akshaybahadur21/BreastCancer_Classification) - Machine learning classifier for cancer tissues
* [AyeshaShafique/Breast-Cancer-Study](https://github.com/AyeshaShafique/Breast-Cancer-Study) - An experiment comparing the precision of MLP, XGBoost Classifier, Random Forest, Logistic Regression, SVM Linear Kernel, SVM RBF Kernel, KNN, and AdaBoost in classifying breast tumors
* [bhavaniprasad73/Artificial-Neural-Network](https://github.com/bhavaniprasad73/Artificial-Neural-Network) - [[Paper](https://github.com/bhavaniprasad73/Artificial-Neural-Network/blob/master/Final%20report.pdf) | [Slides](https://github.com/bhavaniprasad73/Artificial-Neural-Network/raw/master/CANCER%20PREDICTION%20USING%20ARTIFICIAL%20NEURAL%20NETWORKS.pptx)] - An artificial neural network (ANN) for the Wisconsin dataset
* [bhrzali/breast_cancer_classification](https://github.com/bhrzali/breast_cancer_classification) - A simple breast tumor classifier written in iPython and R
* [EliasVansteenkiste/dsb3](https://github.com/EliasVansteenkiste/dsb3) - A deep learning framework for lung cancer prediction built with Lasagne/Theano
* [gpreda/breast-cancer-prediction-from-cytopathology-data](https://www.kaggle.com/gpreda/breast-cancer-prediction-from-cytopathology-data) - Breast cancer prediction from cytopathology data in R Markdown (Kaggle)
* [Jean-njoroge/Breast-cancer-risk-prediction](https://github.com/Jean-njoroge/Breast-cancer-risk-prediction) - Classification of Breast Cancer diagnosis Using Support Vector Machines
* [kaas3000/exercise-breast-cancer](https://github.com/kaas3000/exercise-breast-cancer) - Python feed-forward neural network to predict breast cancer. Trained using stochastic gradient descent in combination with backpropagation
* [MahmudulHassan5809/BreastCancerPredictionPython3](https://github.com/MahmudulHassan5809/BreastCancerPredictionPython3) - Breast Cancer Prediction Using SVM in Python3
* [mathewyang/Wisconsin-Breast-Cancer](https://github.com/mathewyang/Wisconsin-Breast-Cancer) - Classifying benign and malignant breast tumor cells
* [millyleadley/breast-cancer-genes](https://github.com/millyleadley/breast-cancer-genes) - Predicting survival outcome in breast cancer patients based on their gene expression
* [MoganaD/Machine-Learning-on-Breast-Cancer-Survival-Prediction](https://github.com/MoganaD/Machine-Learning-on-Breast-Cancer-Survival-Prediction) - Model evaluation, Random Forest further modelling, variable importance, decision tree, and survival analysis in R
* [MrKhan0747/Breast-Cancer-Detection](https://github.com/MrKhan0747/Breast-Cancer-Detection) - Simple model for the Wisconsin dataset using logistic regression, support vector machine, KNN, naive Bayes, and random forest classifier
* [shibajyotidebbarma/Machine-Learning-with-Scikit-Learn-Breast-Cancer-Winconsin-Dataset](https://github.com/shibajyotidebbarma/Machine-Learning-with-Scikit-Learn-Breast-Cancer-Winconsin-Dataset) - A project comparing KNN, logistic regression, and neural network performance on the Wisconsin dataset
* [Shilpi75/Breast-Cancer-Prediction](https://github.com/Shilpi75/Breast-Cancer-Prediction) - Breast Cancer Prediction using fuzzy clustering and classification
* [spatri22/BenignMalignantClassification](https://github.com/spatri22/BenignMalignantClassification) - [[Paper](https://www.researchgate.net/publication/282954526_A_new_features_extraction_method_based_on_polynomial_regression_for_the_assessment_of_breast_lesion_Contours)] - Classification of breast lesion contours to benign and malignant categories
* [Surya-Prakash-Reddy/Breast-Cancer-Prediction](https://github.com/Surya-Prakash-Reddy/Breast-Cancer-Prediction) - ANN and SVN classifiers for the Wisconsin dataset written in Python
* [vkasojhaa/Clinical-Decision-Support-using-Machine-Learning](https://github.com/vkasojhaa/Clinical-Decision-Support-using-Machine-Learning) - Predicting the Stage of Breast Cancer - M (Malignant) and B (Benign) using different Machine learning models and comparing their performance
* [Wigder/inns](https://github.com/Wigder/inns) - An experiment using neural networks to predict obesity-related breast cancer over a small dataset of blood samples
* [YogiOnBioinformatics/Bioimaging-Informatics-Machine-Learning-for-Cancerous-Breast-Cell-Detection](https://github.com/YogiOnBioinformatics/Bioimaging-Informatics-Machine-Learning-for-Cancerous-Breast-Cell-Detection) - Evaluation of Automated Breast Density Segmentation Software on Segmenting Digital Mammograms Acquired at Magee-Womens Hospital

#### Esophageal
* [agalea91/esoph_linear_model](https://github.com/agalea91/esoph_linear_model) - [[Article](https://galeascience.wordpress.com/2016/05/05/linear-models-for-predicting-esophagus-cancer-likelihood/)] - Modeling of an esophageal cancer study in iPython
* [alexislechat/Screening-and-Diagnosis-of-esophageal-cancer-from-in-vivo-microscopy-images](https://github.com/alexislechat/Screening-and-Diagnosis-of-esophageal-cancer-from-in-vivo-microscopy-images) - A simple, multi-class image classifier for esophageal cancer
* [justinvyu/siemens-2017](https://github.com/justinvyu/siemens-2017) - Histopathology analysis for esophageal cancer and the automatic classification of tumors / cancer progression

#### Lung
* [ncoudray/DeepPATH](https://github.com/ncoudray/DeepPATH) - Classification of Lung cancer slide images using deep-learning
* [AiAiHealthcare/ProjectAiAi](https://github.com/AiAiHealthcare/ProjectAiAi) - A project with the slated goal of building an FDA-approved, open source deep learning system for detecting lung cancer

#### Oral
* [smg478/OralCancerDetectionOnCTImages](https://github.com/smg478/OralCancerDetectionOnCTImages) - [[Paper](https://arxiv.org/abs/1611.09769)] - C++ implementation of oral cancer detection on CT images with a neural network classifier
* [naveen3124/Oral-Cancer-Deep-Learning-](https://github.com/naveen3124/Oral-Cancer-Deep-Learning-) - CNN on oral cancer using holistic images

#### Prostate
* [bikramb98/Prostate-cancer-prediction](https://github.com/bikramb98/Prostate-cancer-prediction) - A simple prostate cancer prediction model built using KNN on a small dataset
* [eiriniar/gleason_CNN](https://github.com/eiriniar/gleason_CNN) - An attempt to reproduce the results of an earlier paper using a CNN and original TMA dataset
* [I2Cvb/prostate](https://github.com/I2Cvb/prostate) - Prostate cancer research repository from the Initiative for Collaborative Computer Vision Benchmarking
* [jameshorine/prostate_cancer](https://github.com/jameshorine/prostate_cancer) - A random forest model for predicting prostate cancer survival
* [Jin93/PCa-mpMRI](https://github.com/Jin93/PCa-mpMRI) - [[Paper](https://doi.org/10.1002/sim.7810)] - Four models for detecting prostate cancer with multi-parametric MRI utilizing anatomic structure
* [kkchau/NACEP-Analysis-of-Prostate-Cancer-Cells](https://github.com/kkchau/NACEP-Analysis-of-Prostate-Cancer-Cells) - NACEP Analysis of Tumorigenic prostate epithelial cells
* [kristiyanto/DREAM9.5](https://github.com/kristiyanto/DREAM9.5) - Predicting discontinuation of docetaxel treatment for metastatic castration-resistant prostate cancer (mCRPC)
* [ProstateCancer.ai](http://prostatecancer.ai) - An open-source web-application developed on the Tesseract-Medical Imaging platform designed to assist radiologists with a second opinion while simultaneously providing a standard image viewing and reporting scheme
* [rycoley/prediction-prostate-surveillance](https://github.com/rycoley/prediction-prostate-surveillance) - [[Paper](https://arxiv.org/pdf/1508.07511.pdf)] - Joint modeling of latent prostate cancer state and observed clinical outcomes for active surveillance of low risk prostate cancer
* [sahirbhatnagar/prostate](https://github.com/sahirbhatnagar/prostate) - Life expectancy tool for curative prostate cancer patients using Shiny app for R
* [swiri021/PAM50_Classifier_for_PRCA](https://github.com/swiri021/PAM50_Classifier_for_PRCA) - [[Paper](https://jamanetwork.com/journals/jamaoncology/article-abstract/2626510)] - PAM50 classifier for prostate cancer in Python based on Zhao el al on JAMA for the microarray data
* [Tesseract-MI/prostatecancer.ai](https://github.com/Tesseract-MI/prostatecancer.ai) - An OHIF-based, zero-footprint DICOMweb medical image viewer that utilizes AI to identify clinically significant prostate cancer
* [thom1178/Prostate-Cancer-Capsule](https://github.com/thom1178/Prostate-Cancer-Capsule) - [[Paper](https://github.com/thom1178/Prostate-Cancer-Capsule/blob/master/ProstateCancer.pdf)] - A logistic regression model to classify the capsule penetration of a tumor

### Datasets
* [The Cancer Genome Atlas Program](https://www.cancer.gov/about-nci/organization/ccg/research/structural-genomics/tcga) - Over 2.5 petabytes of genomic, epigenomic, transcriptomic, and proteomic data at Cancer.gov
* [Cancer Data Access System](https://biometry.nci.nih.gov/cdas/) @ National Cancer Institute
* [Genomics of Drug Sensitivity in Cancer](https://www.cancerrxgene.org) - 1000 human cancer cell lines characterized and screened with 100s of compounds
* [HarmonizomePythonScripts](https://github.com/MaayanLab/HarmonizomePythonScripts) - A collection of 116 datasets, 41 resources, and scripts that include many worthwhile in cancer research

#### Brain
* [Clinical, Genomic, Imaging Data & Specimens](https://cbttc.org/research/specimendata/) @ Children’s Brain Tumor Tissue Consortium

#### Breast
* [BACH dataset](https://iciar2018-challenge.grand-challenge.org/Dataset/) - Hematoxylin and eosin (H&E) stained breast histology microscopy and whole-slide images from the ICIAR 2018 Grand Challenge on BreAst Cancer Histology
* [Bioimaging Challenge 2015 Breast Histology Dataset](https://rdm.inesctec.pt/dataset/nis-2017-003) - Breast histology images from four classes: normal, benign, in situ carconima and invasive carcinoma
* [Breast Cancer Histopathological Image Classification Database (BreakHis)](https://web.inf.ufpr.br/vri/databases/breast-cancer-histopathological-database-breakhis/) - Laboratório Visão Robótica e Imagem, Prevenção & Diagnose Lab
* [Breast Cancer Wisconsin (Diagnostic) Data Set](https://www.kaggle.com/uciml/breast-cancer-wisconsin-data) - [[Kernels](https://www.kaggle.com/uciml/breast-cancer-wisconsin-data/kernels) | [Discussion](https://www.kaggle.com/uciml/breast-cancer-wisconsin-data/discussion)]
* [CAMELYON16 dataset](https://camelyon16.grand-challenge.org/Data/) - 400 whole-slide images (WSIs) of sentinel lymph node from two independent datasets collected in Radboud University Medical Center and University Medical Center Utrecht
* [CAMELYON17 dataset](https://camelyon17.grand-challenge.org/Data/) - 1000 whole-slide images (WSI) of hematoxylin and eosin (H&E) stained lymph node sections from patients at 5 hospitals in the Netherlands
* [MIAS Mammography](https://www.kaggle.com/kmader/mias-mammography)
* [MOBCdb a comprehensive database integrating multi-omics data of breast cancer](http://bigd.big.ac.cn/MOBCdb/) 

#### Pancreatic
* [PCMdb: Pancreatic Cancer Methylation Database](http://crdd.osdd.net/raghava/pcmdb/)

#### Prostate
* [Arvaniti prostate cancer TMA dataset](https://doi.org/10.7910/DVN/OCYCMP) - [[Paper](https://www.nature.com/articles/s41598-018-30535-1)] - H&E stained images from five prostate cancer Tissue Microarrays (TMAs) and corresponding Gleason annotation masks

## Papers 
* [Applications of Machine Learning in Cancer Prediction and Prognosis](https://www.ncbi.nlm.nih.gov/pmc/articles/PMC2675494/) - Joseph A. Cruz, David S. Wishart (2006)
* [Architectures and accuracy of artificial neural network for disease classification from omics data](https://www.ncbi.nlm.nih.gov/pmc/articles/PMC6399893/) - Hui Yu, David C. Samuels, Ying-yong Zhao, Yan Guo (2019)
* [Artificial intelligence in cancer imaging: Clinical challenges and applications](https://onlinelibrary.wiley.com/doi/full/10.3322/caac.21552) - Wenya Linda Bi, Ahmed Hosny, Matthew B. Schabath, Maryellen L. Giger, Nicolai J. Birkbak, Alireza Mehrtash, Tavis Allison, Omar Arnaout, Christopher Abbosh, Ian F. Dunn, Raymond H. Mak, Rulla M. Tamimi, Clare M. Tempany, Charles Swanton, Udo Hoffmann, Lawrence H. Schwartz, Robert J. Gillies, Raymond Y. Huang, Hugo J. W. L. Aerts (2019)

### Meta reviews
* [Artificial Intelligence in Pathology](https://www.ncbi.nlm.nih.gov/pmc/articles/PMC6344799/) - Hye Yoon Chang, Chan Kwon Jung, Junwoo Isaac Woo, Sanghun Lee, Joonyoung Cho, Sun Woo Kim, Tae-Yeong Kwak (2018)
* [Deep Learning in Medical Imaging](https://doi.org/10.14245/ns.1938396.198) - Mingyu Kim, Jihye Yun, Yongwon Cho, Keewon Shin, Ryoungwoo Jang, Hyun-jin Bae, and Namkug Kim (2019)

### Detection and diagnosis
* [Deep learning for tumor classification in imaging mass spectrometry](https://academic.oup.com/bioinformatics/article/34/7/1215/4604594) - Jens Behrmann, Christian Etmann, Tobias Boskamp, Rita Casadonte, Jörg Kriegsmann,  Peter Maaβ (2018)
* [Spatial Organization and Molecular Correlation of Tumor-Infiltrating Lymphocytes Using Deep Learning on Pathology Images](https://www.sciencedirect.com/science/article/pii/S2211124718304479) - JoelSaltz, Rajarsi Gupta, Le Hou, Tahsin Kurc, Pankaj Singh, Vu Nguyen, Dimitris Samaras, Kenneth R. Shroyer, Tianhao Zhao, Rebecca Batiste, John Van Arnam (2018)
* [Retina U-Net: Embarrassingly Simple Exploitation of Segmentation Supervision for Medical Object Detection](https://arxiv.org/pdf/1811.08661.pdf) - [[Code](https://github.com/pfjaeger/medicaldetectiontoolkit)] - Paul F. Jaeger, Simon A. A. Kohl, Sebastian Bickelhaupt, Fabian Isensee, Tristan Anselm Kuder, Heinz-Peter Schlemmer, Klaus H. Maier-Hein (2018)
* [A Probabilistic U-Net for Segmentation of Ambiguous Images](http://papers.nips.cc/paper/7928-a-probabilistic-u-net-for-segmentation-of-ambiguous-images) - [[Code 1](https://github.com/SimonKohl/probabilistic_unet) | [Code2](https://github.com/stefanknegt/Probabilistic-Unet-Pytorch) | [Video](https://youtu.be/-cfFxQWfFrA)] - Simon A. A. Kohl, Bernardino Romera-Paredes, Clemens Meyer, Jeffrey De Fauw, Joseph R. Ledsam, Klaus H. Maier-Hein, S. M. Ali Eslami, Danilo Jimenez Rezende, Olaf Ronneberger (2018)
* [A Hierarchical Probabilistic U-Net for Modeling Multi-Scale Ambiguities](https://arxiv.org/pdf/1905.13077.pdf) - Simon A. A. Kohl, Bernardino Romera-Paredes, Klaus H. Maier-Hein, Danilo Jimenez Rezende, S. M. Ali Eslami, Pushmeet Kohli, Andrew Zisserman, Olaf Ronneberger (2019)
* [nnU-Net: Breaking the Spell on Successful Medical Image Segmentation](https://arxiv.org/abs/1904.08128) - [[Code](https://github.com/MIC-DKFZ/nnunet)] - Fabian Isensee, Jens Petersen, Simon A. A. Kohl, Paul F. Jäger, Klaus H. Maier-Hein (2019)

#### Techniques
##### Hyperspectral imaging
* [Deep convolutional neural networks for classifying head and neck cancer using hyperspectral imaging](https://doi.org/10.1117/1.JBO.22.6.060503) - Martin Halicek, Guolan Lu, James V. Little, Xu Wang, Mihir Patel, Christopher C. Griffith, Mark W. El-Deiry, Amy Y. Chen, Baowei Fei (2017)
* [Deep Learning based Classification for Head and Neck Cancer Detection with Hyperspectral Imaging in an Animal Model](https://www.ncbi.nlm.nih.gov/pmc/articles/PMC6138463/) - Ling Ma, Guolan Lu, Dongsheng Wang, Xu Wang, Zhuo Georgia Chen, Susan Muller, Amy Chen, Baowei Fei (2018)
* [Optical Biopsy of Head and Neck Cancer Using Hyperspectral Imaging and Convolutional Neural Networks](https://www.ncbi.nlm.nih.gov/pmc/articles/PMC6123819/) - Martin Halicek, James V. Little, Xu Wang, Mihir Patel, Christopher C. Griffith, Mark W. El-Deiry, Amy Y. Chen, Baowei Fei (2018)
* [Tumor margin classification of head and neck cancer using hyperspectral imaging and convolutional neural networks](https://www.ncbi.nlm.nih.gov/pmc/articles/PMC6149520/) - Martin Halicek, James V. Little, Xu Wang, Mihir Patel, Christopher C. Griffith, Amy Y. Chen, Baowei Fei (2018)
* [In-Vivo and Ex-Vivo Tissue Analysis through Hyperspectral Imaging Techniques: Revealing the Invisible Features of Cancer](https://doi.org/10.3390/cancers11060756) - Martin Halicek, Himar Fabelo, Samuel Ortega, Gustavo M. Callico, Baowei Fei (2019)
* [Surgical aid visualization system for glioblastoma tumor identification based on deep learning and in-vivo hyperspectral images of human patients](https://doi.org/10.1117/12.2512569) - Himar Fabelo, Martin Halicek, Samuel Ortega, Adam Szolna, Jesus Morera, Roberto Sarmiento, Gustavo M. Callico, Baowei Fei (2019)

##### Proton radiography
* [High quality proton portal imaging using deep learning for proton radiation therapy: a phantom study](https://doi.org/10.1088/2057-1976/ab8a74) - Serdar Charyyev, Yang Lei, Joseph Harms, Bree Eaton, Mark McDonald, Walter J. Curran, Tian Liu, Jun Zhou, Rongxiao Zhang, Xiaofeng Yang (2020)

#### Brain
* [Deep Probabilistic Modeling of Glioma Growth](https://arxiv.org/abs/1907.04064v1) - [[Code](https://github.com/jenspetersen/probabilistic-unet)] - Jens Petersen, Paul F. Jäger, Fabian Isensee, Simon A. A. Kohl, Ulf Neuberger, Wolfgang Wick, Jürgen Debus, Sabine Heiland, Martin Bendszus, Philipp Kickingereder, Klaus H. Maier-Hein (2019)

#### Breast
* [A new features extraction method based on polynomial regression for the assessment of breast lesion Contours](https://www.researchgate.net/publication/282954526_A_new_features_extraction_method_based_on_polynomial_regression_for_the_assessment_of_breast_lesion_Contours) - Spandana Paramkusham, Shivakshit Patri, K.M.M. Rao, B.V.V.S.N.P. Rao (2015)
* [Diagnostic Assessment of Deep Learning Algorithms for Detection of Lymph Node Metastases in Women With Breast Cancer](https://jamanetwork.com/journals/jama/article-abstract/2665774) - Babak Ehteshami Bejnordi, Mitko Veta, Paul Johannes van Diest (2017)
* [Machine Learning with Applications in Breast Cancer Diagnosis and Prognosis](https://www.researchgate.net/publication/325064884_Machine_Learning_with_Applications_in_Breast_Cancer_Diagnosis_and_Prognosis) - Wenbin Yue, Zidong Wang, Hongwei Chen, Annette Margaret Payne, Xiaohui Liu (2018)
* [Learning a low dimensional manifold of real cancer tissue with PathologyGAN](https://arxiv.org/abs/2004.06517) - Adalberto Claudio Quiros, Roderick Murray-Smith, Ke Yuan (2020)
* [RDA-UNET-WGAN: An Accurate Breast Ultrasound Lesion Segmentation Using Wasserstein Generative Adversarial Networks](https://doi.org/10.1007/s13369-020-04480-z) - Anuja Negi, Alex Noel Joseph Raj, Ruban Nersisson, Zhemin Zhuang, M. Murugappan (2020)

#### Cervical
* [Intelligent Inverse Treatment Planning via Deep Reinforcement Learning, a Proof-of-Principle Study in High Dose-rate Brachytherapy for Cervical Cancer](https://arxiv.org/pdf/1811.10102.pdf) - Chenyang Shen, Yesenia Gonzalez, Peter Klages, Nan Qin, Hyunuk Jung, Liyuan Chen, Dan Nguyen, Steve B. Jiang, Xun Jia (2019)

#### Esophageal
* [Artificial Neural Networks and Gene Filtering Distinguish Between Global Gene Expression Profiles of Barrett’s Esophagus and Esophageal Cancer](http://cancerres.aacrjournals.org/content/62/12/3493) - Yan Xu, Florin M. Selaru, Jing Yin, Tong Tong Zou, Valentina Shustova, Yuriko Mori, Fumiaki Sato, Thomas C. Liu, Andreea Olaru, Suna Wang, Martha C. Kimos, Kellie Perry, Kena Desai, Bruce D. Greenwald, Mark J. Krasna, David Shibata, John M. Abraham, Stephen J. Meltzer (2002)
* [Convolutional neural network classifier for distinguishing Barrett's esophagus and neoplasia endomicroscopy images](https://ieeexplore.ieee.org/document/8037461) - Jisu Hong, Bo-yong Park, Hyunjin Park (2017)
* [Esophagus segmentation in CT via 3D fully convolutional neural network and random walk](https://aapm.onlinelibrary.wiley.com/doi/pdf/10.1002/mp.12593) - Tobias Fechter, Sonja Adebahr, Dimos Baltas, Ismail Ben Ayed, Christian Desrosiers, Jose Doiz (2017)
* [Interpretable Fully Convolutional Classification of Intrapapillary Capillary Loops for Real-Time Detection of Early Squamous Neoplasia](https://arxiv.org/pdf/1805.00632.pdf) - Luis C. Garcia-Peraza-Herrera, Martin Everson, Wenqi Li, Inmanol Luengo, Lorenz Berger, Omer Ahmad, Laurence Lovat, Hsiu-Po Wang, Wen-Lun Wang, Rehan Haidry, Danail Stoyanov, Tom Vercauteren, Sebastien Ourselin (2018)
* [Diagnostic outcomes of esophageal cancer by artificial intelligence using convolutional neural networks](https://www.sciencedirect.com/science/article/pii/S0016510718329262) - Yoshimasa Horie, Toshiyuki Yoshio, Kazuharu Aoyama, Shoichi Yoshimizu, Yusuke Horiuchi, Akiyoshi Ishiyama, Toshiaki Hirasawa, Tomohiro Tsuchida, Tsuyoshi Ozawa, Soichiro Ishihara, Youichi Kumagai, Mitsuhiro Fujishiro, Iruru Maetani, Junko Fujisaki, Tomohiro Tada (2018)
* [Early esophageal adenocarcinoma detection using deep learning methods](https://link.springer.com/article/10.1007/s11548-019-01914-4) - Noha Ghatwar, Massoud Zolgharni, Xujiong Ye (2019)

#### Liver
* [Deep learning based classification of focal liver lesions with contrast-enhanced ultrasound](https://iciar2018-challenge.grand-challenge.org/media/evaluation-supplementary/176/16918/dc233d3b-ffe1-49c7-bacd-17beb04c10f6/Deep_learning_ba_w8Kpz96.pdf) - Kaizhi Wu, Xi Chen, Mingyue Ding (2014)
* [Classification of focal liver lesions on ultrasound images by extracting hybrid textural features and using an artificial neural network](https://pdfs.semanticscholar.org/cba4/9defe542f8208ea59bb91280dab93664369c.pdf) - Yoo Na Hwanga, Ju Hwan Leeb, Ga Young Kimb, Yuan Yuan Jiangb,  Sung Min Kim (2015)
* [Deep learning based liver cancer detection using watershed transform and Gaussian mixture model techniques](https://www.researchgate.net/publication/329772767_Deep_learning_based_liver_cancer_detection_using_watershed_transform_and_Gaussian_mixture_model_techniques) - Sukanta Sabut, Amita Das, U Rajendra Acharya, Soumya S. Panda (2018)
* [Automatic liver tumor segmentation in CT with fully convolutional neural networks and object-based postprocessing](https://www.nature.com/articles/s41598-018-33860-7) - Grzegorz Chlebus, Andrea Schenk, Jan Hendrik Moltz, Bram van Ginneken, Horst Karl Hahn, Hans Meine (2018)
* [Deep learning for liver tumor diagnosis part I: development of a convolutional neural network classifier for multi-phasic MRI](https://www.ncbi.nlm.nih.gov/pubmed/31016442) - Charlie A. Hamm, Clinton J. Wang, Lynn Jeanette Savic, Marc Ferrante, Isabel Schobert, Todd Schlachter, MingDe Lin, James S. Duncan, Jeffrey C. Weinreb, Julius Chapiro, Brian Letzen (2019)
* [Deep learning for liver tumor diagnosis part II: convolutional neural network interpretation using radiologic imaging features](https://www.ncbi.nlm.nih.gov/pubmed/31093705) - Clinton J. Wang, Charlie A. Hamm, Lynn Jeanette Savic, Marc Ferrante, Isabel Schobert, Todd Schlachter, MingDe Lin, Jeffrey C. Weinreb, James S. Duncan, Julius Chapiro, Brian Letzen (2019)
* [Deep Learning for Automated Segmentation of Liver Lesions at CT in Patients with Colorectal Cancer Liver Metastases](https://pubs.rsna.org/doi/abs/10.1148/ryai.2019180014) - Eugene Vorontsov, Milena Cerny, Philippe Régnier, Lisa Di Jorio, Christopher J. Pal, Réal Lapointe, Franck Vandenbroucke-Menu, Simon Turcotte, Samuel Kadoury, An Tang (2019)
* [Dynamic contrast-enhanced computed tomography diagnosis of primary liver cancers using transfer learning of pretrained convolutional neural networks: Is registration of multiphasic images necessary?](https://www.researchgate.net/publication/332855380_Dynamic_contrast-enhanced_computed_tomography_diagnosis_of_primary_liver_cancers_using_transfer_learning_of_pretrained_convolutional_neural_networks_Is_registration_of_multiphasic_images_necessary) - Akira Yamada, Kazuki Oyama, Sachie Fujita, Eriko Yoshizawa, Fumihito Ichinohe, Daisuke Komatsu, Yasunari Fujinaga (2019)
* [A probabilistic approach for interpretable deep learning in liver cancer diagnosis](https://www.spiedigitallibrary.org/conference-proceedings-of-spie/10950/109500U/A-probabilistic-approach-for-interpretable-deep-learning-in-liver-cancer/10.1117/12.2512473.short) - Clinton J. Wang, Charlie A. Hamm, Brian S. Letzen, James S. Duncan (2019)
* [A Joint Deep Learning Approach for Automated Liver and Tumor Segmentation](https://arxiv.org/abs/1902.07971) - Nadja Gruber, Stephan Antholzer, Werner Jaschke, Christian Kremser, Markus Haltmeier (2019)
* [Tripartite-GAN: Synthesizing Liver Contrast-Enhanced MRI to Improve Tumor Detection](https://www.researchgate.net/publication/340853012_Tripartite-GAN_Synthesizing_Liver_Contrast-Enhanced_MRI_to_Improve_Tumor_Detection) - Jianfeng Zhao, Dengwang Li, Zahra Kassam, Joanne Howey, Jaron Chong, Bo Chen, Shuo Li (2020)

#### Lung
* [Combining deep learning and coherent anti-Stokes Raman scattering imaging for automated differential diagnosis of lung cancer](https://www.researchgate.net/publication/320723321_Combining_deep_learning_and_coherent_anti-Stokes_Raman_scattering_imaging_for_automated_differential_diagnosis_of_lung_cancer) - Sheng Weng, Xiaoyun Xu, Jiasong Li, Stephen TC Wong (2017)
* [Deep reinforcement learning for automated radiation adaptation in lung cancer](https://doi.org/10.1002/mp.12625) - Huan‐Hsin Tseng, Yi Luo, Sunan Cui, Jen‐Tzung Chien, Randall K. Ten Haken, Issam El Naqa (2017)
* [Lung Nodule Detection via Deep Reinforcement Learning](https://www.frontiersin.org/articles/10.3389/fonc.2018.00108/full) - Issa Ali, Gregory R. Hart, Gowthaman Gunabushanam, Ying Liang, Wazir Muhammad, Bradley Nartowt, Michael Kane, Xiaomei Ma, Jun Deng (2018)
* [Convolutional Neural Networks Promising in Lung Cancer T-Parameter Assessment on Baseline FDG-PET/CT](https://doi.org/10.1155/2018/1382309) - Margarita Kirienko, Martina Sollini, Giorgia Silvestri, Serena Mognetti, Emanuele Voulaz, Lidija Antunovic, Alexia Rossi, Luca Antiga, Arturo Chiti (2018)
* [Deep reinforcement learning with its application for lung cancer detection in medical Internet of Things](https://www.sciencedirect.com/science/article/pii/S0167739X19303772) - Zhuo Liua, Chenhui Yaob, Hang Yuc, Taihua Wu (2019)
* [Automated Pulmonary Nodule Classification in Computed Tomography Images Using a Deep Convolutional Neural Network Trained by Generative Adversarial Networks](https://www.ncbi.nlm.nih.gov/pmc/articles/PMC6334309/) - Yuya Onishi, Atsushi Teramoto, Masakazu Tsujimoto, Tetsuya Tsukamoto, Kuniaki Saito, Hiroshi Toyama, Kazuyoshi Imaizumi, Hiroshi Fujita (2019)
* [Unsupervised Multi-Discriminator Generative Adversarial Network for Lung Nodule Malignancy Classification](https://doi.org/10.1109/ACCESS.2020.2987961) - Yan Kuang, Tian Lan, Xueqiao Peng, Gati Elvis Selashi, Qiao Liu, Junyi Zhang (2020)

#### Mesothelioma
* [Development and validation of a deep learning model using biomarkers in pleural effusion for prediction of malignant pleural mesothelioma](https://doi.org/10.1093/annonc/mdy301.003) - S. Ikegaki, Y. Kataoka, T. Otoshi, T. Takemura, M. Shimada (2018)
* [Diagnosis of mesothelioma with deep learning](https://www.ncbi.nlm.nih.gov/pmc/articles/PMC6341823/) - Xue Hu, Zebo Yu (2019)

#### Nasopharyngeal
* [Deep Deconvolutional Neural Network for Target Segmentation of Nasopharyngeal Cancer in Planning Computed Tomography Images](https://www.frontiersin.org/articles/10.3389/fonc.2017.00315/full) - Kuo Men, Xinyuan Chen, Ye Zhang, Tao Zhang, Jianrong Dai, Junlin Yi, Yexiong Li (2017)
* [Development and validation of an endoscopic images-based deep learning model for detection with nasopharyngeal malignancies](https://cancercommun.biomedcentral.com/articles/10.1186/s40880-018-0325-9) - Chaofeng Li, Bingzhong Jing, Liangru Ke, Bin Li, Weixiong Xia, Caisheng He, Chaonan Qian, Chong Zhao, Haiqiang Mai, Mingyuan Chen, Kajia Cao, Haoyuan Mo, Ling Guo, Qiuyan Chen, Linquan Tang, Wenze Qiu, Yahui Yu, Hu Liang, Xinjun Huang, Guoying Liu, Wangzhong Li, Lin Wang, Rui Sun, Xiong Zou, Shanshan Guo, Peiyu Huang, Donghua Luo, Fang Qiu, Yishan Wu, Yijun Hua, Kuiyuan Liu, Shuhui Lv, Jingjing Miao, Yanqun Xiang, Ying Sun, Xiang Guo, Xing Lv (2018)

#### Oral
* [Computer Aided Detection of Oral Lesions on CT Images](https://arxiv.org/abs/1611.09769) - [[Code](https://github.com/smg478/OralCancerDetectionOnCTImages)] - Shaikat Galib, Fahima Islam, Muhammad Abir, Hyoung-Koo Lee (2016)

#### Ovarian
* [Automatic classification of ovarian cancer types from cytological images using deep convolutional neural networks](https://doi.org/10.1042/BSR20180289) - Miao Wu, Chuanbo Yan, Huiqiang Liu, Qian Liu (2018)

#### Pancreatic
* [DeepOrgan: Multi-level Deep Convolutional Networks for Automated Pancreas Segmentation](https://link.springer.com/chapter/10.1007/978-3-319-24553-9_68) - Holger R. Roth, Le Lu, Amal Farag, Hoo-Chang Shin, Jiamin Liu, Evrim B. Turkbey, Ronald M. Summers (2015)
* [Deep convolutional networks for pancreas segmentation in CT imaging](https://www.spiedigitallibrary.org/conference-proceedings-of-spie/9413/94131G/Deep-convolutional-networks-for-pancreas-segmentation-in-CT-imaging/10.1117/12.2081420.short?SSO=1) - Holger R. Roth, Amal Farag, Le Lu, Evrim B. Turkbey, Ronald M. Summers (2015)
* [Pancreas Segmentation in Abdominal CT Scan: A Coarse-to-Fine Approach](https://pdfs.semanticscholar.org/788f/341d02130e1807edf88c8c64a77e4096437e.pdf) - Yuyin Zhou, Lingxi Xie, Wei Shen, Elliot Fishman, Alan Yuille (2016)
* [Combining Machine Learning and Nanofluidic Technology To Diagnose Pancreatic Cancer Using Exosomes](https://pubs.acs.org/doi/abs/10.1021/acsnano.7b05503) - Jina Ko, Neha Bhagwat, Stephanie S. Yee, Natalia Ortiz, Amine Sahmoud, Taylor Black, Nicole M. Aiello, Lydie McKenzie, Mark O’Hara, Colleen Redlinger, Janae Romeo, Erica L. Carpenter, Ben Z. Stanger, David Issadore (2017)
* [Deep learning based Nucleus Classification in pancreas histological images](https://ieeexplore.ieee.org/abstract/document/8036914) - Young Hwan Chang, Guillaume Thibault, Owen Madin, Vahid Azimi, Cole Meyers, Brett Johnson (2017)
* [Deep Supervision for Pancreatic Cyst Segmentation in Abdominal CT Scans](https://link.springer.com/chapter/10.1007/978-3-319-66179-7_26) - Yuyin Zhou, Lingxi Xie, Elliot K. Fishman, Alan L. Yuille (2017)
* [Deep LOGISMOS: Deep learning graph-based 3D segmentation of pancreatic tumors on CT scans](https://ieeexplore.ieee.org/abstract/document/8363793/) - Zhihui Guo, Ling Zhang, Le Lu, Mohammadhadi Bagheri, Ronald M. Summers, Milan Sonka, Jianhua Yao (2018)
* [Development of a prediction model for pancreatic cancer in patients with type 2 diabetes using logistic regression and artificial neural network models](https://doi.org/10.2147/CMAR.S180791) - Meng Hsuen Hsieh, Li-Min Sun, Cheng-Li Lin, Meng-Ju Hsieh, Chung-Y Hsu, Chia-Hung Kao (2018)
* [Deep learning convolutional neural network (CNN) With Gaussian mixture model for predicting pancreatic cancer](https://link.springer.com/article/10.1007%2Fs11042-019-7419-5) - Kaushik Sekaran, P. Chandana, N. Murali Krishna, Seifedine Kadry (2019)
* [Personalized Pancreatic Cancer Management](https://doi.org/10.1097/MPA.0000000000001312) - Alison Bradley, Robert van der Meer, Colin McKay (2019)
* [Establishment and application of an artificial intelligence diagnosis system for pancreatic cancer with a faster region-based convolutional neural network](https://mednexus.org/doi/full/10.1097/CM9.0000000000000544) - Shang-Long Liu, Shuo Li, Yu-Ting Guo, Yun-Peng Zhou, Zheng-Dong Zhang, Shuai Li, Yun Lu (2019)
* [Response to repeat echoendoscopic celiac plexus neurolysis in pancreatic cancer patients: A machine learning approach](https://doi.org/10.1016/j.pan.2019.07.038) - Antonio Facciorusso, Valentina Del Prete, Matteo Antonino, Vincenzo Rosario, Buccino Nicola, Muscatiello (2019)
* [Construction of a convolutional neural network classifier developed by computed tomography images for pancreatic cancer diagnosis](https://doi.org/10.3748/wjg.v26.i34.5156) - Han Ma, Zhong-Xin Liu, Jing-Jing Zhang, Feng-Tian Wu, Cheng-Fu Xu, Zhe Shen, Chao-Hui Yu, You-Ming Li (2019)
* [Establishment and clinical testing of pancreatic cancer Faster R-CNN AI system based on fast regional convolutional neural network](https://doi.org/10.3760/cma.j.cn112139-20191017-00515) - S. J. Yang, Y. Lu, X. F. Zheng, Y. J. Zhang, F. J. Xin, P. Sun, Y. Li, S. S. Liu, S. Li, Y. T. Guo, S. L. Liu (2020)
* [Convolutional neural network for the detection of pancreatic cancer on CT scans](https://doi.org/10.1016/S2589-7500(20)30190-4) - Garima Suman, Ananya Panda, Panagiotis Korfiatis, Ajit H Goenka (2020)
* [Pancreatic cancer tumor analysis in CT images using patch-based multi-resolution convolutional neural network](https://doi.org/10.1016/j.bspc.2021.102652) - Vahid Asadpour, Rex A.Parker, Patrick R. Mayock, Samuel E.Sampson, Wansu Chen, Bechien Wu (2021)
* [Detection of pancreatic cancer by convolutional-neural-network-assisted spontaneous Raman spectroscopy with critical feature visualization](https://doi.org/10.1016/j.neunet.2021.09.006) - Zhongqiang Li, Zheng Li, Qing Chen, Alexandra Ramos, Jian Zhang, J. Philip Boudreaux, Ramcharan Thiagarajan, Yvette Bren-Mattison, Michael E.Dunham, Andrew J. McWhorter. Xin Li. Ji-Ming Feng. Yanping Li, Shaomian Yao, Jian Xu (2021)
* [Fully Automatic Deep Learning Framework for Pancreatic Ductal Adenocarcinoma Detection on Computed Tomography](https://doi.org/10.3390/cancers14020376) - Natália Alves, Megan Schuurmans, Geke Litjens, Joeran S Bosma, John Hermans, Henkjan Huisman (2022)

#### Prostate
* [A Bayesian Hierarchical Model for Prediction of Latent Health States from Multiple Data Sources with Application to Active Surveillance of Prostate Cancer](https://arxiv.org/abs/1508.07511) - [[Code](https://github.com/rycoley/prediction-prostate-surveillance)] - R. Yates Coley, Aaron J. Fisher, Mufaddal Mamawala, H. Ballentine Carter, Kenneth J. Pienta, Scott L. Zeger (2016)
* [Adversarial Networks for the Detection of Aggressive Prostate Cancer](https://arxiv.org/abs/1702.08014) - Simon Kohl, David Bonekamp, Heinz-Peter Schlemmer, Kaneschka Yaqubi, Markus Hohenfellner, Boris Hadaschik, Jan-Philipp Radtke, Klaus Maier-Hein (2017)
* [Adversarial Networks for Prostate Cancer Detection](https://www.researchgate.net/publication/321347307_Adversarial_Networks_for_Prostate_Cancer_Detection) - Simon Kohl, David Bonekamp, Heinz-Peter Schlemmer, Kaneschka Yaqubi, Markus Hohenfellner, Boris Hadaschik, Jan-Philipp Radtke, Klaus Maier-Hein (2017)
* [Radiomic Machine Learning for Characterization of Prostate Lesions with MRI: Comparison to ADC Values](https://pubs.rsna.org/doi/abs/10.1148/radiol.2018173064) - [[Editorial](https://pubs.rsna.org/doi/10.1148/radiol.2018181304)] - David Bonekamp, Simon Kohl, Manuel Wiesenfarth, Patrick Schelb, Jan Philipp Radtke, Michael Götz, Philipp Kickingereder, Kaneschka Yaqubi, Bertram Hitthaler, Nils Gählert, Tristan Anselm Kuder, Fenja Deister, Martin Freitag, Markus Hohenfellner, Boris A. Hadaschik, Heinz-Peter Schlemmer, Klaus H. Maier-Hein (2018)
* [Automated Gleason grading of prostate cancer tissue microarrays via deep learning](https://www.nature.com/articles/s41598-018-30535-1) - [[Dataset](https://doi.org/10.7910/DVN/OCYCMP)] - Eirini Arvaniti, Kim S. Fricker, Michael Moret, Niels Rupp, Thomas Hermanns, Christian Fankhauser, Norbert Wey, Peter J. Wild, Jan H. Rüschoff, Manfred Claassen (2018)
* [Computational histological staining and destaining of prostate core biopsy RGB images with generative adversarial neural networks](https://www.media.mit.edu/publications/computational-histological-staining-and-destaining-of-prostate-core-biopsy-rgb-images-with-generative-adversarial-neural-networks/) - Aman Rana, Gregory Yauney, Alarice Lowe, Pratik Shah (2018)
* [Comparison of Artificial Intelligence Techniques to Evaluate Performance of a Classifier for Automatic Grading of Prostate Cancer From Digitized Histopathologic Images](https://jamanetwork.com/journals/jamanetworkopen/fullarticle/2727273) - Guy Nir, Davood Karimi, S. Larry Goldenberg, Ladan Fazli, Brian F. Skinnider, Peyman Tavassoli, Dmitry Turbin, Carlos F. Villamil, Gang Wang, Darby J. S. Thompson, Peter C. Black, Septimiu E. Salcudean (2019)
* [Improving Prostate Cancer Detection with Breast Histopathology Images](https://arxiv.org/abs/1903.05769) - Umair Akhtar Hasan Khan, Carolin Stürenberg, Oguzhan Gencoglu, Kevin Sandeman, Timo Heikkinen, Antti Rannikko, Tuomas Mirtti (2019)
* [Improving Prostate Cancer (PCa) Classification Performance by Using Three-Player Minimax Game to Reduce Data Source Heterogeneity](https://doi.org/10.0.4.85/TMI.2020.2988198) - Yanan Shao, Jane Wang, Brian Wodlinger, Septimiu E. Salcudean (2020)

#### Skin
* [Generating Synthetic Medical Images by Using GAN to Improve CNN Performance in Skin Cancer Classification](https://doi.org/10.1109/ICRoM48714.2019.9071823) - Pooyan Sedigh, Rasoul Sadeghian, Mehdi Tale Masouleh (2019)

#### Stomach
* [Deep convolutional neural networks for automatic classification of gastric carcinoma using whole slide images in digital histopathology](https://www.ncbi.nlm.nih.gov/pubmed/28676295) - Harshita Sharma, Norman Zerbe, Iris Klempert, Olaf Hellwich, Peter Hufnag (2017)
* [Detecting gastric cancer from video images using convolutional neural networks](https://doi.org/10.1111/den.13306) - Mitsuaki Ishioka, Toshiaki Hirasawa, Tomohiro Tada (2018)
* [Whole Slide Image Classification of Gastric Cancer using Convolutional Neural Networks](https://www.semanticscholar.org/paper/Whole-Slide-Image-Classification-of-Gastric-Cancer-Shou-Li/426b6516a0b4c959b9380fd8d5b9bf1942778f02) - Junni Shou, Yan Li, Guanzhen Yu, Guannan Li (2018)
* [Accurate Gastric Cancer Segmentation in Digital Pathology Images Using Deformable Convolution and Multi-Scale Embedding Networks](https://ieeexplore.ieee.org/document/8721664) - Muyi Sun, Guanhong Zhang, Hao Dang, Xingqun Qi, Xiaoguang Zhou, Qing Chang (2019)

### Prognostics
* [Machine learning applications in cancer prognosis and prediction](https://www.sciencedirect.com/science/article/pii/S2001037014000464) - Konstantina Kourou, Themis P. Exarchos, Konstantinos P. Exarchos, Michalis V. Karamouzis, Dimitrios I. Fotiadis (2015)
* [Predicting clinical outcomes from large scale cancer genomic profiles with deep survival models](https://www.ncbi.nlm.nih.gov/pmc/articles/PMC5601479/) - Safoora Yousefi, Fatemeh Amrollahi, Mohamed Amgad, Coco Dong, Joshua E. Lewis, Congzheng Song, David A Gutman, Sameer H. Halani, Jose Enrique Velazquez Vega, Daniel J Brat, Lee AD Cooper (2017)

#### Breast
* [Machine Learning with Applications in Breast Cancer Diagnosis and Prognosis](https://www.researchgate.net/publication/325064884_Machine_Learning_with_Applications_in_Breast_Cancer_Diagnosis_and_Prognosis) - Wenbin Yue, Zidong Wang, Hongwei Chen, Annette Margaret Payne, Xiaohui Liu (2018)
* [On Breast Cancer Detection: An Application of Machine Learning Algorithms on the Wisconsin Diagnostic Dataset](https://arxiv.org/abs/1711.07831) - Abien Fred M. Agarap (2019)
* [Application of Machine Learning Models for Survival Prognosis in Breast Cancer Studies](https://doi.org/10.3390/info10030093) - Iliyan Mihaylov, Maria Nisheva, Dimitar Vassilev (2019)
* [Breast Cancer Prognosis Using a Machine Learning Approach](https://www.researchgate.net/publication/331619628_Breast_Cancer_Prognosis_Using_a_Machine_Learning_Approach) - Patrizia Ferroni, Fabio Massimo Zanzotto, Silvia Riondino, Noemi Scarpato, Fiorella Guadagni, Mario Roselli (2019)
* [Predicting factors for survival of breast cancer patients using machine learning techniques](https://link.springer.com/article/10.1186/s12911-019-0801-4) - Mogana Darshini Ganggayah, Nur Aishah Taib, Yip Cheng Har, Pietro Lio, Sarinder Kaur Dhillon (2019) 

#### Colorectal 
* [Predicting survival from colorectal cancer histology slides using deep learning: A retrospective multicenter study](https://journals.plos.org/plosmedicine/article?id=10.1371/journal.pmed.1002730) - Jakob Nikolas Kather, Johannes Krisam, Pornpimol Charoentong, Tom Luedde, Esther Herpel, Cleo-Aron Weis, Timo Gaiser, Alexander Marx, Nektarios A. Valous, Dyke Ferber, Lina Jansen, Constantino Carlos Reyes-Aldasoro, Inka Zörnig, Dirk Jäger, Hermann Brenner, Jenny Chang-Claude, Michael Hoffmeister, Niels Halama (2019)

#### Liver
* [Deep Learning–Based Multi-Omics Integration Robustly Predicts Survival in Liver Cancer](http://clincancerres.aacrjournals.org/content/24/6/1248) - Kumardeep Chaudhary, Olivier B. Poirion, Liangqun Lu, Lana X. Garmire (2018)

#### Pancreatic
* [An Improved Method for Prediction of Cancer Prognosis by Network Learning](https://www.ncbi.nlm.nih.gov/pmc/articles/PMC6210393/) - Minseon Kim, Ilhwan Oh, Jaegyoon Ahn (2018)
* [Development, Validation and Comparison of Artificial Neural Network Models and Logistic Regression Models Predicting Survival of Unresectable Pancreatic Cancer](https://doi.org/10.3389/fbioe.2020.00196) - Zhou Tong, Yu Liu, Hongtao Ma, Jindi Zhang, Bo Lin, Xuanwen Bao, Xiaoting Xu, Changhao Gu, Yi Zheng, Lulu Liu, Weijia Fang, Shuiguang Deng, Peng Zhao (2020)
* [Pancreatic Cancer Survival Prediction: A Survey of the State-of-the-Art](https://doi.org/10.1155/2021/1188414) - Wilson Bakasa, Serestina Viriri, Huiling Chen (2021)


#### Prostate
* [Prognosis of prostate cancer by artificial neural networks](https://www.sciencedirect.com/science/article/pii/S095741741000237X) - Ismail Saritas, Ilker Ali Ozkan, Ibrahim Unal Sert (2010)

### Treatment
#### Drug design
* [Multicellular Target QSAR Model for Simultaneous Prediction and Design of Anti-Pancreatic Cancer Agents](https://pubs.acs.org/doi/10.1021/acsomega.8b03693) - Alejandro Speck-Planche (2019)

#### Personalized chemotherapy
* [Reinforcement learning design for cancer clinical trials](https://www.researchgate.net/publication/26808581_Reinforcement_learning_design_for_cancer_clinical_trials) - Yufan Zhao, Michael R. Kosorok, Donglin Zeng (2009)
* [Personalized Cancer Chemotherapy Schedule: a numerical comparison of performance and robustness in model-based and model-free scheduling methodologies](https://arxiv.org/abs/1904.01200) - Jesus Tordesillas, Juncal Arbelaiz (2019)

#### Radiation therapy
* [Automated Treatment Planning in Radiation Therapy using Generative Adversarial Networks](https://arxiv.org/pdf/1807.06489.pdf) - Rafid Mahmood, Aarron Babier, Andrea McNiven, Adam Diamant, Timothy C. Y. Chan (2018)
* [Machine Learning in Radiation Oncology: Opportunities, Requirements, and Needs](https://doi.org/10.3389/fonc.2018.00110) - Mary Feng, Gilmer Valdes, Nayha Dixit, Timothy D. Solberg (2018)
* [Applications and limitations of machine learning in radiation oncology](https://doi.org/10.1259/bjr.20190001) - Daniel Jarrett, Eleanor Stride, Katherine Vallis, Mark J. Gooding (2019)
* [Machine Learning with Radiation Oncology Big Data](https://doi.org/10.3389/978-2-88945-730-4) - Jun Deng, Issam El Naqa, Lei Xing (2019)

##### Cervical
* [Intelligent Inverse Treatment Planning via Deep Reinforcement Learning, a Proof-of-Principle Study in High Dose-rate Brachytherapy for Cervical Cancer](https://www.researchgate.net/publication/329206758_Intelligent_Inverse_Treatment_Planning_via_Deep_Reinforcement_Learning_a_Proof-of-Principle_Study_in_High_Dose-rate_Brachytherapy_for_Cervical_Cancer) - Chenyang Shen, Yesenia Gonzalez, Peter Klages, Nan Qin, Hyunuk Jung, Liyuan Chen, Dan Nguyen, Steve B. Jiang, Xun Jia (2018)

##### Head and neck
* [Development of a Novel Deep Learning Algorithm for Autosegmentation of Clinical Tumor Volume and Organs at Risk in Head and Neck Radiation Therapy Planning](https://www.redjournal.org/article/S0360-3016(16)30887-2/fulltext) - Bulat Ibragimov, F Pernuš, P Strojan, L Xing (2016)
* [Three-Dimensional Radiotherapy Dose Prediction on Head and Neck Cancer Patients with a Hierarchically Densely Connected U-net Deep Learning Architecture](https://arxiv.org/abs/1805.10397) - Dan Nguyen, Xun Jia, David Sher, Mu-Han Lin, Zohaib Iqbal, Hui Liu, Steve Jiang (2019)

##### Pancreatic
* [An Interpretable Planning Bot for Pancreas Stereotactic Body Radiation Therapy](https://www.redjournal.org/article/S0360-3016(20)34407-2/fulltext) - Jiahan Zhang, Chunhao Wang, Yang Sheng, Manisha Palta, Brian Czito, Christopher Willett, Jiang Zhang, P. James Jensen, Fang-Fang Yin, Qiuwen Wu, Yaorong Ge, Q. Jackie Wu (2020)

##### Pelvic
* [Dose evaluation of fast synthetic-CT generation using a generative adversarial network for general pelvis MR-only radiotherapy](https://pdfs.semanticscholar.org/b185/c48ec9445fa411505049366d73fd9895049a.pdf) - Matteo Maspero, Mark H. F. Savenije, Anna M. Dinkla, Peter R. Seevinck, Martijn P. W. Intven, Ina M. Jurgenliemk-Schulz, Linda G. W. Kerkmeijer, Cornelis A. T. van den Berg (2018)

##### Prostate
* [Adversarial optimization for joint registration and segmentation in prostate CT radiotherapy](https://arxiv.org/abs/1906.12223) - Mohamed S. Elmahdy, Jelmer M. Wolterink, Hessam Sokooti, Ivana Išgum, Marius Staring (2019)

#### Surgery
* [Surgical aid visualization system for glioblastoma tumor identification based on deep learning and in-vivo hyperspectral images of human patients](https://doi.org/10.1117/12.2512569) - Himar Fabelo, Martin Halicek, Samuel Ortega, Adam Szolna, Jesus Morera, Roberto Sarmiento, Gustavo M. Callico, Baowei Fei (2019)

## Presentations
* [Deep Learning with Big Health Data for Early Cancer Detection and Prevention](http://chapter.aapm.org/orv/meetings/fall2017/5-Deep%20Learning%20with%20Big%20Health%20Data%20for%20Early%20Cancer%20Detection%20and%20Prevention.pdf) - Jun Deng, Yale (2017)
* [Automatic Prostate Cancer Classification using Deep Learning](http://www2.maths.lth.se/matematiklth/personal/kalle/deeplearning2018/LCCC_DeepLearning_IA.pdf) - Ida Arvidsson, Lund University, Sweden (2018)

## Other and meta resources
* [Machine Learning Is The Future Of Cancer Prediction](https://towardsdatascience.com/machine-learning-is-the-future-of-cancer-prediction-e4d28e7e6dfa) - Sohail Sayed (2018)
* [Google Deep Learning Tool 99% Accurate at Breast Cancer Detection](https://healthitanalytics.com/news/google-deep-learning-tool-99-accurate-at-breast-cancer-detection) - Jessica Kent (2018)
* [New Open-Source AI Machine Learning Tools to Fight Cancer](https://www.psychologytoday.com/us/blog/the-future-brain/201907/new-open-source-ai-machine-learning-tools-fight-cancer) - Cami Rosso (2019)
* [Deep Learning in Oncology – Applications in Fighting Cancer](https://emerj.com/ai-sector-overviews/deep-learning-in-oncology/) - Abder-Rahman Ali (2019)
* [Google Develops Deep Learning Tool to Enhance Lung Cancer Detection](https://healthitanalytics.com/news/google-develops-deep-learning-tool-to-enhance-lung-cancer-detection) - Jessica Kent (2019)
* [Histopathological Cancer Detection with Deep Neural Networks](https://towardsdatascience.com/histopathological-cancer-detection-with-deep-neural-networks-3399be879671) - Antonio de Perio (2019)
