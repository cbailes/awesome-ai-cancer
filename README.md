[![Awesome](https://cdn.rawgit.com/sindresorhus/awesome/d7305f38d29fed78fa85652e3a63e154dd8e8829/media/badge.svg)](https://github.com/sindresorhus/awesome)

List of awesome code, papers, and resources for AI/deep learning/machine learning/neural networks applied to oncology.

Open access: all rights granted for use and re-use of any kind, by anyone, at no cost, under your choice of either the free MIT License or Creative Commons CC-BY International Public License.

© 2019 Craig Bailes ([@cbailes](https://github.com/cbailes) | [contact@craigbailes.com](mailto:contact@craigbailes.com))

## Index
* [Code](#code)
  + [Challenges](#challenges)
  + [Repositories](#repositories)
    - [Breast](#breast)
    - [Lung](#lung)
    - [Prostate](#prostate)
  + [Datasets](#datasets)
    - [Breast](#breast-1)
    - [Pancreatic](#pancreatic)
    - [Prostate](#prostate-1)
* [Papers](#papers)
  + [Detection and diagnosis](#detection-and-diagnosis)
    - [Breast](#breast-2)
    - [Cervical](#cervical)
    - [Liver](#liver)
    - [Lung](#lung-1)
    - [Mesothelioma](#mesothelioma)
    - [Nasopharyngeal](#nasopharyngeal)
    - [Pancreatic](#pancreatic-1)
    - [Prostate](#prostate-2)
  + [Prognostics](#prognostics)
    - [Breast](#breast-3)
    - [Colorectal](#colorectal)
    - [Liver](#liver-1)
    - [Pancreatic](#pancreatic-2)
  + [Treatment](#treatment)
    - [Drug design](#drug-design)
    - [Personalized chemotherapy](#personalized-chemotherapy)
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

### Repositories
* [zizhaozhang/nmi-wsi-diagnosis](https://github.com/zizhaozhang/nmi-wsi-diagnosis) - Pathologist-level interpretable whole-slide cancer diagnosis with deep learning
* [mark-watson/cancer-deep-learning-model](https://github.com/mark-watson/cancer-deep-learning-model) - Keras deep Learning neural network model for University of Wisconsin Cancer data that uses the Integrated Variants library to explain predictions made by a trained model with 97% accuracy
* [prasadseemakurthi/Deep-Neural-Networks-HealthCare](https://github.com/prasadseemakurthi/Deep-Neural-Networks-HealthCare/tree/master/Project%202%20--%20SurvNet%20--%20%20Cancer%20Clinical%20Outcomes%20Predictor) - Practical deep learning repository for predicting cancer clinical outcomes

#### Breast
* [AFAgarap/wisconsin-breast-cancer](https://github.com/AFAgarap/wisconsin-breast-cancer) - Codebase for *On Breast Cancer Detection: An Application of Machine Learning Algorithms on the Wisconsin Diagnostic Dataset [ICMLSC 2018 / arxiv 1711.07831]*
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

#### Lung
* [ncoudray/DeepPATH](https://github.com/ncoudray/DeepPATH) - Classification of Lung cancer slide images using deep-learning
* [AiAiHealthcare/ProjectAiAi](https://github.com/AiAiHealthcare/ProjectAiAi) - A project with the slated goal of building an FDA-approved, open source deep learning system for detecting lung cancer

#### Prostate
* [bikramb98/Prostate-cancer-prediction](https://github.com/bikramb98/Prostate-cancer-prediction) - A simple prostate cancer prediction model built using KNN on a small dataset
* [eiriniar/gleason_CNN](https://github.com/eiriniar/gleason_CNN) - An attempt to reproduce the results of an earlier paper using a CNN and original TMA dataset
* [I2Cvb/prostate](https://github.com/I2Cvb/prostate) - Prostate cancer research repository from the Initiative for Collaborative Computer Vision Benchmarking
* [jameshorine/prostate_cancer](https://github.com/jameshorine/prostate_cancer) - A random forest model for predicting prostate cancer survival
* [kkchau/NACEP-Analysis-of-Prostate-Cancer-Cells](https://github.com/kkchau/NACEP-Analysis-of-Prostate-Cancer-Cells) - NACEP Analysis of Tumorigenic prostate epithelial cells
* [ProstateCancer.ai](http://prostatecancer.ai) - An open-source web-application developed on the Tesseract-Medical Imaging platform designed to assist radiologists with a second opinion while simultaneously providing a standard image viewing and reporting scheme
* [rycoley/prediction-prostate-surveillance](https://github.com/rycoley/prediction-prostate-surveillance) - [[Paper](https://arxiv.org/pdf/1508.07511.pdf)] - Joint modeling of latent prostate cancer state and observed clinical outcomes for active surveillance of low risk prostate cancer
* [sahirbhatnagar/prostate](https://github.com/sahirbhatnagar/prostate) - Life expectancy tool for curative prostate cancer patients using Shiny app for R
* [Tesseract-MI/prostatecancer.ai](https://github.com/Tesseract-MI/prostatecancer.ai) - An OHIF-based, zero-footprint DICOMweb medical image viewer that utilizes AI to identify clinically significant prostate cancer
* [thom1178/Prostate-Cancer-Capsule](https://github.com/thom1178/Prostate-Cancer-Capsule) - [[Paper](https://github.com/thom1178/Prostate-Cancer-Capsule/blob/master/ProstateCancer.pdf)] - A logistic regression model to classify the capsule penetration of a tumor

### Datasets
* [Cancer Data Access System](https://biometry.nci.nih.gov/cdas/) @ National Cancer Institute
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
* [Architectures and accuracy of artificial neural network for disease classification from omics data](https://www.ncbi.nlm.nih.gov/pmc/articles/PMC6399893/) - Hui Yu, David C. Samuels, Ying-yong Zhao, Yan Guo (2019)
* [Artificial intelligence in cancer imaging: Clinical challenges and applications](https://onlinelibrary.wiley.com/doi/full/10.3322/caac.21552) - Wenya Linda Bi, Ahmed Hosny, Matthew B. Schabath, Maryellen L. Giger, Nicolai J. Birkbak, Alireza Mehrtash, Tavis Allison, Omar Arnaout, Christopher Abbosh, Ian F. Dunn, Raymond H. Mak, Rulla M. Tamimi, Clare M. Tempany, Charles Swanton, Udo Hoffmann, Lawrence H. Schwartz, Robert J. Gillies, Raymond Y. Huang, Hugo J. W. L. Aerts (2019)

### Detection and diagnosis
* [Deep learning for tumor classification in imaging mass spectrometry](https://academic.oup.com/bioinformatics/article/34/7/1215/4604594) - Jens Behrmann, Christian Etmann, Tobias Boskamp, Rita Casadonte, Jörg Kriegsmann,  Peter Maaβ (2018)
* [Spatial Organization and Molecular Correlation of Tumor-Infiltrating Lymphocytes Using Deep Learning on Pathology Images](https://www.sciencedirect.com/science/article/pii/S2211124718304479) - JoelSaltz, Rajarsi Gupta, Le Hou, Tahsin Kurc, Pankaj Singh, Vu Nguyen, Dimitris Samaras, Kenneth R. Shroyer, Tianhao Zhao, Rebecca Batiste, John Van Arnam (2018)

#### Breast
* [A new features extraction method based on polynomial regression for the assessment of breast lesion Contours](https://www.researchgate.net/publication/282954526_A_new_features_extraction_method_based_on_polynomial_regression_for_the_assessment_of_breast_lesion_Contours) - Spandana Paramkusham, Shivakshit Patri, K.M.M. Rao, B.V.V.S.N.P. Rao (2015)
* [Diagnostic Assessment of Deep Learning Algorithms for Detection of Lymph Node Metastases in Women With Breast Cancer](https://jamanetwork.com/journals/jama/article-abstract/2665774) - Babak Ehteshami Bejnordi, Mitko Veta, Paul Johannes van Diest (2017)
* [Machine Learning with Applications in Breast Cancer Diagnosis and Prognosis](https://www.researchgate.net/publication/325064884_Machine_Learning_with_Applications_in_Breast_Cancer_Diagnosis_and_Prognosis) - Wenbin Yue, Zidong Wang, Hongwei Chen, Annette Margaret Payne, Xiaohui Liu (2018)

#### Cervical
* [Intelligent Inverse Treatment Planning via Deep Reinforcement Learning, a Proof-of-Principle Study in High Dose-rate Brachytherapy for Cervical Cancer](https://arxiv.org/pdf/1811.10102.pdf) - Chenyang Shen, Yesenia Gonzalez, Peter Klages, Nan Qin, Hyunuk Jung, Liyuan Chen, Dan Nguyen, Steve B. Jiang, Xun Jia (2019)

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

#### Lung
* [Deep Reinforcement Learning for Automated Radiation Adaptation in Lung Cancer](https://www.ncbi.nlm.nih.gov/pmc/articles/PMC5734677/) - Huan-Hsin Tseng, Yi Luo, Sunan Cui, Jen-Tzung Chien, Randall K. Ten Haken, Issam El Naqa (2017)
* [Lung Nodule Detection via Deep Reinforcement Learning](https://www.frontiersin.org/articles/10.3389/fonc.2018.00108/full) - Issa Ali, Gregory R. Hart, Gowthaman Gunabushanam, Ying Liang, Wazir Muhammad, Bradley Nartowt, Michael Kane, Xiaomei Ma, Jun Deng (2018)
* [Deep reinforcement learning with its application for lung cancer detection in medical Internet of Things](https://www.sciencedirect.com/science/article/pii/S0167739X19303772) - Zhuo Liua, Chenhui Yaob, Hang Yuc, Taihua Wua (2019)
* [Automated Pulmonary Nodule Classification in Computed Tomography Images Using a Deep Convolutional Neural Network Trained by Generative Adversarial Networks](https://www.ncbi.nlm.nih.gov/pmc/articles/PMC6334309/) - Yuya Onishi, Atsushi Teramoto, Masakazu Tsujimoto, Tetsuya Tsukamoto, Kuniaki Saito, Hiroshi Toyama, Kazuyoshi Imaizumi, Hiroshi Fujita (2019)

#### Mesothelioma
* [Development and validation of a deep learning model using biomarkers in pleural effusion for prediction of malignant pleural mesothelioma](https://doi.org/10.1093/annonc/mdy301.003) - S. Ikegaki, Y. Kataoka, T. Otoshi, T. Takemura, M. Shimada (2018)
* [Diagnosis of mesothelioma with deep learning](https://www.ncbi.nlm.nih.gov/pmc/articles/PMC6341823/) - Xue Hu, Zebo Yu (2019)

#### Nasopharyngeal
* [Deep Deconvolutional Neural Network for Target Segmentation of Nasopharyngeal Cancer in Planning Computed Tomography Images](https://www.frontiersin.org/articles/10.3389/fonc.2017.00315/full) - Kuo Men, Xinyuan Chen, Ye Zhang, Tao Zhang, Jianrong Dai, Junlin Yi, Yexiong Li (2017)

#### Pancreatic
* [DeepOrgan: Multi-level Deep Convolutional Networks for Automated Pancreas Segmentation](https://link.springer.com/chapter/10.1007/978-3-319-24553-9_68) - Holger R. Roth, Le Lu, Amal Farag, Hoo-Chang Shin, Jiamin Liu, Evrim B. Turkbey, Ronald M. Summers (2015)
* [Deep convolutional networks for pancreas segmentation in CT imaging](https://www.spiedigitallibrary.org/conference-proceedings-of-spie/9413/94131G/Deep-convolutional-networks-for-pancreas-segmentation-in-CT-imaging/10.1117/12.2081420.short?SSO=1) - Holger R. Roth, Amal Farag, Le Lu, Evrim B. Turkbey, Ronald M. Summers (2015)
* [Pancreas Segmentation in Abdominal CT Scan: A Coarse-to-Fine Approach](https://pdfs.semanticscholar.org/788f/341d02130e1807edf88c8c64a77e4096437e.pdf) - Yuyin Zhou, Lingxi Xie, Wei Shen, Elliot Fishman, Alan Yuille (2016)
* [Combining Machine Learning and Nanofluidic Technology To Diagnose Pancreatic Cancer Using Exosomes](https://pubs.acs.org/doi/abs/10.1021/acsnano.7b05503) - Jina Ko, Neha Bhagwat, Stephanie S. Yee, Natalia Ortiz, Amine Sahmoud, Taylor Black, Nicole M. Aiello, Lydie McKenzie, Mark O’Hara, Colleen Redlinger, Janae Romeo, Erica L. Carpenter, Ben Z. Stanger, David Issadore (2017)
* [Deep learning based Nucleus Classification in pancreas histological images](https://ieeexplore.ieee.org/abstract/document/8036914) - Young Hwan Chang, Guillaume Thibault, Owen Madin, Vahid Azimi, Cole Meyers, Brett Johnson (2017)
* [Deep Supervision for Pancreatic Cyst Segmentation in Abdominal CT Scans](https://link.springer.com/chapter/10.1007/978-3-319-66179-7_26) - Yuyin Zhou, Lingxi Xie, Elliot K. Fishman, Alan L. Yuille (2017)
* [Deep LOGISMOS: Deep learning graph-based 3D segmentation of pancreatic tumors on CT scans](https://ieeexplore.ieee.org/abstract/document/8363793/) - Zhihui Guo, Ling Zhang, Le Lu, Mohammadhadi Bagheri, Ronald M. Summers, Milan Sonka, Jianhua Yao (2018)
* [Deep learning convolutional neural network (CNN) With Gaussian mixture model for predicting pancreatic cancer](https://link.springer.com/article/10.1007%2Fs11042-019-7419-5) - Kaushik Sekaran, P. Chandana, N. Murali Krishna, Seifedine Kadry (2019)

#### Prostate
* [Automated Gleason grading of prostate cancer tissue microarrays via deep learning](https://www.nature.com/articles/s41598-018-30535-1) - [[Dataset]([Arvaniti prostate cancer TMA dataset](https://doi.org/10.7910/DVN/OCYCMP)] - Eirini Arvaniti, Kim S. Fricker, Michael Moret, Niels Rupp, Thomas Hermanns, Christian Fankhauser, Norbert Wey, Peter J. Wild, Jan H. Rüschoff, Manfred Claassen (2018)
* [Comparison of Artificial Intelligence Techniques to Evaluate Performance of a Classifier for Automatic Grading of Prostate Cancer From Digitized Histopathologic Images](https://jamanetwork.com/journals/jamanetworkopen/fullarticle/2727273) - Guy Nir, Davood Karimi, S. Larry Goldenberg, Ladan Fazli, Brian F. Skinnider, Peyman Tavassoli, Dmitry Turbin, Carlos F. Villamil, Gang Wang, Darby J. S. Thompson, Peter C. Black, Septimiu E. Salcudean (2019)
* [Improving Prostate Cancer Detection with Breast Histopathology Images](https://arxiv.org/abs/1903.05769) - Umair Akhtar Hasan Khan, Carolin Stürenberg, Oguzhan Gencoglu, Kevin Sandeman, Timo Heikkinen, Antti Rannikko, Tuomas Mirtti (2019)

### Prognostics
* [Machine learning applications in cancer prognosis and prediction](https://www.sciencedirect.com/science/article/pii/S2001037014000464) - Konstantina Kourou, Themis P. Exarchos, Konstantinos P. Exarchos, Michalis V. Karamouzis, Dimitrios I. Fotiadis (2015)

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

### Treatment
#### Drug design
* [Multicellular Target QSAR Model for Simultaneous Prediction and Design of Anti-Pancreatic Cancer Agents](https://pubs.acs.org/doi/10.1021/acsomega.8b03693) - Alejandro Speck-Planche (2019)

#### Personalized chemotherapy
* [Reinforcement learning design for cancer clinical trials](https://www.researchgate.net/publication/26808581_Reinforcement_learning_design_for_cancer_clinical_trials) - Yufan Zhao, Michael R. Kosorok, Donglin Zeng (2009)
* [Personalized Cancer Chemotherapy Schedule: a numerical comparison of performance and robustness in model-based and model-free scheduling methodologies](https://arxiv.org/abs/1904.01200) - Jesus Tordesillas, Juncal Arbelaiz (2019)

## Presentations
* [Deep Learning with Big Health Data for Early Cancer Detection and Prevention](http://chapter.aapm.org/orv/meetings/fall2017/5-Deep%20Learning%20with%20Big%20Health%20Data%20for%20Early%20Cancer%20Detection%20and%20Prevention.pdf) - Jun Deng, Yale (2017)

## Other and meta resources
* [Machine Learning Is The Future Of Cancer Prediction](https://towardsdatascience.com/machine-learning-is-the-future-of-cancer-prediction-e4d28e7e6dfa) - Sohail Sayed (2018)
* [Google Deep Learning Tool 99% Accurate at Breast Cancer Detection](https://healthitanalytics.com/news/google-deep-learning-tool-99-accurate-at-breast-cancer-detection) - Jessica Kent (2018)
* [Deep Learning in Oncology – Applications in Fighting Cancer](https://emerj.com/ai-sector-overviews/deep-learning-in-oncology/) - Abder-Rahman Ali (2019)
* [Google Develops Deep Learning Tool to Enhance Lung Cancer Detection](https://healthitanalytics.com/news/google-develops-deep-learning-tool-to-enhance-lung-cancer-detection) - Jessica Kent (2019)
* [Histopathological Cancer Detection with Deep Neural Networks](https://towardsdatascience.com/histopathological-cancer-detection-with-deep-neural-networks-3399be879671) - Antonio de Perio (2019)
