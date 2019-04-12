# Analysis of Structure in Semi-Structured Resumes for Prediction and Classification

Sophia X Cui
W 266: Natural Language Processing
UC Berkeley School of Information
sophia@ischool.berkeley.edu

## Abstract
Semi-structured documents encompass a wide corpus of documents available on the web (medical records, resumes, Wikipedia, etc). For these types of documents, metadata embedded in the structure as well as the hierarchy often provides extra insight for contextual understanding and interpretation. (add a line for motivation) This project explored the impact of structure and metadata from a set of semi-structured resumes with respect to classification and prediction. The relatively high accuracy rates obtained on both baseline and bi-directional LSTM RNN on several types of predictions suggest that utilizing the HTML structure of documents could be a promising approach to certain types of analysis on such corpi.

Keywords: Semi-Structured Documents; Resumes; LSTM RNN

## 1 Introduction

"Semi-structured data is a form of structured data that does not conform with the formal structure of data models associated with relational databases or other forms of data tables, but nonetheless contains tags or other markers to separate semantic elements and enforce hierarchies of records and fields within the data." - Wikipedia 

There has been rapid growth of digital semi-structured data in the last couple of decades, examples being medical records, online profiles and linked semantic data. Many technologies and tools also gained prominence as a medium of exchanging such information, such as json, XML for loose structure definition and MongoDB and NoSQL for storage. Understanding, parsing, extraction and prediction on these types of data set has also spurned different kinds of tools. From MongoDB query language to JQuery class selectors, loosely quering and parsing this type of information is a very different paradigm from both traditional relational databases as well as raw text. 

While machine learning based NLP has primarily focused on raw text as the primary input, there has been increasily more research on how deep learning can be applied to semi-structured data for information extraction and prediction [1][2][3]. This paper explores the impact of HTML structure in semi-structured resumes for prediction in bi-directional LSTM RNN using TFIDF logistic regression as a baseline.

Reasons why applications of machine learning is interesting

Traditional NLP focused on NGrams.


We aim to compare these two methodologies with respect to each other on the corpus of resumes, which represents a substantial class of semi-structured documents (medical records, resumes, Wikipedia, etc). These documents contain discrete sections and most times contain HTML markup (available on the web). 


## 2 Project Overview
### 2.1 Background
Previous work successfully leveraged HTML markup structure in web documents with deep neural networks to successfully parse and extract information [6][7]. Other work extracted structure and clustered text as a means to narrow the search space and produce more accurate results [4][5]. Hence, we know it's possible that using structure in deep neural networks for parsing and extraction can be beneficial. 

< some other background on problem approach>

### 2.2 Datasets Utilized
We used pre-scraped dump of 8 million unique English resumes from Indeed from 2017 (https://resumes.indeed.com/). The resumes are in HTML with consistent classes and ids for corresponding sections of the resume. For example, individual work experiences are consistently classed 'work-experience-section' which nests under a work experience section, with id 'work-experience-items'. 

We also used word2vec slim (https://github.com/eyaler/word2vec-slim) as pretrained embeddings, with a vocabulary size of 300k trained from Google News. Because the vocabulary of resumes is dominated contain common English terms, and that pretrained embeddings can boost performance on the relatively small sampled data sets we used to run models [9]. < why word2vec? >


### 2.3 Problem Approach

- insert line on why these predictions were chosen

This project focused on benchmarking accuracies from three types of classifications and predictions from the dataset:

1. Has 10+ years of work experience
2. Stayed at current job next year
3. Switched careers from their last job

Our methodology was to start with an exploratory analysis, ensuring that for unbalanced predictions, we choose the metrics that would accurately gauge performance. The breakdown for types for each prediction < insert graph >

We sampled our dataset for training sets of roughly 30,000 to 40,000, validating on 10,000 resumes with random resampling. The development set was a separate batch of 50,000 resumes. Originally, we aimed for an order of magnitude larger for training, but hit time and resource limitations when training 50 variants of LSTM networks for our predictor variables.

As baselines, we ran both TFIDF Multinomial Naive Bayes and TFIDF Logistic Regression, which are common and fast baseline models for text classification [10].  

Variants of Bi-Directional LSTM RNN was chosen as the main model of comparison against baselines. Due to the nature of HTML contain open and end tags to represent a section, we hypothesized that bi-directional will allow us to capture backward and future context for a HTML structured representation. We chose a LSTM RNN because we wanted to capture long range dependencies in the resume, potentially extrapolating conclusions from different sections of the resume.  

Because LSTM RNNs can suffer from over training on a smaller dataset, we also ran multiple variations of parameter inputs, namely the number of epochs and sequence length. 

### 2.4 Parsing Tools and Data Preparation 

Our data set was parsed by BeautifulSoup (https://pypi.org/project/beautifulsoup4/) which helped us standardize and clean HTML as well as extract raw text.

For each resume, we verified that the expected sections exist: education, work experience and has had at least 2 jobs. We also validate that the expected values are complete: years of employment and full job titles for past jobs. Resumes that were incomplete were not included in the dataset.

For each of the predictions, we systematically extracted the predictor variable and stripped the resume of information when applicable. Some of these predictors could have a more nuanced interpretation of actual True or False, but we defaulted to simplified proxies for those cases as described below:

1. Has 10+ years of work experience: We did not alter the resume as the training set. The difference between earliest date of their employment history compared to 2017 was the boolean predictor of whether they had at least 10 years of experience.
2. Stayed at current job at year +1: We removed the end date from the current or last position of the job and cleaned the HTML of any potential references to current position. If the end date was 'to present', we assigned that predictor as True.
3. Switched careers from their last job: We removed the job title from the current or last position of the job. If the extracted position is similar by counting the number of matching tokens compared to their previous position, we assigned that predictor as True. 

For each of the three predictor variables above, we created two training sets, one with validated HTML as part of the resume, and one which we stripped the HTML but kept the spacing and line breaks. 

< maybe add more here >

### 2.5 Word2Vec Embeddings

Pretrained embeddings such as GloVe, FastText and Word2Vec encode latent relationships between words which can be utilized as part of machine learning model. We could have trained a version of Word2Vec on the more specific vocabulary of resumes to encode stronger relationships more specific to this corpus, but this would be a more tangential exploration to our project. 

Because our sampled training dataset was relatively small, we incorporated the pretrained Word2Vec slim embeddings (300k) to our LSTM models to boost training performance and semantic relationships. The same embeddings were used across all LSTM models. Embeddings were not updated during training.

### 2.5 Data Limitations

- original predict classes of job titles
- small training set, could have scaled up with more resumes trained
- better data preparation

### 2.6 Metrics

- minimal class skew
- confusion matrix shows fairly even precision and recall 
- no need to differentiate false positives / false negatives -> straight up accuracy

- f1, loss, accuracy


## 3 Model
### 3.1 Baseline Models

Given the training datasets we created, we built two baseline models using TFIDF Multinomial Naive Bayes and TFIDF Logistic Regression, which are common baseline models for text classification [10]. Naive Bayes was able to statistically identify high salience terms while Logistic Regression serves as a baseline and sanity test, as it is essentially a single neuron neural network with a linear decision boundary. TFIDF is chosen over absolute counts for vectorization as it more effectively categorizes relevant terms in a document [11]. 

We created 4 baselines for each predictor, using Logistic Regression and Naive Bayes for both plain text resumes and HTML decorated resumes. Results are shown in figure <>.

### 3.2 Baseline Feature Engineering

In addition to our baseline models, we also attempted rudimentary feature engineering on our dataset to incorporate into our baseline models. Some of the features included the length of the document, the average length of tokens, and number of unique tokens. Some of these features are shown to have correlation with the type of industry a resume is in and the length of experience. Although these features mainly applied to the plain text baseline, we ran across both plain text and HTML markup for completeness of results.

Logistic Regression models that included engineered features did not perform substantially better than vanilla baseline models. Results are shown in figure <>.

### 3.3 Bi-Directional LSTM

The training objective of our model is cross-entropy loss using ADAM optimizer. A recurrent dropout layer was applied and the rate set to 0.1. We used pretrained Word2Vec slim to initialize token embeddings. Words not in the vocabulary were initialized by random normals with mean and standard deviation derived from the loaded embeddings. Batch size was set to 32 examples. The model is implemented using the deep learning framework Keras [12].

### 3.4 Tuning Bi-Directional LSTM

We limited tuning of our main model to two parameters, number of epochs and sequence length. Due to the smaller training set size and avoid overfitting, we varied the number of epochs per predictor to retain the best fit model for the development set. We also tuned sequence length because the corpus of resumes tend to have different lengths of relevant context compared to a news article. Although 200 to 300 characters is standard for sentences, we tried 400 and 800 as an attempt to capture more state in any particular resume section and to address the padded length of HTML context around text.

< insert figure for test accuracy over time >

## 4 Results

We focused on the absolute accuracy of the test dataset as the main metric since for all of our predictors, we have a fairly balanced dataset and differentiating between precision and recall is not as important. We did investigate f1-accuracy as a sub point below.

### 4.1 Analysis of Tuning

The effects of overfitting a small dataset is clearly represented in most of the predictor variables particularly when utilizing a shorter sequence length. For both plain text and HTML, 5 out of 6 test accuracies decreased for a sequence length of 400 from 5 epochs to 15 epochs. This is not true for a longer sequence length of 800, which highlights the importance of tuning dependent input parameters in parallel specific to the dataset. 

Using a longer sequence length of 800 also increased the performance of all of the LSTM models for each predictor. This indicates that while part of speech and sentence specific NLP models do well with a shorter sequence length, a longer sequence length is better for long range dependencies in a document. 

Although the same parameter values were ran for both plain text and HTML models, it's also clear that plain text and HTML benefit from different values of parameters. For predicting a decade of experience, sequence length strongly impacted HTML's accuracy whereas it did not for plain text. For predicting career switch, additional epochs decreased the accuracy for plain text, but increased the performance of HTML for a sequence length of 800. See figure for more details.

< insert tunable figure here >

### 4.2 Analysis against Baselines

In general, Bi-Directional LSTM did well compared to the baseline models, with an average increase of 7% accuracy over TFIDF logistic regression across three predictor variables. LSTM was the worst at predicting career switch as it underperformed baselines by a smidgen, accuracy decreased by 0.3% for both plain text and HTML. LSTM saw the best gains in predicting stay at job, accuracy increasing 23% over the baseline. 

Comparing results between plain text and HTML models, Naive Bayes performed worst in general and worse for HTML models. However, other models generally had comparable performance between the two types of data input (accuracies differed less than 0.5%) the only exception of predicting stay at job. Particularly notable is that HTML data worked the same or better for LSTM models across all three predictors.

- line on why predict job is an outlier

< insert baseline comparison figure here >

### 4.3 F1 Accuracy on LSTM Models

Although absolute accuracy was similar for predicting decade of experience on LSTM models, the same is not true for f1-accuracy. Accounting for both precision and recall, the LSTM HTML models scored on average 4% better on f1-accuracy than plain text. Comparing confusion matrix for the best LSTM models on predicting experience, we see that not only is HTML a fraction more accurate, but also it's balances false positives and false negatives better than plain text.

< insert f1 chart >
< insert confusion matrices >

### 4.4 Analysis of Predictors

Among the three predictors, predicting career switchers had highest error rate for LSTMs. Partly due to the inherit difficulty of the task, error analysis showed that even humans would have a hard time correctly identifying the cases where LSTM had failed. The most salient terms identified by the Logistic regression baseline consisted of specific job titles, e.g. 'nurse', 'truck driver', 'teacher' and career attributes e.g. 'senior' and 'certified'. We conjecture that logistic regression has very similar performance to LSTM partly due to linear relationships for specific job function retention.

Classifying whether someone had a decade more of work experience had a fairly high accuracy baseline of 90%, where salient terms identified from logistic regression were individual years previous to this decade. However, where LSTM improved significantly was the ability to identify work experience dates separately from dates in education, certifications or skill, e.g. 'SQL Server 2005' and reach a near perfect accuracy rate. The best model was HTML LSTM at sequence length 800 running 10 epochs, which outperformed plain text by a fraction of a percent.

LSTMs had the most success for HTML resumes on predicting whether someone stayed at the same job following year, a 23% increase from baseline. 

- insert more stuff here on analysis for staying at job


## 5 Conclusion

We demonstrated the viability of using structured HTML in Bi-Directional LSTM RNN models a resume corpus for prediction and classification. Model features included pretrained Word2Vec embeddings and features generated by the RNN. While LSTM was either on par or significantly better than baseline logistic regression models, using HTML in the neural models also demonstrated performance on par or better than plain text. 

Despite the positive results, utilizing larger training sets, creating a custom trained set of embeddings from the resume corpus, and further tuning of LSTM input parameters would likely improve performance for each of the predictor variables we tested. 

## References

[1] H. Wang, X. Zhang, A Neural Question Answering Model Based on Semi-Structured Tab
http://aclweb.org/anthology/C18-1165

[2] J. Krishnamurthy, P. Dasigi, and M. Gardner, Neural Semantic Parsing with Type Constraints for Semi-Structured Tables
https://www.aclweb.org/anthology/D17-1160

[3] G. Qin, J.G. Yao, Learning Latent Semantic Annotations for Grounding Natural Language to Structured Data
http://aclweb.org/anthology/D18-1411

[4] E. Angelino, Extracting Structure from human-readable Semistructured Text
https://people.eecs.berkeley.edu/~elaine/pubs/angelino-structure.pdf

[5] A. Hotho, S. Staab, G. Stumme, Text Clustering Based on Background Knowledge
https://pdfs.semanticscholar.org/56d0/7518b3a83229656a993adf0dd64ba26da3a6.pdf

[6] T. Gogar, O. Hubacek, J. Sedivy, Deep Neural Networks for Web Page Information Extraction
https://hal.inria.fr/hal-01557648/document

[7] A. García-Plaza, V. Fresno, Using Fuzzy Logic to Leverage HTML Markup for Web Page Representation
https://ieeexplore.ieee.org/abstract/document/7505655

[8] Wikipedia: Semi-structured data
https://en.m.wikipedia.org/wiki/Semi-structured_data

[9] Y. Qi, D. Sachan, When and Why are Pre-trained Word Embeddings Useful for Neural Machine Translation?
https://www.aclweb.org/anthology/N18-2084

[10] S. Wang, C. Manning, Baselines and bigrams: simple, good sentiment and topic classification
https://dl.acm.org/citation.cfm?id=2390688

[11] J. Ramos, Using TF-IDF to Determine Word Relevance in Document Queries 
https://www.semanticscholar.org/paper/Using-TF-IDF-to-Determine-Word-Relevance-in-Queries-Ramos/b3bf6373ff41a115197cb5b30e57830c16130c2c

[12] Keras: The Python Deep Learning library
https://keras.io/

