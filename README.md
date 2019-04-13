# W266 Final Project 

## Analysis of Structure in Resumes for Prediction and Classification

W 266: Natural Language Processing
UC Berkeley School of Information

Findings and [Final Paper](ResumeStructureLSTM.pdf)

## Abstract

Semi-structured documents encompass a wide corpus of documents available on the web, e.g. medical records, online profiles, semantic linked data, or Wikipedia. For these types of documents, metadata embedded in the structure as well as the hierarchy often provides additional insight for contextual understanding and interpretation. For example, while a bold larger font face can denote a section heading to human readers, machine learning may not glean the importance of the code that styled the text. This project explored the impact of structure and metadata from a set of semi-structured HTML-formatted resumes with respect to classification and prediction. The relatively high accuracy rates obtained by bidirectional LSTM RNNs on several types of predictions suggest that utilizing the structure and metadata of documents could be a promising approach to machine learning on such corpi.

## Data not in Repository

Some datasets and models are too large for the repository and have been cross linked here:

[Word2Vec Slim](https://github.com/eyaler/word2vec-slim)

[Resume Excerpt 1](https://s3-us-west-2.amazonaws.com/sophiaxcui.com/data/resume-nlp/indeed_com-job_deduped_n_merged_20170315_201357376193103.xml)
[Resume Excerpt 2](https://s3-us-west-2.amazonaws.com/sophiaxcui.com/data/resume-nlp/indeed_com-job_deduped_n_merged_20170315_201536923698467.xml)

[Cleaned Training Dataset 1](https://s3-us-west-2.amazonaws.com/sophiaxcui.com/data/resume-nlp/train_2_xml.p)
[Cleaned Training Dataset 2](https://s3-us-west-2.amazonaws.com/sophiaxcui.com/data/resume-nlp/train_3_xml.p)
[Cleaned Training Dataset 3](https://s3-us-west-2.amazonaws.com/sophiaxcui.com/data/resume-nlp/train_4_xml.p)

[Sample Saved Results from LSTM](https://s3-us-west-2.amazonaws.com/sophiaxcui.com/data/resume-nlp/results_LSTM_years.p) 
