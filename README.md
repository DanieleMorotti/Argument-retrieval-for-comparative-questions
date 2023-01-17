# Argument-retrieval-for-comparative-questions

This repository contains the work done for the Natural Language Processing course A.Y. 2022-2023 at the University of Bologna, Master's Degree in Artificial Intelligence.

We chose the project from the ["Touché shared tasks at CLEF 2022"](https://touche.webis.de/clef22/touche22-web/argument-retrieval-for-comparative-questions.html) website.
The synopsis for the challenge was:  
>Given a comparative topic and a collection of documents, the task is to retrieve relevant argumentative passages for either compared object or for both and to detect their respective stances with respect to the object they talk about.

We had to solve 2 different tasks during the project, the first one was more related to Information Retrieval while the second is about text classification.

## 1. Document retrieval for comparative questions
As explained above, we need to retrieve the most relevant text passages, with respect to a query, from a corpus of ~850k elements taken from the ClueWeb12 dataset. You can find all the available datasets for the task at [this link](https://zenodo.org/record/6873567).
<div align="center">
    <img src="/images/IR_recap.png" width="60%" />
    <p style="font-size:0.8rem" align="center">
        <em>Description of a basic pipeline</em> 
    </p>
</div>

### Implementation
In order to perform document retrieval efficiently, we had to create an index. We built our indexes with 2 different libraries, [Pyserini](https://github.com/castorini/pyserini) for creating the sparse indexes and [autofaiss](https://github.com/criteo/autofaiss) for the dense indexes.

We built several indexes, created on some variants of the corpus (e.g. expanded, pre-processed) to get the best results for our pipelines.

The main goal was to find models that perform well on both quality and relevance (the Touché team gave some files to evaluate this metrics considering the nDCG).
Indeed, we didn't create different models with different goals, but compact models with the objective of optimizing both scores.


You can look at the details of the different implemented pipelines in the ```src/``` directory, where the classes are located. 
You can find all the information that you need, in order to reproduce the results, in the ```document_retrieval.ipynb``` notebook. 

We suggest you to run the notebook in [Colab](https://colab.research.google.com/) to perform the heaviest operations and to import in an easier way the files from Google Drive. In the notebook and in the report we did some references to a Drive shared folder but it was only available for the professors to make them test the project. However, in the notebook you will find all the instructions to reproduce our experiments.


## 2. Stance detection
In this task we had to classify between 4 different classes: 
- NO, if the text doesn't express a stance;
- NEUTRAL, if the text doesn't favour any of the 2 objects of the query;
- FIRST, when the text favour the first object;
- SECOND, when the text favour the second object.
<div align="center">
    <img src="/images/stance_schema.png" width="55%" />
    <p style="font-size:0.8rem" align="center">
        <em>Description of a basic pipeline</em> 
    </p>
</div>

### Implementation
Our first idea was to create a unique model to classify the four classes. 
We imported a pre-trained version of DistilBERT from [Huggingface](https://huggingface.co/typeform/distilbert-base-uncased-mnli), we add a classifier layer on top of it and we fine-tuned on our data. 
<div align="center">
    <img src="/images/stance_model.png" width="30%" />
    <p style="font-size:0.8rem" align="center">
        <em>Unique model for stance detection</em> 
    </p>
</div>

Unfortunately we discovered that this approach didn't work very well, probably due to the few data available for the fine-tuning and the unbalanced classes.

In order to improve the performance, we read [a paper](https://dl.acm.org/doi/10.1145/3488560.3498534) that inspired us to split the entire pipeline in two different models.
- The ***first model*** had to detect if the text was favouring an object or not.
- The ***second model*** determined whether the text was favouring the first or the second object.
<div align="center">
    <img src="/images/stance_model2.png" width="55%" />
    <p style="font-size:0.8rem" align="center">
        <em>System with two models</em>
    </p>
</div>
This system worked better and we also deal with class imbalance setting different weights to the four classes.

You can find the whole implementation and a detailed explanation in the ```stance_detection.ipynb``` notebook.

## Project structure

    .
	├── document_retrieval.ipynb -> Notebook to run the document retrieval task on our models.
    |
    ├── stance_detection.ipynb   -> Notebook to run the stance detection models.
    │
	├── images/	             -> Directory that contains images for the README
    │
	├── src/     	             -> Directory that contains the classes for our pipelines and evaluation scripts
	│
	├── utils/                   -> Directory that contains some files to manage the download of the files and other useful functions.
    |
	├── README.md
    ├── LICENSE
	└── requirements.txt





