# Jeopardy--NLP

<b> Problem Statement:</b>

<pre> 
Build a model to predict the value of the question in the TV game show  “Jeopardy!”. 
Data can be downloaded from this link: https://www.kaggle.com/tunguz/200000-jeopardy-questions 

Data description 
▪ 'category' : the question category, e.g. "HISTORY" 
▪ ‘value' : $ value of the question as string, e.g. "$200" (Note - "None" for Final Jeopardy! and Tiebreaker questions) 
▪ 'question' : text of question (Note: This sometimes contains hyperlinks and other things messy text such as when there's a picture or video question) 
▪ 'answer' : text of answer 
▪ round' : one of "Jeopardy!","Double Jeopardy!","Final Jeopardy!"  or "Tiebreaker" (Note: Tiebreaker questions do happen but they're very rare (like once every 20 years)) 
▪ 'show_number' : string of show number, e.g '4680' 
▪ 'air_date' : the show air date in format YYYY-MM-DD </pre>

<b> Data Preparation </b>

<pre>
1- First 100k samples from the datatset were taken.
2- Only samples from Jeopardy round were selected.
3- Redundant features like 'round','show_number','airdate' were dropped.
3- Preprocess data : stopwords removal,stemming,lemmatization,lower-casing etc.
4- Depending upon binary/ multi class classification -> A class balanced dataset was prepared.

</pre>


<b> Approach: </b>


<pre>
1- Important features are: Question, Ans and Category
2- Using these three features -> value is predicted
3- To generate word embeddings -> fasttext model is fine-tuned on pretrained wiki news dataset.
   Pre-trained embeddings downloaded from: https://fasttext.cc/docs/en/english-vectors.html.
5- To generate sentence vectors from these word embeddings -> concatenated power means method is followed.
   Pmeans paper: https://arxiv.org/pdf/1803.01400.pdf
6- Sentence vectors of "ques","ans" and "category" were concatented together to generate final feature matrix.
7- Using these feature matrix-> Various ML and DL models were trained.
</pre>
    
<b> Results </b>
<pre>

<b> Case A: Binary Classification</b>

Baseline for binary classification: https://github.com/yashajoshi/Predicting-Value-of-Jeopardy-Questions

Best reported metric are: ![alt text](https://github.com/mayank05942/Jeopardy--NLP/blob/Images/res.jpg?raw=true)

Fasttext with pmeans->

<b> Case B: Multi-class Classification</b>





</pre>

