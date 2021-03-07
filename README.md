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
`
<b> Details: </b>


<pre>
1- Jeopardy.ipynb -> Generate modified dataset and generate fastext models for "ques","ans","value".
2- ML/DL folders contain code for different machine and deep learning codes.
3- Concatenated power means appraoch was proposed in:
4- Fasttext model and pre-trained embeddings downloaded from:

Baseline for binary classification: https://github.com/yashajoshi/Predicting-Value-of-Jeopardy-Questions
</pre>
