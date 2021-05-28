import flask
import pickle
import nltk
nltk.download('stopwords')
nltk.download('wordnet')
nltk.download('averaged_perceptron_tagger')
nltk.download('punkt')
import pandas as pd
import numpy as np
import re
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer


app = flask.Flask(__name__, template_folder='templates')

with open('/home/vpoc1/mysite/mlb.pkl','rb') as f:
    mlb = pickle.load(f)

with open('/home/vpoc1/mysite/TF-IDF_25.pkl','rb') as f:
    vectorizer = pickle.load(f)

with open('/home/vpoc1/mysite/PCA_25.pkl','rb') as f:
    pca = pickle.load(f)

with open('/home/vpoc1/mysite/lr_opti.pkl','rb') as f:
    lr = pickle.load(f)

def cleanPunc(sentence) :
    cleaned = re.sub(r'[?|!|\'|"|#|@]', r'',sentence)
    cleaned = re.sub(r'[.|,|;|:|}|{|)|(|\|/]', r' ',cleaned)
    cleaned = cleaned.strip()
    cleaned = cleaned.replace("\n"," ")
    return cleaned

def keepAlpha(sentence) :
    alpha_sentence = ""
    for word in sentence.split():
        alpha_word = re.sub('[^a-z A-Z]+', ' ', word)
        alpha_sentence += alpha_word
        alpha_sentence += " "
    alpha_sentence = alpha_sentence.strip()
    return alpha_sentence

def cleanStopwords(sentence) :
  text_tokens = word_tokenize(sentence)
  stop_words = nltk.corpus.stopwords.words('english')
  text = ' '.join([word for word in text_tokens if not word in stop_words])
  return text

def lemmatization(sentence) :
    lemmatizer = WordNetLemmatizer()
    text = [lemmatizer.lemmatize(word) for word in sentence.split()]
    return text

def cleanDigit(sentence) :
  exception = ['a', 'b', 'd', 'e', 'f', 'g', 'h', 'i', 'j', 'k', 'l', 'm', 'n', 'o', 'p', 'q', 's', 't', 'u', 'v', 'w', 'x', 'y', 'z']
  text = ' '.join([w for w in sentence.split() if (w not in exception)])
  return text

def cleanPos(sentence) :
  verb_adverb_pronoun = ['PRP','PRP$', 'RB', 'RBR', 'RBS', 'TO', 'VB', 'VBG', 'VBD', 'VBN', 'VBP', 'VBZ', 'WDT', 'WP', 'WP$', 'WRB']
  tagged_sentence = nltk.tag.pos_tag(sentence.split())
  edited_sentence = ' '.join([word for word, tag in tagged_sentence if (tag not in verb_adverb_pronoun)])
  return edited_sentence

def get_tags(question) :
    predict = lr.predict(question)
    predict = mlb.inverse_transform(predict)
    strpredict = str(predict)
    strpredict = strpredict.replace("[(", "")
    strpredict = strpredict.replace(")]", "")
    return (strpredict)


@app.route('/', methods=['GET', 'POST'])
def main() :
    if flask.request.method == 'GET':
        return(flask.render_template('template.html'))

    if flask.request.method == 'POST':
        question = flask.request.form['question']
        col_name = ['question_text']
        df_question = pd.DataFrame([question], columns = col_name)
        df_question['question_text'] = df_question['question_text'].str.lower()
        df_question['question_text'] = df_question['question_text'].apply(cleanPos)
        df_question['question_text'] = df_question['question_text'].apply(cleanPunc)
        df_question['question_text'] = df_question['question_text'].apply(keepAlpha)
        df_question['question_text'] = df_question['question_text'].apply(cleanStopwords)
        df_question['question_text'] = df_question['question_text'].apply(lemmatization)
        df_question['question_text'] = df_question['question_text'].apply(lambda x: ' '.join(map(str, x)))
        df_question['question_text'] = df_question['question_text'].apply(cleanDigit)
        list_question = list(df_question['question_text'])
        question_vect = vectorizer.transform(list_question)
        question_vect_array = question_vect.toarray()
        question_pca = pca.transform(question_vect_array)
        tags = get_tags(question_pca)
        return(flask.render_template('result.html', tags_suggested = tags))

if __name__ == '__main__':
    app.run(debug=True)