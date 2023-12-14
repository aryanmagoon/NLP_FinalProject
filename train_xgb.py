import xgboost as xgb
import pandas as pd
from sklearn.model_selection import train_test_split
import optuna
import contractions
import re
from string import punctuation
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
import scipy

#read the dataset generated by 'generate_cohere_embeddings.py'
cohere_df=pd.read_csv('drive/MyDrive/Quora_Pairs/train_embedded_light.csv')

#custom callback function for XGB to retreieve evaluation log loss for Optuna
class getEvalLog(xgb.callback.TrainingCallback):
    def __init__(self, eval_log):
        self._eval_log = eval_log

    def after_iteration(self, model, epoch, evals_log):
        self._eval_log.append(evals_log)

#Cohere Embeddings
def objective_cohere(trial):
    X=cohere_df.drop(['is_duplicate'], axis=1)
    y=cohere_df['is_duplicate']

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)
    dtrain = xgb.DMatrix(X_train, label=y_train)
    dtest = xgb.DMatrix(X_test, label=y_test)
    eval_log = []

    params = {
    'booster': 'gbtree',
    'tree_method': 'gpu_hist',
    'max_depth': 10,
    'eta': 0.1,
    'objective': 'binary:logistic',
    'eval_metric': 'logloss',
    "lambda": trial.suggest_loguniform("lambda", 1e-8, 1.0),
    "alpha": trial.suggest_loguniform("alpha", 1e-8, 1.0),
    "max_depth": trial.suggest_int("max_depth", 1, 9),
    "eta" : trial.suggest_loguniform("eta", 1e-8, 1.0),
    "gamma": trial.suggest_loguniform("gamma", 1e-8, 1.0),
    }

    # Add a callback for pruning.
    pruning_callback = optuna.integration.XGBoostPruningCallback(trial, "validation-logloss")
    bst = xgb.train(params, dtrain, evals=[(dtest, "validation")], callbacks=[pruning_callback, getEvalLog(eval_log)], num_boost_round=100)

    return eval_log[0]['validation']['logloss'][-1]
study_cohere = optuna.create_study(direction="minimize")
study_cohere.optimize(objective_cohere, n_trials=30)
print(study_cohere.best_params)
print("Best Log-Loss from Custom Embeddigns: " + str(study_cohere.best_value))

#Preprocess data for TFIDF and Bag of Words Embeddings

train_df = pd.read_csv('train.csv')

train_df=train_df.dropna()
print(len(train_df))

train_df['question1'] = train_df['question1'].str.lower()
train_df['question2'] = train_df['question2'].str.lower()

without_contractions_q1= train_df['question1'].tolist()
without_contractions_q2 = train_df['question2'].tolist()

for i in range(len(without_contractions_q1)):
  #utilize contractions library to expand contractions
  without_contractions_q1[i] = contractions.fix(without_contractions_q1[i])
  without_contractions_q2[i] = contractions.fix(without_contractions_q2[i])


train_df['question1']=without_contractions_q1
train_df['question2']=without_contractions_q2
#remove punctuation
train_df['question1'] = [''.join([c for c in text if c not in punctuation]) for text in train_df['question1']]
train_df['question2'] = [''.join([c for c in text if c not in punctuation]) for text in train_df['question2']]

##remove comma between numbers
train_df['question1'] = train_df['question1'].apply(lambda x: re.sub('(?<=[0-9])\,(?=[0-9])', "", x))
train_df['question2'] = train_df['question2'].apply(lambda x: re.sub('(?<=[0-9])\,(?=[0-9])', "", x))
##replace non-ascii characters
train_df['question1'] = train_df['question1'].apply(lambda x: re.sub('[^\x00-\x7F]+', 'non-ascii-text', x))
train_df['question2'] = train_df['question2'].apply(lambda x: re.sub('[^\x00-\x7F]+', 'non-ascii-text', x))
##remove extra spaces
train_df['question1'] = train_df['question1'].apply(lambda x: re.sub('\s+', ' ', x))
train_df['question2'] = train_df['question2'].apply(lambda x: re.sub('\s+', ' ', x))

all_texts_q1=" ".join(train_df['question1']).split()
all_texts_q2=" ".join(train_df['question2']).split()

all_texts_q1.extend(all_texts_q2)
#vocab size
print('Vocab Size after Preprocessing: '+ str(len(set(all_texts_q1))))

#TFIDF Embeddings Word, Character, and N-Gram

tfidf_vectorizer_word = TfidfVectorizer(analyzer='word', token_pattern=r'\w{1,}', max_features=1000)
tfidf_vectorizer_word.fit(pd.concat((train_df['question1'],train_df['question2'])).unique())

y_word_tfidf = train_df['is_duplicate'].values
X_word_tfidf = scipy.sparse.hstack((tfidf_vectorizer_word.transform(train_df['question1'].values),tfidf_vectorizer_word.transform(train_df['question2'].values)))
def tfidf_objective_word(trial):
    global X_word_tfidf, y_word_tfidf

    X_train, X_test, y_train, y_test = train_test_split(X_word_tfidf, y_word_tfidf, test_size=0.2)
    dtrain = xgb.DMatrix(X_train, label=y_train)
    dtest = xgb.DMatrix(X_test, label=y_test)
    eval_log = []

    params = {
    'booster': 'gbtree',
    'tree_method': 'gpu_hist',
    'eta': 0.1,
    'objective': 'binary:logistic',
    'eval_metric': 'logloss',
    "lambda": trial.suggest_loguniform("lambda", 1e-8, 1.0),
    "alpha": trial.suggest_loguniform("alpha", 1e-8, 1.0),
    "max_depth": trial.suggest_int("max_depth", 1, 9),
    "eta" : trial.suggest_loguniform("eta", 1e-8, 1.0),
    "gamma": trial.suggest_loguniform("gamma", 1e-8, 1.0),
    }

    # Add a callback for pruning.
    pruning_callback = optuna.integration.XGBoostPruningCallback(trial, "validation-logloss")
    bst = xgb.train(params, dtrain, evals=[(dtest, "validation")], callbacks=[pruning_callback, getEvalLog(eval_log)], num_boost_round=100)

    return eval_log[0]['validation']['logloss'][-1]
study_tfidf_word = optuna.create_study(direction="minimize")
study_tfidf_word.optimize(tfidf_objective_word, n_trials=30)
print(study_tfidf_word.best_params)
print("Best Log-Loss from TFIDF Word Embeddings: " + str(study_tfidf_word.best_value))

tfidf_vectorizer_character = TfidfVectorizer(analyzer='char', token_pattern=r'\w{1,}', max_features=1000)
tfidf_vectorizer_character.fit(pd.concat((train_df['question1'],train_df['question2'])).unique())

y_char_tfidf = train_df['is_duplicate'].values
X_char_tfidf = scipy.sparse.hstack((tfidf_vectorizer_character.transform(train_df['question1'].values),tfidf_vectorizer_character.transform(train_df['question2'].values)))
def tfidf_objective_char(trial):
    global X_char_tfidf, y_char_tfidf

    X_train, X_test, y_train, y_test = train_test_split(X_char_tfidf, y_char_tfidf, test_size=0.2)
    dtrain = xgb.DMatrix(X_train, label=y_train)
    dtest = xgb.DMatrix(X_test, label=y_test)
    eval_log = []

    params = {
    'booster': 'gbtree',
    'tree_method': 'gpu_hist',
    'max_depth': 10,
    'eta': 0.1,
    'objective': 'binary:logistic',
    'eval_metric': 'logloss',
    "lambda": trial.suggest_loguniform("lambda", 1e-8, 1.0),
    "alpha": trial.suggest_loguniform("alpha", 1e-8, 1.0),
    "max_depth": trial.suggest_int("max_depth", 1, 9),
    "eta" : trial.suggest_loguniform("eta", 1e-8, 1.0),
    "gamma": trial.suggest_loguniform("gamma", 1e-8, 1.0),
    }

    # Add a callback for pruning.
    pruning_callback = optuna.integration.XGBoostPruningCallback(trial, "validation-logloss")
    bst = xgb.train(params, dtrain, evals=[(dtest, "validation")], callbacks=[pruning_callback, getEvalLog(eval_log)], num_boost_round=100)

    return eval_log[0]['validation']['logloss'][-1]
study_tfidf_char = optuna.create_study(direction="minimize")
study_tfidf_char.optimize(tfidf_objective_char, n_trials=30)
print(study_tfidf_char.best_params)
print("Best Log-Loss from TFIDF Character Embeddings: " + str(study_tfidf_char.best_value))

tfidf_vectorizer_ngram = TfidfVectorizer(analyzer='char_wb', token_pattern=r'\w{1,}', max_features=5000)
tfidf_vectorizer_ngram.fit(pd.concat((train_df['question1'],train_df['question2'])).unique())
y_ngram_tfidf = train_df['is_duplicate'].values
X_ngram_tfidf = scipy.sparse.hstack((tfidf_vectorizer_ngram.transform(train_df['question1'].values),tfidf_vectorizer_ngram.transform(train_df['question2'].values)))
def tfidf_objective_ngram(trial):
    global X_ngram_tfidf, y_ngram_tfidf

    X_train, X_test, y_train, y_test = train_test_split(X_ngram_tfidf, y_ngram_tfidf, test_size=0.2)
    dtrain = xgb.DMatrix(X_train, label=y_train)
    dtest = xgb.DMatrix(X_test, label=y_test)
    eval_log = []

    params = {
    'booster': 'gbtree',
    'tree_method': 'gpu_hist',
    'max_depth': 10,
    'eta': 0.1,
    'objective': 'binary:logistic',
    'eval_metric': 'logloss',
    "lambda": trial.suggest_loguniform("lambda", 1e-8, 1.0),
    "alpha": trial.suggest_loguniform("alpha", 1e-8, 1.0),
    "max_depth": trial.suggest_int("max_depth", 1, 9),
    "eta" : trial.suggest_loguniform("eta", 1e-8, 1.0),
    "gamma": trial.suggest_loguniform("gamma", 1e-8, 1.0),
    }

    # Add a callback for pruning.
    pruning_callback = optuna.integration.XGBoostPruningCallback(trial, "validation-logloss")
    bst = xgb.train(params, dtrain, evals=[(dtest, "validation")], callbacks=[pruning_callback, getEvalLog(eval_log)], num_boost_round=100)

    return eval_log[0]['validation']['logloss'][-1]
study_tfidf_ngram = optuna.create_study(direction="minimize")
study_tfidf_ngram.optimize(tfidf_objective_ngram, n_trials=30)
print(study_tfidf_ngram.best_params)
print("Best Log-Loss from TFIDF N-Gram Embeddings: " + str(study_tfidf_ngram.best_value))

#Bag of Words Embeddings

bagofwords_vectorizer = CountVectorizer(analyzer='word', token_pattern=r'\w{1,}')
bagofwords_vectorizer.fit(pd.concat((train_df['question1'],train_df['question2'])).unique())
y_BOW = train_df['is_duplicate'].values
X_BOW = scipy.sparse.hstack((bagofwords_vectorizer.transform(train_df['question1'].values),bagofwords_vectorizer.transform(train_df['question2'].values)))
def objective_BOW(trial):
    global X_BOW, y_BOW

    X_train, X_test, y_train, y_test = train_test_split(X_BOW, y_BOW, test_size=0.2)
    dtrain = xgb.DMatrix(X_train, label=y_train)
    dtest = xgb.DMatrix(X_test, label=y_test)
    eval_log = []

    params = {
    'booster': 'gbtree',
    'tree_method': 'gpu_hist',
    'max_depth': 10,
    'eta': 0.1,
    'objective': 'binary:logistic',
    'eval_metric': 'logloss',
    "lambda": trial.suggest_loguniform("lambda", 1e-8, 1.0),
    "alpha": trial.suggest_loguniform("alpha", 1e-8, 1.0),
    "max_depth": trial.suggest_int("max_depth", 1, 9),
    "eta" : trial.suggest_loguniform("eta", 1e-8, 1.0),
    "gamma": trial.suggest_loguniform("gamma", 1e-8, 1.0),
    }

    # Add a callback for pruning.
    pruning_callback = optuna.integration.XGBoostPruningCallback(trial, "validation-logloss")
    bst = xgb.train(params, dtrain, evals=[(dtest, "validation")], callbacks=[pruning_callback, getEvalLog(eval_log)], num_boost_round=100)

    return eval_log[0]['validation']['logloss'][-1]
study_BOW = optuna.create_study(direction="minimize")
study_BOW.optimize(objective_BOW, n_trials=30)
print(study_BOW.best_params)
print("Best Log-Loss from Bag of Words Embeddings: " + str(study_BOW.best_value))