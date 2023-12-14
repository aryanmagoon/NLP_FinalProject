import matplotlib.pyplot as plt


models = ['TF-IDF Word', 'Bag of Words', 'Custom Embeddings', 'TF-IDF Char', 'TF-IDF ngram']
logloss = [0.49408566783432223, 0.47856332066775586, 0.3273114265075499,0.46981763319679787, 0.48349765260936056]  # log loss input by hand

plt.figure(figsize=(10, 6))
plt.bar(models, logloss, color=['blue', 'green', 'red', 'black', 'orange'])
plt.xlabel('Embeddings')
plt.ylabel('Validation Log Loss')
plt.title('Final Validation Log Loss of XGBoost Models')
plt.ylim(0, 1) #we didnt' get log loss greater than 1
plt.show()

training_logloss = [.09, .0628, .168, .3048, .2759, .1995]
validation_logloss = [.31, .4222, .349, .31287, .290721, .278074]  #training and validation logloss input by hand

import numpy as np
bar_width = 0.35
models = ['trial ' +str(i) for i in range(1,7)]
index = np.arange(len(models))
plt.figure(figsize=(12, 6))
bar1 = plt.bar(index, training_logloss, bar_width, label='Training Log Loss', color='b')
bar2 = plt.bar(index + bar_width, validation_logloss, bar_width, label='Validation Log Loss', color='g')

plt.xlabel('Trials')
plt.ylabel('Training and Validation Log Loss')
plt.title('Training and Validation Accuracies of Fine-Tuned BERT Models')
plt.xticks(index + bar_width / 2, models)
plt.xticks(rotation=90)
plt.legend()
plt.ylim(0, 1)  # Did not receive log loss greater than 1

plt.show()