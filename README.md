# Emotions-Generator
A custom emotion generating Bigram LM trained on a twitter emotion dataset available on Hugging Face. Modified the standard implementation of Bigram LM to generate emotion-oriented sentences for 6 emotions: 'joy', 'love', 'sadness', 'surprise', 'fear', 'anger'. Acheived an accuracy of 0.74 and macro-F1 score of 0.69 during an extrinsic evaluation of the LM by training a SVC model and conducting Grid Search to find out the best parameters.
