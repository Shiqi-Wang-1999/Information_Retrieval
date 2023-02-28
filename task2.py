import nltk
import string
from nltk.stem import WordNetLemmatizer
from nltk.corpus import stopwords
import pandas as pd
import ast
import numpy as np
from nltk.book import FreqDist
nltk.download('stopwords')


def preprocessing(text):
    # Tokenization
    words_list = nltk.word_tokenize(text)
    # Remove the word which is not alpha and change to lowercase
    words_list_without_symbol = [word.lower() for word in words_list if word.isalpha()]
    # Remove punctuation
    text_without_punc = [word.translate(str.maketrans('', '', string.punctuation)) for word in words_list_without_symbol]
    # Lemmatisation
    wordnetlemmatize = WordNetLemmatizer()
    lemma_list = [wordnetlemmatize.lemmatize(w) for w in text_without_punc]
    return lemma_list


def inverted_index(vocabulary_, data_):
    inverted_table = {}
    for word in vocabulary_:
        inverted_table[word] = []
    for index, row in data_.iterrows():
        print(index, "/189876")
        pid = row['pid']
        passage = row['passage']
        # data process
        processed_data = preprocessing(passage)
        passage_dict = FreqDist(processed_data)
        for word_ in passage_dict:
            if word_ in inverted_table:
                inverted_table[word_].append((pid, passage_dict[word_]))

    return inverted_table


if __name__ == '__main__':
    with open('vocabulary.txt', encoding='utf-8') as f:
        data_list = ast.literal_eval(f.read())

    vocabulary = list(data_list)

    # remove stop words
    vocabulary_without_stopwords = [word for word in vocabulary if word not in stopwords.words('english')]
    # remove word with only one digit
    remove_one_digit = [word for word in vocabulary_without_stopwords if (len(word) != 1)]

    f = open("remove_stop_word.txt", "w", encoding='utf-8')
    f.write(str(remove_one_digit))
    f.close()

    # Read tsv file
    candidate_passages_top1000 = pd.read_csv('candidate-passages-top1000.tsv', sep='\t', header=None,names=['qid', 'pid', 'query', 'passage'])
    remove_duplicate_pid_passage = candidate_passages_top1000[['pid', 'passage']].drop_duplicates('passage')
    # Generate inverted index
    Inverted_index = inverted_index(remove_one_digit, remove_duplicate_pid_passage)

    np.save("inverted_index.npy", Inverted_index)


