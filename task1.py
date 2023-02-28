import string
import nltk
from nltk.stem import WordNetLemmatizer
from matplotlib import pyplot as plt
import numpy as np
nltk.download('words')
nltk.download('wordnet')


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


def count_terms(word_list):
    count_dict = {}
    for word in word_list:
        if word not in count_dict:
            count_dict[word] = 1
        else:
            count_dict[word] += 1
    return count_dict


def proba_zipf(k, s, N):

    return (float(k))**((-1)*s)/N


if __name__ == '__main__':
    # Read data
    with open('passage-collection.txt', encoding='utf-8') as f:
        text = f.read()
    # Data preprocessing
    data_single_word = preprocessing(text)
    # Count terms
    word_count_dict = count_terms(data_single_word)
    vocabulary = list(word_count_dict.keys())
    print("The the size of the identified index of terms is ", len(vocabulary))

    f = open("vocabulary.txt", "w", encoding='utf-8')
    f.write(str(vocabulary))
    f.close()

    # The total number of words in corpus
    n_total_words = len(data_single_word)

    # The format of 'sorted_list_with_tuple' is [('the',27215),('and',5861),...]
    sorted_list_with_tuple = sorted(word_count_dict.items(), key=lambda x: x[1], reverse=True)

    # Compute the frequency of each word and store in 'word_frequency_list'
    word_frequency_list = []
    for item in sorted_list_with_tuple:
        word_frequency_list.append((int(item[1]) / n_total_words))

    # Total number of unique terms in the corpus
    n_total_terms = len(vocabulary)

    # Value of x-coordinate
    x_plot = np.linspace(1, n_total_terms, n_total_terms)

    # Plot the normalised frequency against their frequency ranking
    plt.plot(x_plot, word_frequency_list, label='Passage-collection Data')
    plt.legend()
    plt.savefig('figure1.pdf', bbox_inches='tight')
    plt.show()

    i = 1
    zipf_denominator = 0.0
    while i <= n_total_terms:
        zipf_denominator += i**(-1)
        i += 1

    y_plot_zipf = []
    for x in x_plot:
        y_plot_zipf.append(proba_zipf(x, 1, zipf_denominator))

    plt.loglog(x_plot, word_frequency_list, label='Corpus Data')
    plt.loglog(x_plot, y_plot_zipf, label="Zipf's law")
    plt.legend()
    plt.savefig('figure2.pdf', bbox_inches='tight')
    plt.show()
