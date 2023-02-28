import math
import string
import nltk
from nltk.stem import WordNetLemmatizer
from nltk.book import FreqDist
import csv
import ast
import pandas as pd


def write_top100_to_csv(score_dict, file_name):
    output_list = []
    for key, value in score_dict.items():
        sorted_list = sorted(value, key=lambda x: (x[2], x[1]), reverse=True)
        output_list.extend(sorted_list[0:100])

    with open(file_name, "w", newline='') as f:
        writer = csv.writer(f)
        for value in output_list:
            writer.writerow(value)


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


def compute_query_likelihood_laplace(query_word_list, passage, class_num, epsilon):
    passage_word_list = preprocessing(passage)
    passage_count_dict = FreqDist(passage_word_list)
    passage_total_words = len(passage_word_list)
    query_likelihood = 1.
    for word in query_word_list:
        if word in passage_count_dict:
            freq = passage_count_dict[word]
        else:
            freq = 0
        likelihood = (freq + epsilon) / (passage_total_words + epsilon*class_num)
        query_likelihood = query_likelihood * likelihood
    return query_likelihood


def retrieve_laplace(query_df, total_df, epsilon):
    score_dict = {}
    total_passages = total_df.shape[0]
    for index, row in query_df.iterrows():
        print(index, '/199')
        qid = row['qid']
        score_dict[qid] = []
        query = row['query']
        query_word_list = preprocessing(query)
        thousand_candidates = total_df[total_df['qid'] == qid]
        for index, row in thousand_candidates.iterrows():
            pid = row['pid']
            passage = row['passage']
            score = compute_query_likelihood_laplace(query_word_list, passage, total_passages, epsilon)
            score_dict[qid].append((qid, pid, score))

    return score_dict


def compute_query_likelihood_dirichlet(query_word_list, passage, total_word_count_dict, mu, n_total_words_corpus):
    query_likelihood_dirichlet = []
    passage_word_list = preprocessing(passage)
    len_passage = len(passage_word_list)
    passage_word_dict = FreqDist(passage_word_list)
    for word in query_word_list:
        if word not in total_word_count_dict:
            continue
        if word not in passage_word_dict:
            query_likeli = math.log((mu/(len_passage+mu))*(total_word_count_dict[word]/n_total_words_corpus))
        else:
            formula1 = (len_passage/(len_passage+mu))*(passage_word_dict[word]/len_passage)
            formula2 = (mu/(len_passage+mu))*(total_word_count_dict[word]/n_total_words_corpus)
            sum_ = formula1 + formula2
            query_likeli = math.log(sum_)
        query_likelihood_dirichlet.append(query_likeli)

    return sum(query_likelihood_dirichlet)


def retrieve_dirichlet(query_df, total_df, total_words_count_dict, mu, n_total_words_corpus):
    score_dict = {}
    for index, row in query_df.iterrows():
        print(index, '/199')
        qid = row['qid']
        score_dict[qid] = []
        query = row['query']
        query_word_list = preprocessing(query)
        thousand_candidates = total_df[total_df['qid'] == qid]
        for index, row in thousand_candidates.iterrows():
            pid = row['pid']
            passage = row['passage']
            score = compute_query_likelihood_dirichlet(query_word_list, passage, total_words_count_dict, mu, n_total_words_corpus)
            score_dict[qid].append((qid, pid, score))

    return score_dict


if __name__ == '__main__':
    qid_query_df = pd.read_csv('test-queries.tsv', sep='\t', header=None, names=['qid', 'query'])
    candidate_passages_top1000 = pd.read_csv('candidate-passages-top1000.tsv', sep='\t',header=None,names=['qid', 'pid', 'query', 'passage'])

    # Laplace smoothing
    laplace_score_dict = retrieve_laplace(qid_query_df, candidate_passages_top1000, 1.)
    write_top100_to_csv(laplace_score_dict, 'laplace.csv')

    # Lidstone correction with ϵ = 0.1
    laplace_with_lidstone_correction_dict = retrieve_laplace(qid_query_df, candidate_passages_top1000, 0.1)
    write_top100_to_csv(laplace_with_lidstone_correction_dict, 'lidstone.csv')

    with open('passage-collection.txt', encoding='utf-8') as f:
        text = f.read()
    # Data preprocessing
    data_single_word = preprocessing(text)

    corpus_count_dict = FreqDist(data_single_word)
    corpus_total_words = len(data_single_word)

    with open('remove_stop_word.txt', encoding='utf-8') as f:
        data_list = ast.literal_eval(f.read())
    remove_one_digit = list(data_list)

    corpus_count_dict_without_stopwords = {}
    for word in remove_one_digit:
        corpus_count_dict_without_stopwords[word] = corpus_count_dict[word]

    # Dirichlet smoothing with µ = 50
    dirichlet_score_dict = retrieve_dirichlet(qid_query_df, candidate_passages_top1000, corpus_count_dict_without_stopwords, 50, corpus_total_words)
    write_top100_to_csv(dirichlet_score_dict, 'dirichlet.csv')



