import math
import pandas as pd
import nltk
from nltk.stem import WordNetLemmatizer
import string
import numpy as np
from nltk.book import FreqDist
from numpy.linalg import norm
import csv
import ast


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


# extract the top 100 similar passages for each query and write into csv file
def write_top100_to_csv(score_dict, file_name):
    output_list = []
    for key, value in score_dict.items():
        sorted_list = sorted(value, key=lambda x: (x[2], x[1]), reverse=True)
        output_list.extend(sorted_list[0:100])

    with open(file_name, "w", newline='') as f:
        writer = csv.writer(f)
        for value in output_list:
            writer.writerow(value)


# ----------------------- functions of tfidf score ----------------------------

# construct idf dict for each term in vocabulary
def construct_idf_dict(total_words_list, invert_index, N):
    idf = dict.fromkeys(total_words_list, 0.)
    for term in idf:
        n = len(invert_index[term])
        idf[term] = math.log10((N / n))
    return idf


# compute tfidf representation for one passage
def single_passage_tfidf_representation(pid, total_massage_df, idf_dic, vocabulary):
    tfidf_dic = dict.fromkeys(vocabulary, 0.)
    passage = total_massage_df[total_massage_df['pid'] == pid]['passage'].get(0)
    processed_passage = preprocessing(passage)
    word_count_dict = FreqDist(processed_passage)
    for word in word_count_dict:
        if word in idf_dic:
            tfidf_dic[word] = word_count_dict[word] * idf_dic[word]
    return pd.DataFrame([tfidf_dic])


# compute tfidf representation for one passage
def tf_idf_representation_query(query_df, idf_dic, vocabulary):
    tfidf_dic_list = []
    for index, row in query_df.iterrows():
        query = row['query']
        tfidf_dic = dict.fromkeys(vocabulary, 0.)
        processed_query = preprocessing(query)
        word_count_dict = FreqDist(processed_query)
        for word in word_count_dict:
            if word in idf_dic:
                tfidf_dic[word] = word_count_dict[word] * idf_dic[word]
        tfidf_dic_list.append(tfidf_dic)
    query_tfidf_df = pd.DataFrame(tfidf_dic_list, index=query_df['qid'].to_list())
    return query_tfidf_df


def calculate_norm(wordcount_dict, idf_):
    tfidf_list = []
    for word, count in wordcount_dict.items():
        if word in idf_:
            tfidf = count * idf_[word]
            tfidf_list.append(tfidf)
    return norm(tfidf_list)


# Compute tf-idf score between each query and its candidate passages
# I find non-zero dimensions for query and passage to reduce the time complexity of the algorithm
def retrieve_with_tfidf(query_df, idf_dic, top1000_df):
    # format of the dict {qid:[(qid, pid1, score1),(qid, pid2, score2),....,(qid, pid1000, score1000)]}
    tfidf_score_dict = {}
    print('Retrieve in tf-idf method')
    for index, row in query_df.iterrows():
        print(index, '/199')
        qid = row['qid']
        tfidf_score_dict[qid] = []
        query = row['query']
        query_processed = preprocessing(query)
        query_wordcount_dict = FreqDist(query_processed)
        query_norm = calculate_norm(query_wordcount_dict, idf_dic)
        thousand_candidates = top1000_df[top1000_df['qid'] == qid]
        for index_, row_ in thousand_candidates.iterrows():
            pid = row_['pid']
            passage = row_['passage']
            passage_processed = preprocessing(passage)
            passage_processed_wordcount_dict = FreqDist(passage_processed)
            passage_norm = calculate_norm(passage_processed_wordcount_dict, idf_dic)
            dot_list = []
            for word in query_wordcount_dict:
                if (word in passage_processed_wordcount_dict) & (word in idf_dic):
                    tfidf = query_wordcount_dict[word] * idf_dic[word] + passage_processed_wordcount_dict[word] * \
                            idf_dic[word]
                    dot_list.append(tfidf)
            inner_product = sum(dot_list)
            # cosine similarity
            score = inner_product / (query_norm * passage_norm)
            tfidf_score_dict[qid].append((qid, pid, score))

    return tfidf_score_dict


# ----------------------- functions of bm25 score ----------------------------

def constructed_idf_bm25(total_words_list, invert_index, N):
    idf = dict.fromkeys(total_words_list, 0.)
    for term in idf:
        n = len(invert_index[term])
        idf[term] = math.log10((N - n + 0.5) / (n + 0.5))
    return idf


# calculate similarity between a single query and a single passage
def bm25(query, bm_idf_dict, passage, k1, k2, b, ave):
    bm_list = []
    query_word_list = preprocessing(query)
    passage_word_list = preprocessing(passage)
    len_passage = len(passage_word_list)
    ratio = len_passage/ave
    for word in query_word_list:
        if (word not in bm_idf_dict) | (word not in passage_word_list):
            continue
        W = bm_idf_dict[word]
        count_dict_passage = FreqDist(passage_word_list)
        K = k1 * (1-b+b*ratio)
        sqid = ((k1+1)*count_dict_passage[word]) / (K + count_dict_passage[word])
        count_dict_dict = FreqDist(query_word_list)
        sqiq = ((k2+1)*count_dict_dict[word]) / (k2+count_dict_dict[word])
        one_word_relevance = W*sqid*sqiq
        bm_list.append(one_word_relevance)

    return sum(bm_list)


def retrieve_with_bm25(query_df, top1000_df, bm25_idf_dic, ave):
    print('retrieve in bm25 method')
    bm25_score_dict = {}
    for index, row in query_df.iterrows():
        print(index, '/199')
        qid = row['qid']
        bm25_score_dict[qid] = []
        query = row['query']
        thousand_candidates = top1000_df[top1000_df['qid'] == qid]
        for index, row in thousand_candidates.iterrows():
            pid = row['pid']
            passage = row['passage']
            bm25_score = bm25(query, bm25_idf_dic, passage, 1.2, 100, 0.75, ave)
            bm25_score_dict[qid].append((qid, pid, bm25_score))

    return bm25_score_dict


def average_length(total_passage_df):
    # List stores the number of words for each passage
    length_list = []
    for index, row in total_passage_df.iterrows():
        passage = row['passage']
        passage_len = len(preprocessing(passage))
        length_list.append(passage_len)
    # Average length of the passages
    ave_length = sum(length_list) / total_passage_df.shape[0]
    return ave_length


if __name__ == '__main__':
    with open('remove_stop_word.txt', encoding='utf-8') as f:
        data_list = ast.literal_eval(f.read())
    remove_one_digit = list(data_list)

    Inverted_index = np.load("inverted_index.npy", allow_pickle=True).item()

    # Read tsv file
    candidate_passages_top1000 = pd.read_csv('candidate-passages-top1000.tsv', sep='\t', header=None,
                                             names=['qid', 'pid', 'query', 'passage'])
    total_passage_remove_duplicate = candidate_passages_top1000.drop_duplicates('passage')

    n_total_passages = total_passage_remove_duplicate.shape[0]

    # store idf values for each term
    idf_dict = construct_idf_dict(remove_one_digit, Inverted_index, n_total_passages)

    qid_query_df = pd.read_csv('test-queries.tsv', sep='\t', header=None, names=['qid', 'query'])

    '''
    Extract the tfidf representation of passage.
    The tfidf representations of all passages are too large, the dimension is 182469 x 98319, I can't store it in my 
    computer. So that I write a method to compute just one passage, you can pass the pid of a passage as the first parameter.
    '''
    single_passage_tfidf_representation = single_passage_tfidf_representation(7130104, total_passage_remove_duplicate,idf_dict, remove_one_digit)
    print(single_passage_tfidf_representation)

    '''
    I am able to get the tfidf representation of all queries because dimension is only 200 x 98319.
    Extract the tfidf representation of all queries
    '''
    query_tfidf_representation = tf_idf_representation_query(qid_query_df, idf_dict, remove_one_digit)
    print(query_tfidf_representation)

    # retrieve by tfidf value
    tfidf_score_dict = retrieve_with_tfidf(qid_query_df, idf_dict, candidate_passages_top1000)
    write_top100_to_csv(tfidf_score_dict, 'tfidf.csv')

    bm25_idf_dict = constructed_idf_bm25(remove_one_digit, Inverted_index, n_total_passages)
    ave_length = average_length(total_passage_remove_duplicate)

    # retrieve by bm25
    bm25_score_dict = retrieve_with_bm25(qid_query_df, candidate_passages_top1000, bm25_idf_dict, ave_length)
    write_top100_to_csv(bm25_score_dict, "bm25.csv")


