import nltk
import sys
import os
import string
import math

FILE_MATCHES = 2
SENTENCE_MATCHES = 2


def main():

    # Check command-line arguments
    if len(sys.argv) != 2:
        sys.exit("Usage: python questions.py corpus")

    # Calculate IDF values across files
    files = load_files(sys.argv[1])
    file_words = {
        filename: tokenize(files[filename])
        for filename in files
    }
    file_idfs = compute_idfs(file_words)

    # Prompt user for query
    query = set(tokenize(input("Query: ")))

    # Determine top file matches according to TF-IDF
    filenames = top_files(query, file_words, file_idfs, n=FILE_MATCHES)

    # Extract sentences from top files
    sentences = dict()
    for filename in filenames:
        for passage in files[filename].split("\n"):
            for sentence in nltk.sent_tokenize(passage):
                tokens = tokenize(sentence)
                if tokens:
                    sentences[sentence] = tokens

    # Compute IDF values across sentences
    idfs = compute_idfs(sentences)

    # Determine top sentence matches
    matches = top_sentences(query, sentences, idfs, n=SENTENCE_MATCHES)
    for match in matches:
        print(match)


def load_files(directory):
    """
    Given a directory name, return a dictionary mapping the filename of each
    `.txt` file inside that directory to the file's contents as a string.
    """

    my_dict = dict()

    for root, dirs, files in os.walk(directory):
        for name in files:
            dir_name = os.path.join(root, name)
            f = open(dir_name, "r", encoding="utf8")
            my_dict[name] = f.read()

    return my_dict


def tokenize(document):
    """
    Given a document (represented as a string), return a list of all of the
    words in that document, in order.

    Process document by coverting all words to lowercase, and removing any
    punctuation or English stopwords.
    """
    punct = string.punctuation
    stop_list = nltk.corpus.stopwords.words("english")

    words = nltk.word_tokenize(document.lower())
    for word in words:
        if word in punct or word in stop_list:
            words.remove(word)

    return words


def compute_idfs(documents):
    """
    Given a dictionary of `documents` that maps names of documents to a list
    of words, return a dictionary that maps words to their IDF values.

    Any word that appears in at least one of the documents should be in the
    resulting dictionary.
    """
    word_dict = dict()

    all_words = set()

    for key in documents:
        for word in documents[key]:
            all_words.add(word)

    total_docs = len(documents)

    for word in all_words:
        count = 0
        for token in documents.values():
            if word in token:
                count += 1
        word_dict[word] = math.log(total_docs/count)

    return word_dict


def top_files(query, files, idfs, n):
    """
    Given a `query` (a set of words), `files` (a dictionary mapping names of
    files to a list of their words), and `idfs` (a dictionary mapping words
    to their IDF values), return a list of the filenames of the the `n` top
    files that match the query, ranked according to tf-idf.
    """

    all_scores = dict()

    for key, value in files.items():
        num = 0
        for word in query:
            num_words = value.count(word)
            num += num_words * idfs[word]
        all_scores[key] = num

    all_scores = sorted(all_scores.items(), key=lambda x: x[1], reverse=True)

    ranked = []
    count = 0
    for item in all_scores:
        ranked.append(item[0])
        count += 1
        if count == n:
            break

    return ranked


def top_sentences(query, sentences, idfs, n):
    """
    Given a `query` (a set of words), `sentences` (a dictionary mapping
    sentences to a list of their words), and `idfs` (a dictionary mapping words
    to their IDF values), return a list of the `n` top sentences that match
    the query, ranked according to idf. If there are ties, preference should
    be given to sentences that have a higher query term density.
    """

    all_scores = dict()

    for sentence, word_list in sentences.items():
        num = 0
        num_in_query = 0
        for word in query:
            if word in word_list:
                num += idfs[word]
                num_in_query += 1

        total_words = len(word_list)
        term_density = num_in_query/total_words

        all_scores[sentence] = {
            'idf_value': num,
            'density': term_density
        }

    all_scores = sorted(all_scores.items(), key=lambda x: (x[1]['idf_value'], x[1]['density']), reverse=True)

    ranked = []
    count = 0
    for item in all_scores:
        ranked.append(item[0])
        count += 1
        if count == n:
            break

    return ranked


if __name__ == "__main__":
    main()
