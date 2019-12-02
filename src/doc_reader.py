"""
Functions to read documents from disk, from data set and from pickle.
"""

from os import listdir
from xml.dom import minidom
import pickle

import numpy as np

SENTENCE_TAG = "sentence"
TOKEN_TAG = "token"
TOKEN_STR_TAG = "word"

TOKEN_SEPARATOR = " "
SENTENCE_SEPARATOR = ". "

TEST_DOCUMENT_PATH = "files/simple_test_doc.txt"
DATASET_FOLDER = "dataset/"
TRAIN_FOLDER = "train/"
TEST_FOLDER = "test/"

DATASET_PICKLE_PATH = "files/dataset_pickle.p"
TRAIN_DOCS_KEY = 0
TEST_DOCS_KEY = 1


def read_doc_from_disk(doc_path: str) -> str:
    with open(doc_path, "r") as file:
        return file.read()


def read_folder_docs(doc_paths: list) -> dict:
        folder_corpus = {}

        # for each xml file
        for xml_file_path in doc_paths:
            xml_doc = minidom.parse(xml_file_path)
            doc_senteces = []
            doc_id = xml_file_path.split(".")[0].split("/")[-1]
            sentence_elems = xml_doc.getElementsByTagName(SENTENCE_TAG)

            # for each sentence element
            for sentence_elem in sentence_elems:
                tokens = sentence_elem.getElementsByTagName(TOKEN_TAG)
                sentence_tokens = []

                # for each token element
                for token_elem in tokens:
                    token_str = token_elem.getElementsByTagName(TOKEN_STR_TAG)[0].firstChild.data
                    sentence_tokens.append(token_str)
            
                doc_senteces.append(TOKEN_SEPARATOR.join(sentence_tokens))
            folder_corpus[doc_id] = SENTENCE_SEPARATOR.join(doc_senteces)

        return folder_corpus


def read_dataset(dataset_path: str, has_test_folder: bool) -> tuple:
    # train documents
    train_files = np.array(listdir(dataset_path + TRAIN_FOLDER), dtype=object)
    train_docs = read_folder_docs(dataset_path + TRAIN_FOLDER + train_files)
    
    # test documents
    test_docs: dict
    if has_test_folder:
        test_files = np.array(listdir(dataset_path + TEST_FOLDER), dtype=object)
        test_docs = read_folder_docs(dataset_path + TEST_FOLDER + test_files)
    
    return train_docs, test_docs


def save_corpus(file_path: str, corpus: dict) -> None:
    with open(file_path, "wb") as file:
        pickle.dump(corpus, file, protocol=pickle.HIGHEST_PROTOCOL)


def read_dataset_from_pickle(pickle_path: str) -> tuple:
    with open(pickle_path, "rb") as file:
        dataset_dict = pickle.load(file)
        train_docs = dataset_dict[TRAIN_DOCS_KEY]
        test_docs = dataset_dict[TEST_DOCS_KEY]
        return train_docs, test_docs


if __name__ == "__main__":
    # read document from disk
    disk_doc = read_doc_from_disk(TEST_DOCUMENT_PATH)
    print("Document for simple test: ", disk_doc)
    print()

    # read dataset
    train_docs, test_docs = read_dataset(DATASET_FOLDER, has_test_folder=True)
    print("Number of train docs: ", len(train_docs))
    print("Number of test docs: ", len(test_docs))

    print()
    print("Example train document: ", train_docs["100"])

    print()
    print("Example test document: ", test_docs["193"])

    # save dataset into pickle
    save_corpus(DATASET_PICKLE_PATH, {TRAIN_DOCS_KEY: train_docs, TEST_DOCS_KEY: test_docs})

    # read docs from pickle and compare with written
    p_train_docs, p_test_docs = read_dataset_from_pickle(DATASET_PICKLE_PATH)
    print()
    print("Reading saved in pickle...")
    print("Train documents are equal: ", train_docs == p_train_docs)
    print("Test documents are equal: ", test_docs == p_test_docs)
