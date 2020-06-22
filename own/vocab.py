from collections import Counter
import os
import itertools
#abspath = os.path.abspath(__file__)
#dname = os.path.dirname(abspath)
#os.chdir(dname)


from own.loading import load_reviews_and_rids
from own.saving import make_dirs

##file_path = os.path.join("..","data", "reviews", "processed_testset.txt")
#review_list, RID_list = load_reviews_and_rids(file_path)

def create_vocab(review_list):
    vocab = Counter()
    token_lists = []
    for review in review_list:
        token_lists.append([sentence.split(" ") for sentence in review])                                    # split sentences into words
    token_list = list(itertools.chain.from_iterable(list(itertools.chain.from_iterable(token_lists))))      # extract word in one list
    vocab.update(token_list)                                                                                # update vocabulary
    return vocab

def define_min_occurrence(vocab, min_occurrence = 2):
    print("Defining min_occurence")
    tokens = [k for k,c in vocab.items() if c >= min_occurrence]                                            # Selecting words over min_occurence
    print(" Vocab length before truncating: {}\n Vocab length after truncating: {}".format(len(vocab), len(tokens)))
    return tokens


def save_vocab(directory, file_name, tokens):
    make_dirs(directory)
    data = "\n".join(tokens)    
    file_path = os.path.join(directory, file_name + ".txt")

    with open(file_path, "w", encoding = "utf-8") as f:
        f.write(data)
    print("File {} saved successfully".format(file_name))


def load_vocab(file_path):
    with open(file_path, "r", encoding = "utf-8") as f:
        return f.read().split("\n")