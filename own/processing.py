# change wd to current file
import os
import re
import math

# Named Entity Detection
import spacy
nlp = spacy.load('en_core_web_sm')

# Word tokenization
from nltk.tokenize import TreebankWordTokenizer
word_tokenizer = TreebankWordTokenizer()

# Stop words
import nltk
nltk.download('stopwords')
from nltk.corpus import stopwords
stop_words = set(stopwords.words("english"))
stop_words.remove("not")



replacement_patterns = [
    (r"its", "it is"),
    (r"it's", "it is"),
    (r"you're", "your are"),
    (r"that's", "that is"),
    (r"'ve", " have"), #general
    (r"n't", "_not"), #general
    (r"'nt", "_not"), #general
    (r"'ll", " will"), #general
    (r"I'm", "i am"),
    (r"he's", "he is"), # also does she's
    (r"there's", "there is"),
    (r"'re", " are"),
    (r"let's", "let us"),
    (r"'d", " would"),
    (r" \.", "."),
    (r"here's", "here is"),
    (r"what's", "what is"),
    (r"some's", "someones"),
    (r"every's", "everyones"),
    (r"any's", "anyones"),
    (r"s'thing", "something"),
    (r"might'have", "might have"),
    (r"who's", "who is"),
    (r"is't" , "is it"),
    (r"did't", "did_not"),
    (r"\d", " "), # replace numbers with " ")
    (r"((?P<pre>\w+)([^a-zA-Z/_\s]+)(?P<suf>\w+))", r"\g<pre>\g<suf>"), # replace "all-star" with "allstar" but let "do_not" be "do_not"
    (r"(\w+)(\W+)(\w*)", r"\1 \3 "), # replace  "experience." with "experience" or "free****" with "free" or "horror/surviv" with "horror surviv"
    (r"(u\+)", ""), # replace rest of unicode characters
    (r"  ", " ")
]


from own.processing_classes import RegexReplacer
regex_replacer = RegexReplacer(replacement_patterns)

from own.processing_classes import SpellingReplacer
spelling_replacer = SpellingReplacer()

from own.processing_classes import LemmaStemmer
lemma_stemmer = LemmaStemmer()

from own.processing_classes import AntonymReplacer
negations_replacer = AntonymReplacer()

from own.saving import make_dirs


def named_entity_recognition(sentence):
    text_wo_entities = sentence
    sentence = nlp(sentence)
    entities = set([entity.text for entity in sentence.ents])

    for entity in set(entities):
        text_wo_entities = text_wo_entities.replace(entity, "")
    return text_wo_entities


def remove_non_alphas(word_list):
    non_alpha = re.compile(r"[^\W+]")    
    return list(filter(non_alpha.match, word_list))


def process_sentence(sentence):
    sentence_entities = named_entity_recognition(sentence)                                  # find and delete named entities
    sentence_regex = regex_replacer.replace(sentence_entities)                              # replace predefined patterns
    word_list = word_tokenizer.tokenize(sentence_regex)                                     # find and seperate word with boundaries
    word_list_alpha = remove_non_alphas(word_list)                                          # filters all alphanumeric characters -> no numbers, punctuation
    word_list_spelled = spelling_replacer.replace_list(word_list_alpha)                     # replace wrong spelled words through min_edit_distance
    word_list_stop_words = [word for word in word_list_spelled if word not in stop_words]   # filter non stop-words
    word_list_stemmed = lemma_stemmer.perform(word_list_stop_words)                         # check if and how the word can be lammatized, otherwise stem it
    word_list_negations = negations_replacer.replace_negations(word_list_stemmed)           # negate words which follow to "not" eg. "not nice" -> nasty
    return word_list_negations

# print(process_sentence("Hello world, I am not a nice dude didn't Barack Obama processsed loved."))


def process_review(sentence_list):
    return [process_sentence(sentence) for sentence in sentence_list]

# sentence_list = ["Hello world, I am not a nice dude didn't Barack Obama processsed loved.", "This is anothr sentence.", "this is not beatiful"]
# print(process_review(sentence_list))

def save_processed_review(dir_processed, RID, processed_review):
    doc = dir_processed + "//p_" + str(RID) + ".txt" 

    data = ""
    for sentence in processed_review:
        data = data + " ".join(sentence) + "\n"

    with open(doc, "w", encoding =" utf-8") as f:
        f.write(data)


from own.saving import save_data_to_file

def save_processed_reviews_to_file(RID_list, review_list, directory, file_name):
    assert len(RID_list) == len(review_list), "lengths of RID_list({}) and review_list({}) doesn't match".format(len(RID_list), len(review_list))
    
    data_list = []
    
    make_dirs(directory)
    file_path = os.path.join(directory, file_name)
    print("Saving", file_name)
    
    for RID, review in zip(RID_list, review_list):
        cur_data = ["Here_starts_the_review "+str(RID)] + review      # Sign that a new review starts -> relevant for loading them back in
        data_list = data_list + cur_data

    sentence_list = []
    for sentence in data_list:
        if isinstance(sentence, str):
            if sentence:
                sentence_list.append(sentence)
            
        else:
            sentence_list.append(" ".join(sentence))

    with open(file_path, "w", encoding = "utf-8") as f:
        f.writelines([(str(item)+"\n") for item in sentence_list])
    print("File saved successfully")



def process_set(set_name, matching_RIDs, matching_reviews):
    directory = os.path.join("data", "reviews")
    file_name = set_name+".txt"
    length = len(matching_RIDs)
    processed_reviews = []
    for i in range(length):
        print("{} of {} Reviews processed!".format(i, length), end = "\r")
        processed_reviews.append(process_review(matching_reviews[i]))

    save_processed_reviews_to_file(matching_RIDs, processed_reviews, directory, file_name)


