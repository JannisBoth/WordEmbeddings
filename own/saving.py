import os
import nltk
nltk.download('punkt')
import numpy as np
import pandas as pd


sent_tokenizer = nltk.data.load("tokenizers/punkt/english.pickle")

def make_dirs(directory):
    try:
        os.makedirs(directory)
        print("Directory " , directory ,  " successfully created ") 
    except FileExistsError:
        print("Directory " , directory ,  " already exists")



def save_data_to_file(file_path, data):
    with open(file_path, "w", encoding ="utf-8") as f:
        f.write(data)


def save_reviews_to_file(RID_list, review_list, file_name):
    directory = os.path.join("data","reviews")
    assert len(RID_list) == len(review_list), "lengths of RID_list({}) and review_list({}) doesn't match".format(len(RID_list), len(review_list))
    
    data_list = []
    
    make_dirs(directory)
    file_path = os.path.join(directory, file_name)
    
    for RID, review in zip(RID_list, review_list):
        cur_data = ["Here_starts_the_review "+str(RID)] + sent_tokenizer.tokenize(review)       # Sign that a new review starts -> relevant for loading them back in
        data_list = data_list + cur_data
    
    data = "\n".join(data_list)
    save_data_to_file(file_path, data)
    print("File saved successfully")



def save_RID_and_rating(RID_list, rating_list, file_name):
    directory = os.path.join("data", "rating")
    make_dirs(directory)
    file_path = os.path.join(directory, file_name)  
    
    rating_array = np.array([1 if item == "positive" else 0 for item in rating_list])
    df = pd.DataFrame(list(zip(RID_list, rating_array)), columns = ["RID", "rating"])
    df.to_csv(file_path, index = False)
    print("File saved successfully")


