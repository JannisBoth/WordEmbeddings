import pandas as pd
import os
import re

#Loading the Data
def load_data_frame(file_name):
    path_df = os.path.join("data", file_name)
    df = pd.read_csv(path_df, sep =',', encoding='utf-8')
    df = df.drop(["Unnamed: 0"], axis = 1)
    print("Data Frame loaded successfully")
    return df


def load_reviews_and_rids(file_path):
    review_list = []
    RID_list = []
    with open(file_path, "r", encoding = "utf-8") as f:
        cur_review = []
        for line in f:
            
            if bool(re.match(r'Here_starts_the_review', line)): #
                RID_list.append(int(line.split()[1]))
                review_list.append(cur_review)
                cur_review = []

            else:
                cur_review.append(line.replace("\n", ""))

        review_list.append(cur_review)
        assert len(review_list[1:(len(review_list)+1)]) == len(RID_list), "ERROR IN LOADING THE REVIEWS AND RID: lengths of RID_list({}) and review_list({}) doesn't match".format(len(RID_list), len(review_list))
        print("File loaded successfully")
        return review_list[1:(len(review_list)+1)], RID_list

    


def load_RID_and_rating():
    directory = os.path.join("data", "rating")
    file_path = os.path.join(directory, "rid_ratings.csv")
    df = pd.read_csv(file_path, sep =',')
    print("File loaded successfully")
    return df


def load_train_test_rid_lists():
    directory = os.path.join("data", "sets")
    train_rids = []
    test_rids = []
    with open (os.path.join(directory,"trainset_rids.txt"), "r", encoding = "utf-8") as f:
        train_rids = (f.read().split("\n"))
        train_rids = [int(RID) for RID in train_rids[0:(len(train_rids)-1)]]
    print("Loaded Trainset successfully")    
    with open (os.path.join(directory,"testset_rids.txt"), "r", encoding = "utf-8") as f:
        test_rids = (f.read().split("\n"))
        test_rids = [int(RID) for RID in test_rids[0:(len(test_rids)-1)]]
    print("Loaded Testset successfully")
    return train_rids, test_rids