import math
import random
import os

from own.saving import make_dirs

def save_train_test_split(trainset, testset):
    directory = os.path.join("data", "sets")
    make_dirs(directory)
    with open (os.path.join(directory,"trainset_rids.txt"), "w", encoding = "utf-8") as f:
        f.writelines([(str(item)+"\n") for item in trainset])
    print("Saved {} RIDs for the Trainset successfully".format(len(trainset)))
    with open (os.path.join(directory,"testset_rids.txt"), "w", encoding = "utf-8") as f:
        f.writelines([(str(item)+"\n") for item in testset])
    print("Saved {} RIDs for the Testset successfully".format(len(testset)))


def stratified_test_train_split(RIDs, ratings, test_ratio):
    # separating pos and neg reviews
    positives = [RID for RID, rating in zip(RIDs, ratings) if rating == 1]
    negatives = [RID for RID, rating in zip(RIDs, ratings) if rating == 0]
    total_reviews = len(positives) + len(negatives)
    print("total reviews: {}\n positives: {}\n negatives: {}".format(total_reviews, len(positives), len(negatives)))

    # calculating size of train- and testset
    # test_ratio = 0.1
    test_size_positives = math.ceil(test_ratio * len(positives))
    test_size_negatives = math.floor(test_ratio * len(negatives))


    test_size = test_size_positives + test_size_negatives
    print("\ntest_ratio: {} \n Test Size Positives: {} \n Test Size Negatives: {} \n Total Testset Size: {} -> {}%".format(test_ratio,test_size_positives, test_size_negatives,test_size, round(test_size/total_reviews*100,4)))

    # drawing random samples von test_size from positives and negatives
    random.seed(30)
    testset_negatives = random.sample(list(negatives), test_size_negatives)
    random.seed(30)
    testset_positives = random.sample(list(positives), test_size_positives)
    trainset_negatives = [RID for RID in negatives if RID not in testset_negatives]
    trainset_positives = [RID for RID in positives if RID not in testset_positives]

    testset = testset_positives + testset_negatives
    trainset = trainset_negatives + trainset_positives

    save_train_test_split(trainset, testset)

    assert len(testset) == test_size , "test_size != len(testset)"
    assert (len(trainset_negatives) + len(testset_negatives)) == len(negatives), "len(trainset_negatives) + len(testset_negatives) != len(negatives)"
    assert (len(trainset_positives) + len(testset_positives)) == len(positives), "len(trainset_positives) + len(testset_positives) != len(positives)"
    
    return trainset, testset


def get_matching_reviews(RID_list, review_list, searched_RID_list):
    matching_reviews = [review for review, RID in zip(review_list, RID_list) if RID in searched_RID_list]
    matching_RIDs = [RID for review, RID in zip(review_list, RID_list) if RID in searched_RID_list]
    print("Found {} of {} seached results".format(len(matching_RIDs), len(searched_RID_list)))
    return matching_reviews, matching_RIDs