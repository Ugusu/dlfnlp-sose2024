import os
import pandas as pd

# check if dev_dataset not in the data folder
if not os.path.exists("data/etpc-paraphrase-dev.csv"):
    print("Split the dataset into train and dev set.")
    # read the dataset from "data/etpc-paraphrase-train.csv", sep="\t"

    train_dataset = pd.read_csv("data/etpc-paraphrase-train.csv", sep="\t")

    # take 0.2 of the train_dataset as dev_dataset randomly and shuffle both
    dev_dataset = train_dataset.sample(frac=0.2, random_state=42)
    train_dataset = train_dataset.drop(dev_dataset.index)

    print (len(dev_dataset), len(train_dataset))

    # save in "data/etpc-paraphrase-dev.csv", sep="\t"
    dev_dataset.to_csv("data/etpc-paraphrase-dev.csv", sep="\t", index=False)
    print("Split the dataset into train and dev set.")

    # save in "data/etpc-paraphrase-train.csv", sep="\t"
    train_dataset.to_csv("data/etpc-paraphrase-train.csv", sep="\t", index=False)
    print("Split the dataset into train and dev set.")
