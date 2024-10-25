from pandas import read_csv
from sklearn.model_selection import train_test_split

if __name__ == "__main__":
    """
    Main of the data analysis
    """

    #0 load dataset PANDAS / NUMPY
    my_dataset = read_csv("./labelled_dataset.csv")
    label_obj = my_dataset["label"]
    data_obj = my_dataset.drop(columns=["label"])

    #1 split dataset
    train_data, test_data, train_label, test_label = \
        train_test_split(data_obj, label_obj, test_size=0.5)

    #2 choose classifier SCIKIT LEARN

    #3 train classifier

    #4 test classifier

    a = 1