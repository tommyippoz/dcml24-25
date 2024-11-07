import numpy
import pandas
import time

# Paths to datasets. If you unpack the ZIPfile in the same folder as the python script you wont need to change them
import sklearn
from pyod.models.hbos import HBOS
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import train_test_split

TRAIN_NAME = "dataset_contest/UNSW_students.csv"
TEST_NAME = "dataset_contest/UNSW_Test.csv"
TEST_UNK = "dataset_contest/UNSW_ZeroDay.csv"


# ---------------------------------- SUPPORT METHODS --------------------------------------------

def read_csv_binary_dataset(dataset_name: str, label_name: str = "multilabel", exclude_categorical: bool = True,
                            normal_tag: str = "normal", limit: float = numpy.nan):
    """
    Method to process an input dataset as CSV
    :param exclude_categorical: True if non-float features have to be excluded
    :param normal_tag: tag that identifies normal data
    :param limit: integer to cut dataset if needed.
    :param dataset_name: name of the file (CSV) containing the dataset
    :param label_name: name of the feature containing the label
    :return: feature set (x), labels (y), feature names, perc of anomalies
    """
    # Loading Dataset
    df = pandas.read_csv(dataset_name)

    # Preprocess
    df = df.fillna(0)
    df = df.replace('null', 0)
    df = df[df.columns[df.nunique() > 1]]

    # Testing Purposes
    if (numpy.isfinite(limit)) & (limit < len(df.index)):
        df = df[0:limit]

    # Binarize label
    if label_name in df.columns:
        y_enc = numpy.where(df[label_name] == normal_tag, 0, 1)

        # Basic Pre-Processing
        normal_frame = df.loc[df[label_name] == "normal"]
        print("\nDataset '" + dataset_name + "' loaded: " + str(len(df.index)) + " items, " + str(
            len(normal_frame.index)) + " normal and 2 labels")
        an_perc = (y_enc == 1).sum() / len(y_enc)
        x = df.drop(columns=[label_name])
    else:
        x = df

    x_no_cat = x.select_dtypes(exclude=['object']) if exclude_categorical else x
    feature_list = x_no_cat.columns

    if label_name in df.columns:
        return x_no_cat.to_numpy(), y_enc, feature_list, an_perc
    else:
        return x_no_cat.to_numpy(), feature_list


def current_ms() -> int:
    """
    Reports the current time in milliseconds
    :return: long int
    """
    return round(time.time() * 1000)


def test_and_save(dataset_name: str, clf_model, student_name: str, tag: str):
    """
    Function to be used to test a classifier model on a test set
    :param dataset_name: name of the dataset (path)
    :param clf_model: trained model to test
    :param student_name: name of the student (to rename the output file)
    :param tag: an additional tag to the filename
    :return:
    """
    x_test, feature_list = read_csv_binary_dataset(dataset_name=dataset_name)
    if clf_model is not None:
        y_pred = clf_model.predict(x_test)
        filename = student_name + '@' + tag + '.csv'
        pandas.DataFrame(y_pred, columns=['predictions']).to_csv(filename, index=False)
        print("File '%s' printed" % filename)
    else:
        print('Model is not valid')


def score_model(clf_model, student_name):
    """
    Function that prints results to be sent to the teacher via email.
    :param clf_model: model to use for scoring
    :param student_name: name of the student
    :return: nothing, prints 2 files
    """
    # General Test Set
    test_and_save(TEST_NAME, clf_model, student_name, 'general')

    # Test Set with normal + unknowns
    test_and_save(TEST_UNK, clf_model, student_name, 'zeroday')

    print("Remember to send both files to tommaso.zoppi@unifi.it, mail tag [DCML24] Context <your_name_here>")


if __name__ == "__main__":
    # General vars
    STUDENT_NAME = 'zoppi'
    MY_CLF = None

    # Loading dataset
    x, y, feature_names, anomaly_perc = read_csv_binary_dataset(TRAIN_NAME)

    # Your code here
    # split dataset
    train_data, test_data, train_label, test_label = train_test_split(x, y, test_size=0.5)

    # choose, train and test classifier from PYOD
    MY_CLF = HBOS(contamination=anomaly_perc, n_bins=20)
    MY_CLF.fit(train_data)
    predicted_labels = MY_CLF.predict(test_data)

    # Computing metrics to understand how good an algorithm is
    accuracy = sklearn.metrics.accuracy_score(test_label, predicted_labels)
    tn, fp, fn, tp = confusion_matrix(test_label, predicted_labels).ravel()
    print("%s: Accuracy is %.4f, TP: %d, TN: %d, FN: %d, FP: %d" % (
        MY_CLF.__class__.__name__, accuracy,  tp, tn, fn, fp))

    # Generating files
    # To be uncommented whenever test files will be available
    # score_model(MY_CLF, STUDENT_NAME)
