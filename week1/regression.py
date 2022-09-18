import os

import pandas
import matplotlib.pyplot as plt

BASE_DIR = os.path.abspath(os.path.join(os.getcwd(), os.pardir))

diabetes_df = pandas.read_csv(os.path.join(BASE_DIR, "week1/dataset/diabetes.csv"))


# Creating feature and target arrays
X = diabetes_df.drop("Glucose", axis=1).values
y = diabetes_df["Glucose"].values

# Making predictions from a single feature
X_bmi = X[:, 3]
# TODO: 2 dimensional array is required, but what does -1 and 1 means in the reshape method? check the reshape method
X_bmi = X_bmi.reshape(-1, 1)

# Plotting glucose vs. body mass index
plt.scatter(X_bmi, y)
plt.ylabel("Blood Glucose (mg/dl)")
plt.xlabel("Body Mass Index")


if __name__ == '__main__':
    print(diabetes_df.head())           # TODO: what is head method

    # print types of X and y
    print(f"X = {type(X)}, y = {type(y)}")

    # print
    print(y.shape, X_bmi.shape)
    print(X_bmi.shape)
    plt.show()


