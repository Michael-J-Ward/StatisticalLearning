import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.linear_model import Ridge

# loading and preparing data
Credit = pd.read_csv('Data/Credit.csv')
Credit.drop('Unnamed: 0', axis=1)

X = Credit.drop('Balance', axis=1)


def gender_map(x):
    # the Gender column has some spaces in it
    if x.strip() == 'Male':
        return 1
    elif x.strip() == 'Female':
        return 0
    else:
        raise ValueError("must be 'Male' or 'Female'")

# I drop ethnicitiy for simplicity here, it does not affect the plot
X['Gender'] = X['Gender'].map(gender_map)
X['Student'] = X['Student'].map({'Yes': 1, 'No': 0})
X['Married'] = X['Married'].map({'Yes': 1, 'No': 0})
X = X.drop('Ethnicity', axis=1)
y = Credit['Balance'].values
# X.head() if you'd like to look at the data

# first without normalization
n_alphas = 200
alphas = np.logspace(-1, 4, n_alphas)
clf = Ridge(normalize=False)
coef_names = list(X.columns)
coefs = []


for a in alphas:
    clf.set_params(alpha=a, normalize=False)
    clf.fit(X.values, y)
    coefs.append(clf.coef_)


def make_plot():
    ax = plt.gca()

    ax.plot(alphas, coefs)
    ax.set_xscale('log')
    plt.xlabel('alpha')
    plt.ylabel('weights')
    plt.title('Ridge coefficients as a function of the regularization')
    plt.axis('tight')
    plt.legend(coef_names)
    plt.show()

make_plot()

# now with normalization
y = (y-y.mean())/y.std()
clf = Ridge(normalize=True)
coefs = []

for a in alphas:
    clf.set_params(alpha=a, normalize=True)
    clf.fit(X.values, y)
    coefs.append(clf.coef_)

make_plot()
