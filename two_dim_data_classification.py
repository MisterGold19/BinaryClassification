import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from function import train_test_split, logistic_func, log_likelihood_derivative, classify


# Expected values of labels
mu_1 = np.array([1, 1])
mu_0 = np.array([-1, 3])

#Covvariance matricies of labels distribution
sigma1 = np.array([[2, 1], [1,1]])
sigma0 = np.array([[1, -0.5], [-0.5, 1.25]])

#Generating the data
n = 200 #Number of 'records'
p = 0.5 # probability belonging to the class - 0.5 means, that probability of 0 and 1 are the same
y  = np.random.choice([0, 1], size=n, p=[1-p, p]) #Bernouilli distributioni

seed = 42
# np.random.seed(seed=seed) #initializing seed generation

features = []
for label in y:
    if label == 1:
        features.append(np.random.multivariate_normal(mu_1, sigma1))
    else:
        features.append(np.random.multivariate_normal(mu_0, sigma0))
features = np.array(features)
# print(x)

data = np.column_stack((features, y))

df = pd.DataFrame(data, columns=["feature_1", "feature_2", 'label'])

df_train, df_test = train_test_split(df, test_size_ratio=0.2, seed=seed)

# print(df_train)

# Gradient method - base


def affine_design(features: pd.DataFrame):
    return np.column_stack((np.ones(features.shape[0]), features))


wage0 = np.zeros(affine_design(df[['feature_1', 'feature_2']]).shape[1])
wage_current = wage0.copy()
# print(wage0)

alpha = 0.01 # linear coeficient to gradient

features_train = affine_design(features=df_train[['feature_1', 'feature_2']])

number_of_iteration = 10000

for iter in range(number_of_iteration):
    gradient = log_likelihood_derivative(wage_current, features=features_train, y_train=df_train['label'], model=logistic_func)
    wage_current = wage_current + alpha*gradient

# print(gradient)

x_train, x_test = df_train.drop('label', axis=1).to_numpy(), df_test.drop('label', axis=1).to_numpy()
y_train, y_test = df_train['label'].to_numpy(), df_test['label'].to_numpy()
object_features_train = affine_design(x_train)
object_features_test = affine_design(x_test)
train_classification = classify(wage=wage_current, features=object_features_train, model=logistic_func)
test_classification = classify(wage=wage_current, features=object_features_test, model=logistic_func)

train_errors = np.sum(train_classification != y_train)
test_errors = np.sum(test_classification != y_test)

print(train_errors*100/df_train.shape[0])
print(test_errors*100/df_test.shape[0])

plt.figure()

x_min, x_max = df['feature_1'].min() - 1, df['feature_1'].max() + 1
y_min, y_max = df['feature_2'].min() - 1, df['feature_2'].max() + 1

xx, yy = np.meshgrid(np.linspace(x_min, x_max, 200), np.linspace(y_min, y_max, 200))

grid_points = pd.DataFrame(np.c_[xx.ravel(), yy.ravel()])
# print(grid_points)
grid_predictions = classify(wage=wage_current, features=affine_design(grid_points), model=logistic_func).reshape(xx.shape)
# print(f'{grid_predictions=}')
# print(f'{grid_predictions.reshape(xx.shape)=}')

plt.contourf(xx, yy, grid_predictions, alpha=0.7, cmap='coolwarm')
plt.scatter(df['feature_1'], df['feature_2'], c=df['label'], edgecolors='k', cmap='coolwarm')
plt.show()