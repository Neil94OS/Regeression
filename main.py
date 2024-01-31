#Neil O'Sullivan
#R00206266
#SDH4-C
import math
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.model_selection import KFold
from sklearn.preprocessing import PolynomialFeatures

# Read in csv file
data = pd.read_csv("energy_performance.csv")

def task1():
    features = data[['Relative compactness', 'Surface area', 'Wall area', 'Roof area', 'Overall height', 'Orientation', 'Glazing area', 'Glazing area distribution']]
    targets = data[['Heating load', 'Cooling load']]

    min_heating = targets['Heating load'].min()
    max_heating = targets['Heating load'].max()
    min_cooling = targets['Cooling load'].min()
    max_cooling = targets['Cooling load'].max()

    print("Minimum heating load: ",min_heating)
    print("Maximum heating load: ",max_heating)
    print()
    print("Minimum cooling load: ",min_cooling)
    print("Maximum cooling load: ",max_cooling)

    array_features = np.array(features)
    array_targets = np.array(targets)

    return array_features, array_targets

def task2a(degree, features, coefficients):
    poly = PolynomialFeatures(degree=degree)
    poly_features = poly.fit_transform(features)

    result = np.dot(poly_features, coefficients)
    return result


def task2b(degree, num_variables):
    num_coef = math.factorial(num_variables + degree) / (math.factorial(num_variables) * math.factorial(degree))
    return int(num_coef)


def task3(degree, features, coefficients):
    f0 = task2a(degree, features, coefficients) #Here is the first place the model function is called to estimate the inital target vector
                                                # that will be used to determine the difference in the other target vectors

    J = np.zeros((len(f0), len(coefficients)))
    epsilon = 1e-6
    for i in range(len(coefficients)):
        coefficients[i] += epsilon
        fi = task2a(degree, features, coefficients)# Here is the second place the model function is called to estimate the new target vectors depending on the differing coeeficients
                                                    #and will be used to calculate the difference between this and the inital target vector

        coefficients[i] -= epsilon
        di = (fi - f0) / epsilon # Here the intial target vector and newly calculated target vector are used to determine the derivatives for the Jacobian by subtracting
                                 #the inital target vector f0 from the new target vector fi
        J[:, i] = di             #Here the partial deriviate is added to the Jacobian matrix

    print("Estimated target vector :", f0)
    print("Jacobian at linearization point :", J)
    return f0, J


def task4(y, f0, J):
    l = 1e-2
    N = J.T @ J + l * np.eye(J.shape[1]) #Here is where the normal equation matrix is calculated and it is regularised by the np.eye(J.shape[1]) function
                                        # as it creates a matrix the same size as the coefficients in the Jacobian matrix with the regularization parameter 1

    r = y - f0                          #Here is where the residuals are calculated and is calculated by getting the difference between target vector y
                                        # and the estimated target vector f0
    n = J.T @ r
    dp = np.linalg.solve(N, n)

    print("Optimal parameter update vector :", dp)
    return dp


def task5(degree, features, target):
    max_iter = 10
    p0 = np.zeros(task2b(degree, features.shape[1])) #Here is the parameter vector
    for i in range(max_iter):                        #Throughout the iterations I would expect the residuals to become less and less as the first few iterations
                                                     #The optimal values may be quiet different from the previous but as the iterations go on the more optimally tuned the parameter update
                                                     # becomes so the residuals will be less and less as the iterations go on
                                                     # The number of iterations required may be determined by locating the iterations where the residuals are minimal and stable.
        f0, J = task3(degree, features, p0)
        dp = task4(target, f0, J)
        p0 += dp                                     #Here is where the parameter vector is updated by adding the result dp
                                                     # which is calculated in task4 returning the optimal parameter update to the parameter vector p0

    print("The best fitting coefficient vector: ", p0)

    return p0


def task6(degrees, features, targets):
    kf = KFold(n_splits=5, shuffle=True, random_state=42)
    best_heat = 0
    best_cool = 0
    best_mean_heat = 1e10
    best_mean_cool = 1e10

    for current_degree in degrees:
        differences_heat = []
        differences_cool = []

        heat_targets = targets[:, 0]
        for train_index, test_index in kf.split(features, heat_targets):
            X_train, y_train = features[train_index, :], heat_targets[train_index]
            X_test, y_test = features[test_index, :], heat_targets[test_index]

            best_coeff = task5(current_degree, X_train, y_train)
            prediction = task2a(current_degree, X_test, best_coeff)
            diff_heat = np.abs(prediction - y_test)
            differences_heat.append(diff_heat)

        cool_targets = targets[:, 1]
        for train_index, test_index in kf.split(features, cool_targets):
            X_train, y_train = features[train_index, :], cool_targets[train_index]
            X_test, y_test = features[test_index, :], cool_targets[test_index]

            best_coeff = task5(current_degree, X_train, y_train)
            prediction = task2a(current_degree, X_test, best_coeff)
            diff_cool = np.abs(prediction - y_test)
            differences_cool.append(diff_cool)

        mean_diff_heat = np.mean(np.concatenate(differences_heat))
        print("Heating loads mean absolute difference: ", mean_diff_heat, " for degree: ", current_degree)

        if mean_diff_heat < best_mean_heat:
            best_heat = current_degree
            best_mean_heat = mean_diff_heat

        mean_diff_cool = np.mean(np.concatenate(differences_cool))
        print("Cooling loads mean absolute difference: ", mean_diff_heat, " for degree: ", current_degree)

        if mean_diff_cool < best_mean_cool:
            best_cool = current_degree
            best_mean_cool = mean_diff_cool

    print("Best degree for heating loads: ", best_heat)
    print("Best degree for cooling loads: ", best_cool)

    return best_heat, best_cool

def task7(degree, features, targets):
    best_degree_heat, best_degree_cool = task6(degree, features, targets)

    best_coeff_heat = task5(best_degree_heat, features, targets[:, 0])
    best_coeff_cool = task5(best_degree_cool, features, targets[:, 1])

    pred_heat = task2a(best_degree_heat, features, best_coeff_heat)
    pred_cool = task2a(best_degree_cool, features, best_coeff_cool)

    plt.subplot(1, 2, 1)
    plt.scatter(targets[:, 0], pred_heat, color='red')
    plt.title('Prediction vs True heating loads')
    plt.xlabel('True heating loads')
    plt.ylabel('Predicted heating loads')

    plt.subplot(1, 2, 2)
    plt.scatter(targets[:, 1], pred_cool, color='blue')
    plt.title('Prediction vs True cooling loads')
    plt.xlabel('True cooling loads')
    plt.ylabel('Predicted cooling loads')

    mean_heat = np.mean(np.abs(pred_heat - targets[:, 0]))
    mean_cool = np.mean(np.abs(pred_cool - targets[:, 1]))
    print("Heating loads mean absolute difference : ", mean_heat)
    print("Cooling loads mean absolute difference : ", mean_cool)

    plt.show()



    return mean_heat, mean_cool

def main():
    new_features, new_targets = task1()
    degrees = [0,1,2]
    task7(degrees, new_features, new_targets)


main()