# model code from https://github.com/aniton/MMODA_NLA_Project

import glob
import os
from os.path import join
import librosa
import matplotlib.pyplot as plt
import numpy as np
from sklearn.model_selection import train_test_split
from tqdm import tqdm


import tensorly as tl
from tensorly.decomposition import symmetric_parafac_power_iteration


# def cosine_distance(x, y):
#     return (1 - np.dot(x, y) / (np.linalg.norm(y) * np.linalg.norm(x))) * .5

def get_new(x, y):
    z = np.polyfit(x, y, 6)
    f = np.poly1d(z)
    y = f(x)
    return x, y

def cosine_distance(x, y):
    return 1 - np.square(np.dot(x, y)) / (np.dot(x, x) * np.dot(y, y))


def get_features_filtered_by_label(features, labels, selected_label):
    return [feature for feature, label in zip(features, labels) if label == selected_label]

def get_k_main_eigenvectors(X, k=1):
    m1 = X.mean(axis=0)
    corr_matrix = np.cov(X.T)
    corr_eigens = np.linalg.eigvalsh(corr_matrix)
    m2 = corr_matrix + np.outer(m1, m1) - corr_eigens[0] * np.eye(corr_matrix.shape[0])
    w, v = np.linalg.eigh(m2)
    return v[:, -k:].T

def whiten(m2, eps=1e-18):
   # eigenvalue decomposition of the covariance matrix
   d, V = np.linalg.eigh(m2)

   # a fudge factor can be used so that eigenvectors associated with
   # small eigenvalues do not get overamplified.
   D = np.diag(1. / np.sqrt(d+eps))

   # whitening matrix
   W = np.dot(V, D)

   return W

def get_k_main_eigenvectors_cpd(X, k=1):
    m1 = X.mean(axis=0)
    N = X.shape[1]
    corr_matrix = np.cov(X.T)
    corr_eigens = np.linalg.eigvalsh(corr_matrix)
    m2 = corr_matrix + np.outer(m1, m1) - corr_eigens[0] * np.eye(corr_matrix.shape[0], dtype='float64')
    m3 = np.einsum('ij,ik,il->jkl',X,X,X) / N
    basis_sum = np.einsum('ij,ik,il->jkl',  m1[None, :], np.eye(N), np.eye(N)) + \
            np.einsum('ij,ik,il->jkl', np.eye(N),  m1[None, :], np.eye(N)) + \
            np.einsum('ij,ik,il->jkl', np.eye(N), np.eye(N),  m1[None, :])
    # basis_sum = np.einsum('j,ik,il->jkl',  m1, np.eye(N), np.eye(N)) + \
    #         np.einsum('ij,k,il->jkl', np.eye(N),  m1, np.eye(N)) + \
    #         np.einsum('ij,ik,l->jkl', np.eye(N), np.eye(N),  m1)
    m3 = m3 - corr_eigens[0] * basis_sum
    w = whiten(m2)
    # m3_whiten = tl.tucker_to_tensor((m3, [w, w, w]))
    m3_whiten = np.einsum('nmp,nk,ml,pd->kld', m3, w, w, w)
    lambdas, V = symmetric_parafac_power_iteration(m3_whiten, rank=k, n_iteration=20)
    A = np.linalg.pinv(w.T) @ V @ np.diag(lambdas)
    return A.T # V.T


def get_name(path):
    return os.path.splitext(os.path.split(path)[1])[0]

# --- test functions ---

def test(eigen_female, eigen_male, features_test, is_correct_function, debug=False, **kwargs):
    counter = 0
    num_eigenvectors = kwargs['num_eigenvectors']
    for feature in features_test:
        eigen_test = get_k_main_eigenvectors(feature, k=num_eigenvectors)
        dist_to_female = min([np.square(ete - etr).sum() for etr, ete in zip(eigen_female, eigen_test)])
        dist_to_male = min([np.square(ete - etr).sum() for etr, ete in zip(eigen_male, eigen_test)])
        correct = is_correct_function(dist_to_female, dist_to_male)
        if correct:
            counter += 1
    if debug:
        print(f'{dist_to_female:.3e}, {dist_to_male:.3e}, {correct}')
    return counter / len(features_test)

def test_cosine(eigen_female, eigen_male, features_test, is_correct_function, debug=False, **kwargs):
    counter = 0
    for feature in features_test:
        dist_to_female = sum([min([cosine_distance(x, etr)
                                   for etr in eigen_female]) for x in feature])
        dist_to_male = sum([min([cosine_distance(x, etr)
                                   for etr in eigen_male]) for x in feature])
        correct = is_correct_function(dist_to_female, dist_to_male)
        if correct:
            counter += 1
    if debug:
        print(f'{dist_to_female:.3e}, {dist_to_male:.3e}, {correct}')
    return counter / len(features_test)


# -----------------------------------------------------------------
# def normalize(x):
#     return x - x.mean()

def normalize(x):
    return x / np.linalg.norm(x, ord=2, axis=1, keepdims=True)

# check normalize
def get_k_main_eigenvectors_svd(features, k=1):
    _, _, vh = np.linalg.svd(normalize(features), full_matrices=False)
    return vh[:k]



def cross_validate(mfcc_list, labels_list, random_state_list, test_function, num_eigenvectors=1):
    results = {}
    accuracy_list = []
    accuracy_list_female = []
    accuracy_list_male = []
    for random_state in random_state_list:
        X_train, X_test, y_train, y_test = train_test_split(mfcc_list, labels_list, test_size=0.33,
                                                            random_state=random_state, stratify=labels_list)

        features = np.concatenate(get_features_filtered_by_label(X_train, y_train, 'F'))
        eigen_female = get_k_main_eigenvectors(features, k=num_eigenvectors)

        features = np.concatenate(get_features_filtered_by_label(X_train, y_train, 'M'))
        eigen_male = get_k_main_eigenvectors(features, k=num_eigenvectors)

        features_test = get_features_filtered_by_label(X_test, y_test, 'F')
        is_correct_function = lambda x, y: x < y
        accuracy_female = test_function(eigen_female, eigen_male, features_test, is_correct_function,
                                        num_eigenvectors=num_eigenvectors)

        features_test = get_features_filtered_by_label(X_test, y_test, 'M')
        is_correct_function = lambda x, y: x > y
        accuracy_male = test_function(eigen_female, eigen_male, features_test, is_correct_function,
                                     num_eigenvectors=num_eigenvectors)

#         print(f'f: {accuracy_female:.3f}, m: {accuracy_male:.3f}')
        accuracy_list.append((accuracy_female + accuracy_male) * .5)
        accuracy_list_female.append(accuracy_female)
        accuracy_list_male.append(accuracy_male)

    results['accuracy'] = accuracy_list
    results['accuracy_female'] = accuracy_list_female
    results['accuracy_male'] = accuracy_list_male

    results['std'] = np.std(accuracy_list)
    results['std_female'] = np.std(accuracy_list_female)
    results['std_male'] = np.std(accuracy_list_male)

    return results

def cross_validate_article(mfcc_list, labels_list, random_state_list, test_function, num_eigenvectors=1):
    results = {}
    accuracy_list = []
    accuracy_list_female = []
    accuracy_list_male = []
    for random_state in random_state_list:
        X_train, X_test, y_train, y_test = train_test_split(mfcc_list, labels_list, test_size=0.33,
                                                            random_state=random_state, stratify=labels_list)

        features = np.concatenate(get_features_filtered_by_label(X_train, y_train, 'F'))
        eigen_female = get_k_main_eigenvectors_cpd(features, k=num_eigenvectors)

        features = np.concatenate(get_features_filtered_by_label(X_train, y_train, 'M'))
        eigen_male = get_k_main_eigenvectors_cpd(features, k=num_eigenvectors)

        features_test = get_features_filtered_by_label(X_test, y_test, 'F')
        is_correct_function = lambda x, y: x < y
        accuracy_female = test_function(eigen_female, eigen_male, features_test, is_correct_function,
                                        num_eigenvectors=num_eigenvectors)

        features_test = get_features_filtered_by_label(X_test, y_test, 'M')
        is_correct_function = lambda x, y: x > y
        accuracy_male = test_function(eigen_female, eigen_male, features_test, is_correct_function,
                                     num_eigenvectors=num_eigenvectors)

#         print(f'f: {accuracy_female:.3f}, m: {accuracy_male:.3f}')
        accuracy_list.append((accuracy_female + accuracy_male) * .5)
        accuracy_list_female.append(accuracy_female)
        accuracy_list_male.append(accuracy_male)

    results['accuracy'] = accuracy_list
    results['accuracy_female'] = accuracy_list_female
    results['accuracy_male'] = accuracy_list_male

    results['std'] = np.std(accuracy_list)
    results['std_female'] = np.std(accuracy_list_female)
    results['std_male'] = np.std(accuracy_list_male)

    return results


paths_mixed = sorted(glob.glob('data_named/*mixed.wav'))
paths_target = sorted(glob.glob('data_named/*target.wav'))
paths_result = sorted(glob.glob('data_named/*result.wav'))

accuracy_scores = []
male_accuracy_scores = []
female_accuracy_scores = []

for dataset_path, dataset_name in [(paths_mixed, 'mixed'), (paths_target, 'target'), (paths_result, 'result')]:
    mfcc_list = []
    labels_list = []
    for path in tqdm(dataset_path):
        print(path)
        splitted = get_name(path).split('-')
        y, sr = librosa.load(path)
        mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=26).T
        mfcc_list.append(mfcc)
        labels_list.append(get_name(path)[0])

    np.random.seed(122)
    random_state_list = np.random.randint(0, 1000000, size=25)


    results = cross_validate(mfcc_list, labels_list, random_state_list, test_function=test_cosine, num_eigenvectors=4)
    print(dataset_name)
    print(results['accuracy'])
    print(results['accuracy_female'])
    print(results['accuracy_male'])

    accuracy_scores.append(results['accuracy'])
    male_accuracy_scores.append(results['accuracy_male'])
    female_accuracy_scores.append(results['accuracy_female'])



for scores, name in [(accuracy_scores, 'general_accuracy'), (male_accuracy_scores, 'male_accuracy'), (female_accuracy_scores, 'female_accuracy')]:
    y1 = scores[0]
    x1 = [i for i in range(len(scores[0]))]
    x1, y1 = get_new(x1, y1)
    # plotting the line 1 points
    plt.plot(x1, y1, label = "Given audio")
    # line 2 points
    y2 = scores[1]
    x2 = [i for i in range(len(scores[1]))]
    x2, y2 = get_new(x2, y2)
    # plotting the line 2 points
    plt.plot(x2, y2, label = "Target audio")

    y3 = scores[2]
    x3 = [i for i in range(len(scores[2]))]
    x3, y3 = get_new(x3, y3)
    # plotting the line 2 points
    plt.plot(x3, y3, label = "Filtered audio")
    plt.xlabel('# of audio')
    # Set the y axis label of the current axis.
    plt.ylabel('Accuracy')
    # Set a title of the current axes.
    plt.title('Gender recognition accuracy ' + name)
    # show a legend on the plot
    plt.legend()

    # plt.rcParams['figure.figsize'] = (20,30)
    plt.savefig('{}-fix.png'.format(name))   # save the figure to file
    plt.close()
