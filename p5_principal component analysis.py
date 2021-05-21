'''
// Main File:        pca.py
// Semester:         CS 540 Fall 2020
// Authors:          Tae Yong Namkoong
// CS Login:         namkoong
// NetID:            kiatvithayak
// References:       Office Hours
                     # https://stats.stackexchange.com/questions/254592/calculating-pca-variance-explained
                     # https://stats.stackexchange.com/questions/31908/what-is-percentage-of-variance-in-pca
                     # https://docs.scipy.org/doc/scipy/reference/generated/scipy.linalg.eigh.html
                     # https://www.programcreek.com/python/example/97737/scipy.sparse.linalg.eigs
                     # https://matplotlib.org/3.1.3/api/_as_gen/matplotlib.pyplot.imshow.html
'''
from scipy.linalg import eigh
import numpy as np
import matplotlib.pyplot as plt
from scipy import linalg


""" load the dataset from a provided .npy file, re-center it around the origin 
    and return it as a NumPy array of floats
    @param filename: the dataset to be loaded
    @return NumPy array of floats  
 """
def load_and_center_dataset(filename):
    # load file
    x = np.load(filename)
    # reshape the data to n x d
    x = np.reshape(x, (2000, 784))
    # recenter data around origin
    average = np.mean(x, axis = 0)
    # subtract the dataset mean from each data point
    x = np.array(x - average)
    return x

def get_covariance(dataset):
    """
    calculate and return the covariance matrix of the dataset as a NumPy matrix (d x d array)
    @param dataset: the dataset to calculate covariance for
    @return: the covariance matrix of the dataset as a NumPy matrix (d x d array)
    """
    # error checking
    if type(dataset) != np.ndarray:
        raise ValueError("dataset is not of type ndarray")
    length = len(dataset) - 1
    # calculate covariance
    covariance = np.dot(np.transpose(dataset), dataset) / length
    # return covariance as a d x d array
    return covariance

def get_eig(S, m):
    """
    #https://docs.scipy.org/doc/scipy/reference/generated/scipy.linalg.eigh.html
    perform eigen decomposition on the covariance matrix S and return a diagonal matrix (NumPy array) with the
    largest m eigenvalues on the diagonal, and a matrix (NumPy array) with the corresponding eigenvectors as columns
    @param S: covariance matrix returned by the get_covariance function
    @param m: integer dimension for size reduction
    @return: return a diagonal matrix (NumPy array) with the
    largest m eigenvalues on the diagonal, and a matrix (NumPy array) with the corresponding eigenvectors as columns
    """
    # get eigen value and vector then reshape
    eigen_value, eigen_vector = eigh(S)
    eigen_value = eigen_value.reshape(1, -1)
    # get in descending order
    descending_order = range(len(S) - 1, len(S) - m - 1, -1)
    # value reordering for largest m eigenvalues of S in desc order
    eigen_value = eigen_value[:, descending_order]
    matrix = np.zeros((m, m), float)
    eigen_vector = eigen_vector[:, descending_order]
    eigen_value = eigen_value[0, :]
    for idx in range(m):
        matrix[idx, idx] = eigen_value[idx]
    return matrix, eigen_vector

"""
    # https://stats.stackexchange.com/questions/254592/calculating-pca-variance-explained
    # https://stats.stackexchange.com/questions/31908/what-is-percentage-of-variance-in-pca
    # https://www.programcreek.com/python/example/97737/scipy.sparse.linalg.eigs
    similar to get_eig, but instead of returning the first m, 
    return all eigenvectors that explains more than perc % of variance
    @param S: covariance matrix returned by the get_covariance function
    @param perc: percentage of variance
    @return: all eigentvectors more than perc of variance 
"""

def get_eig_perc(S, perc):
    #Solve eigen decomposition
    value, vector = linalg.eigh(S)
    eigen_value = []
    #Get sum of eigen value
    sigma = np.sum(value)
    eigen_vector = []
    #Filter for largest m values of S
    for index in range(len(value)):
        if (value[index] / sigma) > perc:
            eigen_value.append(value[index])
            vectors = vector[:, index].tolist()
            eigen_vector.append(vectors)
    eigen_value = np.array(eigen_value)
    eigen_vector = np.array(eigen_vector)
    # reorder descending: slice from end to beginning, counting down by 1
    index = eigen_value.argsort()[::-1]
    eigen_value = eigen_value[index]
    # get diagonal matrix of eigen values
    eigen_value = np.diag(eigen_value)
    # transpose eigen_vector
    eigen_vector = eigen_vector[index, :].transpose()
    # return all eigentvectors more than perc of variance
    return eigen_value, eigen_vector

'''
    project each image into your m-dimensional space and return the new representation as a d x 1 NumPy array
    @param image: the image that we want to project into m-dim space 
    @param U: eigenvector
    return  new representation as a d x 1 NumPy array
'''

def project_image(image, U):
    # transpose then project it into the m-dim space span of eigen vectors
    transpose = np.dot(np.transpose(U),image)
    return np.dot(U, transpose)

'''
    #https://matplotlib.org/3.1.3/api/_as_gen/matplotlib.pyplot.imshow.html
    use matplotlib to display a visual representation of the original image and the projected image side-by-side
    @param original: original image
    @param projection: projection image  
'''
def display_image(orig, proj):
    # Reshape the images to 28x28 for orig and proj
    orig = np.reshape(orig, (28, 28))
    proj = np.reshape(proj, (28, 28))
    #Create a figure with one row of two subplots
    figure, axes = plt.subplots(1, 2)
    #Set title for axes
    axes[0].set_title('Original')
    axes[1].set_title('Projection')
    # Use imshow() with aspect='equal' and cmap='gray' to render the orig image
    # in the first subplot and the proj in the second subplot
    subplot1 = axes[0].imshow(orig, aspect='equal', cmap='gray')
    subplot2 = axes[1].imshow(proj, aspect='equal', cmap='gray')
    # Use the return value of imshow() to create a colorbar for each image
    figure.colorbar(subplot1, ax=axes[0])
    figure.colorbar(subplot2, ax=axes[1])
    # Render plot
    plt.show()
