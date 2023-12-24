import numpy as np
import cv2 as cv
from sklearn.naive_bayes import GaussianNB
from sklearn.model_selection import train_test_split
from pywt import dwt2
from skimage import feature as fd
from sklearn.metrics import confusion_matrix, plot_confusion_matrix
import matplotlib.pyplot as plt
import pickle

def normalize_0_1(oneDarray):
    mx = max(oneDarray)
    mn = min(oneDarray)
    result_array = oneDarray / (mx - mn)
    return result_array

def normalization(array3D):
    R, G, B = cv.split(array3D)
    I = R.astype(float) + G.astype(float) + B.astype(float)
    I = np.where(I == 0, I + 1, I)
    r, g, b = R / I, G / I, B / I
    rgb = cv.merge([b, g, r])
    return rgb

def Feature_Extractor(image):
    features = []

    # Color spaces
    color_spaces = [cv.COLOR_BGR2HSV, cv.COLOR_BGR2LAB]
    for color_space in color_spaces:
        converted_image = cv.cvtColor(image, color_space)

        # Channels
        channels = cv.split(converted_image)
        for channel in channels:
            histogram, _ = np.histogram(channel, bins=8)
            features = np.append(features, normalize_0_1(histogram))

    # Energy
    im = cv.cvtColor(image, cv.COLOR_BGR2GRAY)
    _, (cH, cV, cD) = dwt2(im, 'db1')
    Energy = (cH ** 2 + cV ** 2 + cD ** 2).sum() / im.size
    histogramE, _ = np.histogram(Energy, bins=8)

    features = np.append(features, normalize_0_1(histogramE))

    # FFT
    fft = np.fft.fft2(image)
    fshift = np.fft.fftshift(fft)
    abs1 = abs(fshift)
    log1 = np.log(abs1 + 0.00000001)
    flog = abs(normalization(log1))
    histogramFFT, _ = np.histogram(flog * 255, bins=8)

    features = np.append(features, normalize_0_1(histogramFFT))

    # LBP
    for channel in cv.split(image):
        lbp_channel = fd.local_binary_pattern(channel, 24, 8, method="uniform")
        hist_lbp, _ = np.histogram(lbp_channel.ravel(), bins=8)
        features = np.append(features, normalize_0_1(hist_lbp))

    # Mean
    mean = np.mean(image)
    features = np.append(features, mean)

    # Variance
    var = np.var(image)
    features = np.append(features, var)

    # Haze features
    mu = 5.1
    v = 2.9
    sicma = 0.2461
    landa = 1 / 3
    di = np.min(image, axis=2)
    bi = np.max(image, axis=2)
    d = np.mean(np.mean(di))
    b = np.mean(np.mean(bi))
    c = b - d
    A = landa * np.max(np.max(bi)) + (1 - landa) * b + 0.00001
    x1 = (A - d) / A
    x2 = c / A
    haziness = np.exp(-0.5 * (mu * x1 + v * x2) + sicma)
    features = np.append(features, haziness)

    # Contrast
    for channel in cv.split(cv.cvtColor(image, cv.COLOR_BGR2HSV)):
        cn = np.amax(channel) - np.amin(channel)
        features = np.append(features, cn)

    # New features
    for channel in cv.split(cv.cvtColor(image, cv.COLOR_BGR2HSV)):
        minv = np.amin(channel)
        maxv = np.amax(channel)
        meanv = np.mean(channel)
        means, standard = cv.meanStdDev(image)

        features = np.append(features, minv)
        features = np.append(features, maxv)
        features = np.append(features, meanv)
        features = np.append(features, means)
        features = np.append(features, standard)

    return features

def main():
    # Data loading and preparation
    neglabel = np.zeros((1547))
    poslabel = np.ones((180))
    neglist = np.load("notsmoke.npy")
    poslist = np.load("smoke.npy")
    X = np.concatenate((neglist, poslist), axis=0)
    Y = np.append(neglabel, poslabel)

    # Feature extraction
    flist = [Feature_Extractor(x) for x in X]
    trainx = np.array(flist)

    # Train a classifier
    X_train, X_test, y_train, y_test = train_test_split(trainx, Y, test_size=0.2, random_state=0)
    gnb = GaussianNB()
    model = gnb.fit(X_train, y_train)

    # Save the trained model
    file_name = "GNB_Model-newfeatures.pkl"
    with open(file_name, "wb") as open_file:
        pickle.dump(model, open_file)

    # Confusion Matrix
    y_pred = model.predict(X_test)
    x = confusion_matrix(y_test, y_pred)
    print(x)

    # Save confusion matrix plot
    plot_confusion_matrix(model, X_test, y_test)
    plt.show()

    print("Number of mislabeled points out of a total %d points: %d"
          % (X_test.shape[0], int((y_test == y_pred).sum())))

if __name__ == "__main__":
    main()
