import numpy as np
path = 'c://users//s.valizoda//ML python//dataset//'

def bytes_to_int(byte_data):
    return int.from_bytes(byte_data, 'big')

def readFile(filename, n_max_images=None):
    images = []
    with open(path+filename, 'rb') as f:
        _ = f.read(4)  # magic number
        n_images = bytes_to_int(f.read(4))
        if n_max_images:
            n_images = n_max_images
        n_rows = bytes_to_int(f.read(4))
        n_columns = bytes_to_int(f.read(4))
        for image_idx in range(n_images):
            image = []
            for row_idx in range(n_rows):
                row = []
                for col_idx in range(n_columns):
                    pixel = bytes_to_int(f.read(1))
                    row.append(pixel)
                image.append(row)
            images.append(image)
    return images
def read_labels(filename, n_max_labels=None):
    labels = []
    with open(path + filename, 'rb') as f:
        _ = f.read(4)  # magic number
        n_labels = bytes_to_int(f.read(4))
        if n_max_labels:
            n_labels = n_max_labels
        for label_idx in range(n_labels):
            label = bytes_to_int(f.read(1))
            labels.append(label)
    return labels

def ShowImage(sample):    
    import matplotlib.pyplot as plt
    plt.imshow(sample,cmap='Greys',interpolation='nearest')
    plt.show()
def normalize(a):
    return a.reshape(len(a),len(a[0])*len(a[0][0]))

def dist(a,b):
    s = 0
    for i in range(len(a)):
        s += (a[i]-b[i])**2
    return s**0.5


def knn(X_Train,X_labels,X_test,k=3):
    pred = []
    j = 0
    for tSample in X_test:
        print(j)
        j+=1
        d = []
        for i in range(len(X_Train)):
            d.append([i,dist(tSample,X_Train[i])])
        d.sort(key=lambda x: x[1])
        candidates = [X_labels[x] for x,_ in d[:k]]
        pred.append(max(set(candidates), key = candidates.count))        
    return pred



# the filenames to read
trainFileName = 'train-images.idx3-ubyte'
trainLabels   = 'train-labels.idx1-ubyte'
testFileName  = 't10k-images.idx3-ubyte'
testLabels    = 't10k-labels.idx1-ubyte'

# read files
N = 6000
X_Train  = np.array(readFile(trainFileName, N))  # Nx28x28
X_labels = np.array(read_labels(trainLabels, N)) # NX1

# normalizing the data
X_Train  = normalize(X_Train)         # Nx784


X_test    = np.array(readFile(testFileName,1))
X_test    = normalize(X_test)
XT_labels = np.array(read_labels(testLabels,1))

