class knn:
    

    def __init__(self, **kwargs):
        self.x_train = None
        self.y_train = None

    def train(self, X, y):
        self.x_train = X 
        self.y_train = y

    def predict(self, testX, k):
        distances = []
        targets = []

        for i in range(len(self.x_train)):
            # Calculate Euclidean distance
            distance = np.sqrt(np.sum(np.square(testX - self.x_train[i, :])))
            # Add to the list
            distances.append([distance, i])
        #Sort the list in ascending order
        distances = sorted(distances)
        #Select the k-nearest neighbours
        for i in range(k):
            #From the list the index of the neighbour
            index = distances[i][1]
            #get the target value of the neighbour
            targets.append(self.y_train[index])

        #Return the prediction by calculating the mean of all columns
        return Counter(targets).most_common(1)[0][0]
