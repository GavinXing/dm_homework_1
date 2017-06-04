import numpy as np
import operator

class KNN(object):
    """docstring for KNN."""
    def __init__(self, n_neighbors=5):
        super(KNN, self).__init__()
        self.n_neighbors = n_neighbors

    def fit(self, X, Y):
        self.trainX = X
        self.trainY = Y

    def predict(self, X):
        [N, _] = self.trainX.shape

        pred_y = []
        for pred_x in X:
            # calculate the euclidean distance
            difference = np.tile(pred_x, (N, 1)) - self.trainX # repeat and calculate distance
            difference = difference ** 2 ## square for euclidean distance
            distance = difference.sum(1)

            distance = distance ** 0.5
            sortdiffidx = distance.argsort() ## distance sort index

            # find the nearest n_neighbors
            vote = {}
            for i in range(self.n_neighbors):
                ith_label = self.trainY[sortdiffidx[i]]
                vote[ith_label] = vote.get(ith_label, 0) + 1
            sortedVote = sorted(vote.items(), key=lambda x:x[1], reverse=True)
            pred_y.append(sortedVote[0][0])
        return np.array(pred_y)

if __name__ == '__main__':
    group = np.array([[1.0,1.1],[1.0,1.0],[0,0],[0,0.1]])
    labels = ['A','A','B','B']

    knn = KNN(n_neighbors=2)
    knn.fit(group, labels)
    pred = knn.predict([[0,0], [2, 3], [-1, 2]])
    print(pred)
