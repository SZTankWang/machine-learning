import numpy as np

# threshold to determine whether to include a word in the dictionary/wordlist.
# ie. only words with frequency higher than threshold are included
THRESHOLD = 26
train_path = './spam_train.txt'
test_path = './spam_test.txt'


def read_data(filename):
    """
    Read the dataset from the file given by name $filename.
    The returned object should be a list of pairs of data, such as
        [
            (True , ['a', 'b', 'c']),
            (False, ['d', 'e', 'f']),
            ...
        ]
    """

    with open(filename, 'r') as f:
        return [(bool(int(y)), x) for y, *x in [line.strip().split() for line in f]]


def split_train(original_train_data):
    return original_train_data[:4000], original_train_data[4000:]


def create_wordlist(original_train_data, threshold=26):
    """
    Create a word list from the original training set.
    Only get a word if it appears in at least $threshold emails.
    """
    table = {}
    for _, words in original_train_data:
        words = set(words)
        for word in words:
            if word not in table:
                table[word] = 1
            else:
                table[word] += 1
    result = []
    for word, count in table.items():
        if count >= threshold:
            result.append(word)
    return result


class Model:
    def __init__(self, wordlist):
        self.wordlist = wordlist

    def count_labels(self, data):
        """
        Count the number of positive labels and negative labels.
        Returns (a tuple or a numpy array of two elements):
            * negative_count: a non-negative integer, which represents the number of negative labels;
            * positive_count: a non-negative integer, which represents the number of positive labels.
        """

        positive_count = sum([y for y, x in data])
        negative_count = len(data) - positive_count
        return np.array([negative_count, positive_count])

    def count_words(self, wordlist, data):
        """
        Count the number of times that each word appears in emails under a given label.
        Returns (a numpy array):
            * word_counts: a numpy array with shape (2, L), where L is the length of $wordlist,
                - word_counts[0, i] represents the number of times that word $wordlist[i] appears in non-spam (negative) emails, and
                - word_counts[1, i] represents the number of times that word $wordlist[i] appears in spam (positive) emails.
        """
        result = np.zeros((2, len(wordlist)))
        for entry in data:
            for i in range(len(wordlist)):
                if wordlist[i] in entry[1]:
                    result[int(entry[0]), i] += 1
        return result

    def calculate_probability(self, label_counts, word_counts):
        """
        Calculate the probabilities, both the prior and likelihood.
        Returns (a pair of numpy array):
            * prior_probs: a numpy array with shape (2, ), only two elements, where
                - prior_probs[0] is the prior probability of negative labels, and
                - prior_probs[1] is the prior probability of positive labels.
            * likelihood_probs: a numpy array with shape (2, L), where L is the length of the word list,
                - likelihood_probs[0, i] represents the likelihood probability of the $i-th word in the word list, given that the email is non-spam (negative), and
                - likelihood_probs[1, i] represents the likelihood probability of the $i-th word in the word list, given that the email is spam (positive).
        """
        prior_probs = label_counts / np.sum(label_counts)
        likelihood_probs = np.zeros((2, len(self.wordlist)))
        likelihood_probs[0, :] = (word_counts[0, :] + 1) / (label_counts[0] + 2)
        likelihood_probs[1, :] = (word_counts[1, :] + 1) / (label_counts[1] + 2)
        return prior_probs, likelihood_probs

    def fit(self, data):
        label_counts = self.count_labels(data)
        word_counts = self.count_words(self.wordlist, data)

        self.prior_probs, self.likelihood_probs = self.calculate_probability(label_counts, word_counts)

        # TO AVOID NUMBER OVERFLOW here we use log probability instead.
        self.log_prior_probs = np.log(self.prior_probs)
        self.log_likelihood_probs = np.dstack([np.log(1 - self.likelihood_probs), np.log(self.likelihood_probs)])

    def predict(self, x):
        """
        Predict whether email $x is a spam or not based on the posterior probability
        Returns:
            * y: a boolean value indicating whether $x is a spam or not.
        """
        p_0 = np.sum(
            [self.log_likelihood_probs[0, i, 1 if self.wordlist[i] in x else 0] for i in range(len(self.wordlist))]) + \
              self.log_prior_probs[0]
        p_1 = np.sum(
            [self.log_likelihood_probs[1, i, 1 if self.wordlist[i] in x else 0] for i in range(len(self.wordlist))]) + \
              self.log_prior_probs[1]
        return p_1 > p_0


def main():
    original_train_data = read_data(train_path)
    train_data, val_data = split_train(original_train_data)

    # Create the word list.
    wordlist = create_wordlist(original_train_data, THRESHOLD)
    print("Total # of words:", len(wordlist))

    model = Model(wordlist)
    model.fit(train_data)

    error_count = sum([y != model.predict(x) for y, x in val_data])
    error_percentage = error_count / len(val_data) * 100

    print("Validation error, # = {:>4d}, % = {:>8.4f}%.".format(error_count, error_percentage))

    """
    P3.2 try out different values of vocabulary size and fit the model, and show your result in the PDF file 
    """


# def worker(t):
#     original_train_data = read_data(train_path)
#     train_data, val_data = split_train(original_train_data)
#
#     # Create the word list.
#     wordlist = create_wordlist(original_train_data, t)
#     print("Total # of words:", len(wordlist))
#
#     model = Model(wordlist)
#     model.fit(train_data)
#
#     error_count = sum([y != model.predict(x) for y, x in val_data])
#     error_percentage = error_count / len(val_data) * 100
#
#     print("Threshold = {}\nValidation error, # = {:>4d}, % = {:>8.4f}%.".format(t, error_count, error_percentage))
#     return error_percentage
#
#
# def test_thres():
#     from multiprocessing import Pool
#     import matplotlib.pyplot as plt
#     x = list(range(2, 41, 2))
#     with Pool(4) as pool:
#         error = pool.map(worker, x)
#     print(error)
#     plt.plot(x, error, marker='o')
#     plt.title('Error Percentage')
#     plt.xlabel('Threshold')
#     plt.ylabel('% Error')
#     plt.show()


if __name__ == '__main__':
    main()
    # test_thres()
