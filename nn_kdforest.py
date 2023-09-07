import random
import sys
import numpy as np
import ast
from collections import Counter

k = 11


# load data from file
def load_data(file_path):
    data = np.loadtxt(file_path, skiprows=1)
    features = data[:, :]
    return features


# function to calculate the squared distance of Euclidean
def distance_squared(point1, point2):
    return sum((x - y) ** 2 for x, y in zip(point1[:], point2[:]))


# function to get the most voted labels
def get_most_frequent_element(lst):
    # Count the frequency of each element in the list
    freq_counter = Counter(lst)

    # Find the maximum frequency
    max_freq = max(freq_counter.values())

    # Filter out the elements that do not have the maximum frequency
    most_common = filter(lambda x: x[1] == max_freq, freq_counter.items())

    # Sort the most common elements by their index in the list
    sorted_most_common = sorted(most_common, key=lambda x: lst.index(x[0]))

    # Return the first most common element
    return sorted_most_common[0][0]


# function to sort points along specific dimension and get the median
def sort_points(points, split_dim):
    # Rearrange points array based on sorted values
    sorted_points = points[points[:, split_dim].argsort()]

    # numbers of points
    num_points = sorted_points.shape[0]

    # Calculate median index and extract median point
    if num_points % 2 == 0:  # even number of points
        median_idx = num_points // 2 - 1
        median_point1 = sorted_points[median_idx]
        median_point2 = sorted_points[median_idx + 1]
        median_value = (median_point1[split_dim] + median_point2[split_dim]) / 2
        median_point = (median_point1 + median_point2) / 2
    else:  # odd number of points
        median_idx = num_points // 2
        median_point = sorted_points[median_idx]
        median_value = median_point[split_dim]
    # sorted_points, median_value(val),
    return sorted_points, median_point, median_value, median_idx


# class to represent the node of kd-tree
class Node:
    def __init__(self, point=None, left=None, right=None, d=None, val=0):
        # median point along that dimension
        self.point = point
        self.left = left
        self.right = right
        # d:split dimension
        self.d = d
        # val:medium value along that dimension
        self.val = val


def build_kd_tree(points, depth):
    # set of points is empty
    if points.shape[0] == 0:
        return None

    # dataset only have one data point
    elif points.shape[0] == 1:
        node = Node()
        node.d = 0
        node.val = 0
        node.point = points[0]
        return node

    else:
        # d is the split dimension
        split_dim = depth % k
        sorted_points, median_point, median_value, median_idx = sort_points(points, split_dim)

        node = Node()
        # val is the medium value along that dimension
        val = median_value
        node.d = split_dim
        node.val = val

        # Recursively build the left and right subtrees
        left_points = sorted_points[:median_idx + 1]
        right_points = sorted_points[median_idx + 1:]
        node.left = build_kd_tree(left_points, depth + 1)
        node.right = build_kd_tree(right_points, depth + 1)
        return node


def search_one_nearest_point(node, point, depth, best_distance, best=None):
    depth = node.d
    # base case: get to the leaf node
    if node.left is None and node.right is None:
        distance = distance_squared(node.point, point)
        if distance < best_distance:
            best = node
            best_distance = distance
        return best, best_distance

    # traverse the tree until reach the leaf node
    axis = depth % k
    if point[axis] < node.val:
        next_subtree = node.left
        opposite_subtree = node.right
    else:
        next_subtree = node.right
        opposite_subtree = node.left
    best, best_distance = search_one_nearest_point(next_subtree, point, depth + 1, best_distance, best)

    # unwind the tree
    if best_distance >= (point[axis] - node.val) ** 2:
        best, best_distance = search_one_nearest_point(opposite_subtree, point, depth + 1, best_distance, best)

    return best, best_distance


# function to randomly generate sample_indexes
def generate_sample_indexes(num, num_prime, n_trees, rand_seed):
    # Create a list of indexes for all data
    index_list = list(range(num))

    sample_indexes_list = []
    for j in range(n_trees):
        random.seed(rand_seed + j)
        subsample_idx = random.sample(index_list, k=num_prime)
        sample_indexes_list.append(subsample_idx)

    return sample_indexes_list


# function to build kdForest
def kd_forest(data, d_list, rand_seed):
    n = data.shape[0]
    n_prime = int(n * 0.8)
    n_trees = len(d_list)

    sample_indexes_list = generate_sample_indexes(n, n_prime, n_trees, rand_seed)

    forest = []
    # count = 0
    for count in range(n_trees):
        sampled_data = data[sample_indexes_list[count]]  # select data pairs sequentially
        root = build_kd_tree(sampled_data, int(d_list[count]))
        count += 1
        forest.append(root)

    return forest


# function to predict kd_forest
def predict_kd_forest(forest, data, depth, best_distance, best=None):
    labels = []
    for tree in forest:
        best_point = search_one_nearest_point(tree, data, depth, best_distance, best)
        rating = int(best_point[0].point[11])
        labels.append(rating)
    return get_most_frequent_element(labels)


def main():
    # get the parameters from input
    train_file = open(sys.argv[1])
    test_file = open(sys.argv[2])
    random_seed = int(sys.argv[3])
    d_list_string = sys.argv[4]
    d_list = ast.literal_eval(d_list_string)

    # load the data file
    train_features = load_data(train_file)
    test_features = load_data(test_file)

    forest = kd_forest(train_features, d_list, random_seed)
    # for data in range(test_features.shape[0]):
    #     point = test_features[data]
    #     for i in range(len(d_list)):
    #         current_dimension = int(d_list[i])
    #         most_common_label = predict_kd_forest(forest, point, current_dimension, 100000, None)
    #         print(most_common_label)

    for data in range(test_features.shape[0]):
        point = test_features[data]
        current_dimension = int(d_list[data % len(d_list)])
        most_common_label = predict_kd_forest(forest, point, current_dimension, 100000, None)
        print(most_common_label)


if __name__ == '__main__':
    main()
