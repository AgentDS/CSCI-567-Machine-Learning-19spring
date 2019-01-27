import numpy as np
import utils as Util


class DecisionTree():
    def __init__(self):
        self.clf_name = "DecisionTree"
        self.root_node = None

    def train(self, features, labels):
        # features: List[List[float]], labels: List[int]
        # init
        assert (len(features) > 0)
        self.feature_dim = len(features[0])
        num_cls = np.unique(labels).size

        # build the tree
        self.root_node = TreeNode(features, labels, num_cls)
        if self.root_node.splittable:
            self.root_node.split()

        return

    def predict(self, features):
        # features: List[List[any]]
        # return List[int]
        y_pred = []
        for idx, feature in enumerate(features):
            pred = self.root_node.predict(feature)
            y_pred.append(pred)
            # print ("feature: ", feature)
            # print ("pred: ", pred)
        return y_pred


class TreeNode(object):
    def __init__(self, features, labels, num_cls):
        # features: List[List[any]], labels: List[int], num_cls: int
        self.features = features
        self.labels = labels
        self.children = []
        self.num_cls = num_cls
        self.num_attr = len(features[0])
        self.sample_cnt = len(self.labels)
        # find the most common labels in current node
        count_max = 0
        for label in np.unique(labels):
            if self.labels.count(label) > count_max:
                count_max = labels.count(label)
                self.cls_max = label
        # splitable is false when all features belongs to one class
        if len(np.unique(labels)) < 2:
            self.splittable = False
        else:
            self.splittable = True

        self.dim_split = None  # the index of the feature to be split

        self.feature_uniq_split = None  # the possible unique values of the feature to be split

        # if no feature left, then return self.cls_max
        if len(features[0]) == 0:
            self.splittable = False

    # TODO: try to split current node
    def split(self):
        if len(self.features[0]) != 0:
            # Some auxiliary data structure
            cls = np.unique(self.labels).tolist()
            sorted_features = [[] for i in range(self.num_cls)]
            for i in range(self.sample_cnt):
                cls_idx = cls.index(self.labels[i])
                sorted_features[cls_idx].append(self.features[i])
            self.sorted_features = sorted_features
            self.sorted_featuresT = []
            for i in range(self.num_cls):
                self.sorted_featuresT.append(list(map(list, zip(*sorted_features[i]))))
            featuresT = list(
                map(list, zip(*self.features)))  # transposed feature matrix (List[List[any]]: num_attri x sample_cnt)

            S = 0
            label_cnt = [0 for i in range(self.num_cls)]
            for i in range(self.num_cls):
                label_cnt[i] = len(self.sorted_features[i])
                if label_cnt != 0:
                    p = label_cnt[i] / self.sample_cnt
                    S -= p * np.log2(p)

            attr_value = [[] for i in range(self.num_attr)]
            num_branches = [0 for i in range(self.num_attr)]
            info_gain_mat = [0 for i in range(self.num_attr)]
            for i in range(self.num_attr):
                attr_value[i] = np.unique(featuresT[i]).tolist()
                num_branches[i] = len(attr_value[i])
                branches = [[0 for i in range(self.num_cls)] for k in range(num_branches[i])]
                for k in range(self.num_cls):
                    sample_feature = self.sorted_featuresT[k][i]
                    for v in range(len(attr_value[i])):
                        branches[v][k] = sample_feature.count(attr_value[i][v])
                info_gain_mat[i] = Util.Information_Gain(S, branches)
            best_attr_idx = 0
            for i in range(1, self.num_attr):
                if info_gain_mat[i] > info_gain_mat[best_attr_idx]:
                    best_attr_idx = i
                elif info_gain_mat[i] == info_gain_mat[best_attr_idx]:
                    if num_branches[i] > num_branches[best_attr_idx]:
                        best_attr_idx = i
            self.dim_split = best_attr_idx
            self.feature_uniq_split = attr_value[best_attr_idx]
            child_cnt = len(attr_value[best_attr_idx])

            new_labels = [[] for i in range(child_cnt)]
            new_features = [[] for i in range(child_cnt)]
            for i in range(self.sample_cnt):
                child_idx = self.feature_uniq_split.index(self.features[i][self.dim_split])
                new_labels[child_idx].append(self.labels[i])
                sample = self.features[i][0:self.dim_split] + self.features[i][self.dim_split + 1:]
                new_features[child_idx].append(sample)
            for i in range(child_cnt):
                self.children.append(
                    TreeNode(features=new_features[i], labels=new_labels[i], num_cls=np.unique(new_labels[i]).size))
                if self.children[i].splittable:
                    self.children[i].split()
        return

    # TODO: predict the branch or the class
    def predict(self, feature):
        # feature: List[any]
        # return: int
        raise NotImplementedError
