from register import dataset
from utils import load_model, plot_item_popularity, plot_user_popularity, PCA_analyse
# weight_file = './checkpoints/lgn-gowalla-3-64.pth.tar'
# Recmodel = load_model(weight_file)
# plot_item_popularity(dataset)
# plot_user_popularity(dataset)
# PCA_analyse(Recmodel)
from utils import UniformSample_original_python_popularity
# users = dataset.n_user
# items = dataset.m_item

# user_0 = dataset._allPos[1]
# high_user0 = dataset.highItems[1]
# low_user0 = dataset.lowItems[1]
# print(f"users{users}, items{items}")
# print(f"user_0{user_0},high_user0{high_user0},low_user0{low_user0}")
# for i in range(len(high_user0)):
#     print(dataset.item_popularity_labels[high_user0[i]])

# for i in range(len(low_user0)):
#     print(dataset.item_popularity_labels[low_user0[i]])
S = UniformSample_original_python_popularity(dataset)
print(len(S))