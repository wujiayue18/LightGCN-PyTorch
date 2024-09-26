from register import dataset
from utils import load_model, plot_item_popularity, plot_user_popularity, PCA_analyse
weight_file = './checkpoints/lgn-gowalla-3-64.pth.tar'
Recmodel = load_model(weight_file)
plot_item_popularity(dataset)
plot_user_popularity(dataset)
PCA_analyse(Recmodel)


