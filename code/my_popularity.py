import dataloader
import world
import matplotlib.pyplot as plt
from torch.utils.tensorboard import SummaryWriter
dataset = dataloader.Loader(path="../data/"+world.dataset)
item_popularity = dataset.item_popularity
import numpy as np
import model
import torch
import register
from sklearn.decomposition import PCA 
# plt.hist(item_popularity, bins=100)
# plt.show()
# plt.savefig("plot.png")
# print(dataset.item_popularity.shape)
threshold = 25
highpo_samples = np.where(item_popularity > 50)
lowpo_samples = np.where(item_popularity <= threshold)

Recmodel = register.MODELS[world.model_name](world.config, dataset)
Recmodel = Recmodel.to(world.device)
weight_file = './checkpoints/lgn-gowalla-3-64.pth.tar'
world.cprint(f"loaded model weights from {weight_file}")
Recmodel.load_state_dict(torch.load(weight_file,map_location=torch.device('cpu')))
Recmodel.eval()
with torch.no_grad():
    users_ebm, item_emb = Recmodel.computer()
    # item_emb = Recmodel.embedding_item.weight
    item_emb_high = item_emb[highpo_samples]
    item_emb_low = item_emb[lowpo_samples]
n_components = 2
pca = PCA(n_components=n_components)
item_emb_high_pca = pca.fit_transform(item_emb_high.cpu().numpy())
item_emb_low_pca = pca.fit_transform(item_emb_low.cpu().numpy())
plt.scatter(item_emb_high_pca[:,0],item_emb_high_pca[:,1],c='r',label='high popularity')
plt.show()
plt.savefig("popularity_high.png")
plt.scatter(item_emb_low_pca[:,0],item_emb_low_pca[:,1],c='b',label='low popularity')
plt.show()
plt.savefig("popularity_low.png")







