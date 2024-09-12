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
from steers import SteerNet
from sklearn.utils import shuffle
from steer_model import Steer_model
import utils
from register import dataset

def load_model(weight_file):
    Recmodel = register.MODELS[world.model_name](world.config, dataset)
    Recmodel = Recmodel.to(world.device)
    
    world.cprint(f"loaded model weights from {weight_file}")
    Recmodel.load_state_dict(torch.load(weight_file,map_location=torch.device('cpu')))
    return Recmodel

def plot_item_popularity(Recmodel):
    plt.figure()
    plt.hist(Recmodel.dataset.item_popularity, bins=100)
    plt.show()
    plt.savefig("../imgs/plot_item.png")
    print(Recmodel.dataset.item_popularity.shape)

def plot_user_popularity(Recmodel):
    plt.figure()
    print(len(Recmodel.dataset.highpo_samples))
    print(len(Recmodel.dataset.lowpo_samples))
    print(Recmodel.dataset.user_popularity.shape)
    plt.hist(Recmodel.dataset.user_popularity, bins=100)
    plt.show()
    plt.savefig("../imgs/plot_user.png")


 #TODO：采样进行均衡   
def create_second_dataset():

    # highpo_labels = np.ones(len(Recmodel.dataset.highpo_samples))  # 标签为 1
    # lowpo_labels = -np.ones(len(Recmodel.dataset.lowpo_samples))   # 标签为 -1

    # X = np.concatenate([Recmodel.dataset.highpo_samples, Recmodel.dataset.lowpo_samples], axis=0)  # 合并高流行度和低流行度样本的嵌入向量
    # y = np.concatenate([highpo_labels, lowpo_labels], axis=0)  # 合并标签

    # X_shuffled, y_shuffled = shuffle(X, y, random_state=2020)
    highpo_samples = 
    
     
    return highpo_samples, lowpo_samples


def PCA_analyse(Recmodel,n_components,part):
    Recmodel.eval()
    with torch.no_grad():
        if part == 'all':
            users_ebm, item_emb = Recmodel.computer()
        else:
            users_ebm = Recmodel.embedding_user.weight
            item_emb = Recmodel.embedding_item.weight
        # item_emb = Recmodel.embedding_item.weight
        item_emb_high = item_emb[Recmodel.dataset.highpo_samples]
        item_emb_low = item_emb[Recmodel.dataset.lowpo_samples]
    n_components = 2
    pca = PCA(n_components=n_components)
    item_emb_high_pca = pca.fit_transform(item_emb_high.cpu().numpy())
    item_emb_low_pca = pca.fit_transform(item_emb_low.cpu().numpy())
    plt.figure()
    plt.scatter(item_emb_high_pca[:,0],item_emb_high_pca[:,1],c='r',label='high popularity')
    plt.show()
    plt.savefig(f"../imgs/popularity_high_{part}.png")
    plt.figure()
    plt.scatter(item_emb_low_pca[:,0],item_emb_low_pca[:,1],c='b',label='low popularity')
    plt.show()
    plt.savefig(f"../imgs/popularity_low_{part}.png")

def train_steer(Recmodel):
    Steer_rec_model = Steer_model(Recmodel,world.config)
    highpo_samples = dataset.highpo_samples
    lowpo_samples = dataset.lowpo_samples
    highpo_samples = highpo_samples.to(world.device)
    lowpo_samples = lowpo_samples.to(world.device)
    for epoch in range(200):
        Steer_rec_model.train()
        total_batch = highpo_samples.shape[0] // world.config['bpr_batch_size'] + 1
        aver_loss = 0
        for (batch_i,batch_pos) in enumerate(utils.minibatch(highpo_samples,batch_size=world.config['bpr_batch_size'] )):
            

        

    # steer = Steer_model(num_users,num_items, world.config)



if __name__ == '__main__':
    low_threshold=25
    high_threshold=50
    weight_file = './checkpoints/lgn-gowalla-3-64.pth.tar'
    n_components = 2
    part = 'all'
    Recmodel = load_model(weight_file)
    # plot_item_popularity(Recmodel)
    # plot_user_popularity(Recmodel)
    
    # PCA_analyse(Recmodel,n_components,part)
    train_steer(Recmodel)






