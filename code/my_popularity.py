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
from model import Steer_model
import utils
from register import dataset

def load_model(weight_file):
    Recmodel = register.MODELS[world.model_name](world.config, dataset)
    Recmodel = Recmodel.to(world.device)
    
    world.cprint(f"loaded model weights from {weight_file}")
    Recmodel.load_state_dict(torch.load(weight_file,map_location=torch.device('cpu')))
    return Recmodel

def get_parameter_by_name(model, name):
    # 获取模型的状态字典
    state_dict = model.state_dict()
    
    # 从字典中提取对应参数
    if name in state_dict:
        param = state_dict[name]
        return param
    else:
        raise ValueError(f"Parameter '{name}' not found in the model.")

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
# def create_second_dataset():

    # highpo_labels = np.ones(len(Recmodel.dataset.highpo_samples))  # 标签为 1
    # lowpo_labels = -np.ones(len(Recmodel.dataset.lowpo_samples))   # 标签为 -1

    # X = np.concatenate([Recmodel.dataset.highpo_samples, Recmodel.dataset.lowpo_samples], axis=0)  # 合并高流行度和低流行度样本的嵌入向量
    # y = np.concatenate([highpo_labels, lowpo_labels], axis=0)  # 合并标签

    # X_shuffled, y_shuffled = shuffle(X, y, random_state=2020)
    # highpo_samples = 
    
     
    # return highpo_samples, lowpo_samples


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

def train_steer(dataset, Steer_rec_model, loss_class, epoch, config):
    
    Steer_rec_model.train()
    bpr: utils.BPR2Loss = loss_class
    with timer(name="Sample"):
        S = utils.UniformSample_original(dataset)
    users = torch.Tensor(S[:, 0]).long()
    posItems = torch.Tensor(S[:, 1]).long()
    negItems = torch.Tensor(S[:, 2]).long()
    users = users.to(world.device)
    posItems = posItems.to(world.device)
    negItems = negItems.to(world.device)
    users, posItems, negItems = utils.shuffle(users, posItems, negItems)
    total_batch = len(users) // world.config['bpr_batch_size'] + 1
    aver_loss = 0.
    
    for (batch_i,
         (batch_users,
          batch_pos,
          batch_neg)) in enumerate(utils.minibatch(users,
                                                   posItems,
                                                   negItems,
                                                   batch_size=world.config['bpr_batch_size'])):
        
        cri = bpr.stageOne(batch_users, batch_pos, batch_neg)
        aver_loss += cri
        if world.tensorboard:
            w.add_scalar(f'BPRLoss/BPR', cri, epoch * int(len(users) / world.config['bpr_batch_size']) + batch_i)
    aver_loss = aver_loss / total_batch
    time_info = timer.dict()
    timer.zero()
    return f"loss{aver_loss:.3f}-{time_info}"
        

    # steer = Steer_model(num_users,num_items, world.config)



if __name__ == '__main__':
    weight_file = './checkpoints/lgn-gowalla-3-64.pth.tar'
    n_components = 2
    part = 'all'
    Recmodel = load_model(weight_file)
    embedding_item = get_parameter_by_name(Recmodel, 'embedding_item.weight')
    # plot_item_popularity(Recmodel)
    # plot_user_popularity(Recmodel)
    # PCA_analyse(Recmodel,n_components,part)
    item_popularity_labels = dataset.item_popularity_labels
    steer_values = torch.Tensor(item_popularity_labels)[:,None]
    if world.config['dummy_steer']:
        steer_values = torch.cat([steer_values, torch.ones_like(steer_values[:,0])[:,None]],1).to(world.device)
    Steer_rec_model = Steer_model(Recmodel,world.config,steer_values).to(world.device)
    # print(Steer_rec_model)
    # for name,param in Steer_rec_model.named_parameters():
    #     if param.requires_grad:
    #         print(name,'require grad')
    #     else:
    #         print(name,'not require grad')
    # loss_class = utils.BPRLoss(Steer_rec_model, world.config)
    # epoch = 100
    # train_steer(dataset, Steer_rec_model, loss_class, epoch, world.config)



