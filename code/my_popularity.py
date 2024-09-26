import dataloader
import world
import matplotlib.pyplot as plt
from torch.utils.tensorboard import SummaryWriter
dataset = dataloader.Loader(path="../data/"+world.dataset)
item_popularity = dataset.item_popularity
import numpy as np
import model
import time
import torch
import register
from sklearn.decomposition import PCA 
from steers import SteerNet
from sklearn.utils import shuffle
from model import Steer_model
import utils
from register import dataset
from os.path import join
from utils import timer
from world import cprint
import Procedure







 #TODO：采样进行均衡   
# def create_second_dataset():

    # highpo_labels = np.ones(len(Recmodel.dataset.highpo_samples))  # 标签为 1
    # lowpo_labels = -np.ones(len(Recmodel.dataset.lowpo_samples))   # 标签为 -1

    # X = np.concatenate([Recmodel.dataset.highpo_samples, Recmodel.dataset.lowpo_samples], axis=0)  # 合并高流行度和低流行度样本的嵌入向量
    # y = np.concatenate([highpo_labels, lowpo_labels], axis=0)  # 合并标签

    # X_shuffled, y_shuffled = shuffle(X, y, random_state=2020)
    # highpo_samples = 
    
     
    # return highpo_samples, lowpo_samples




def train_steer(dataset, Steer_rec_model, loss_class, epoch, neg_k=1, w=None):
    Steer_rec_model = Steer_rec_model
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
    utils.set_seed(world.seed)
    weight_file = './checkpoints/lgn-gowalla-3-64.pth.tar'

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
    Steer_rec_model = Steer_model(Recmodel,world.config,world.dataset,steer_values).to(world.device)
    loss_class = utils.BPR2Loss(Steer_rec_model, world.config)
    weight_file = utils.getFileName()
    # if world.config['continue_train']:

    if world.tensorboard:
        w : SummaryWriter = SummaryWriter(
                                        join(world.BOARD_PATH, time.strftime("%m-%d-%Hh%Mm%Ss-") + "-" + world.comment)
                                        )
    else:
        w = None
        world.cprint("not enable tensorflowboard")
    # print(Steer_rec_model)
    # for name,param in Steer_rec_model.named_parameters():
    #     print(param.device)
    #     if param.requires_grad:
    #         print(name,'require grad')
    #     else:
    #         print(name,'not require grad')
    
    Neg_k = 1
    # train_steer(dataset, Steer_rec_model, loss_class, epoch, world.config)
    try:
        for epoch in range(world.TRAIN_epochs):
            start = time.time()
            if epoch %10 == 0:
                cprint("[TEST]")
                Procedure.Test(dataset, Steer_rec_model, epoch, w, world.config['multicore'])
            output_information = train_steer(dataset, Steer_rec_model, loss_class, epoch, neg_k=Neg_k,w=w)
            print(f'EPOCH[{epoch+1}/{world.TRAIN_epochs}] {output_information}')
            torch.save(Steer_rec_model.state_dict(), weight_file)
    finally:
        if world.tensorboard:
            w.close()


