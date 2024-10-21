import world
import utils
from world import cprint
import torch
import numpy as np
from tensorboardX import SummaryWriter
import time
import Procedure
from os.path import join
import matplotlib.pyplot as plt
# ==============================
utils.set_seed(world.seed)
print(">>SEED:", world.seed)
# ==============================
import register
from register import dataset
from utils import plot_count_popularity_bar, plot_count_popularity
print(dataset.n_user)
print(dataset.m_item)

Recmodel = register.MODELS[world.model_name](world.config, dataset)
Recmodel = Recmodel.to(world.device)
weight_file = './checkpoints/lgn-gowalla-3-64-continue_train-20240926170142.pth.tar'
world.cprint(f"loaded model weights from {weight_file}")
Recmodel.load_state_dict(torch.load(weight_file,map_location=torch.device('cpu')))
results = Procedure.Test(dataset, Recmodel,0)
rating_items = results['rating_items']
rating_popularity = results['rating_popularity']
rating_count = results['rating_count'].values()


plot_count_popularity(rating_popularity, rating_count, "rating")

ground_truth_items = results['ground_truth_items']
gt_popularity = results['gt_popularity']
gt_count = results['gt_count'].values()
# print(sum(gt_count))
# print(sum(rating_count))
plot_count_popularity(gt_popularity, gt_count, "gt")

#绘制柱状图
plot_count_popularity_bar(gt_popularity, gt_count, "gt")

#rating画图
plot_count_popularity_bar(rating_popularity, rating_count, "rating")