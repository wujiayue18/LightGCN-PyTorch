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

Recmodel = register.MODELS[world.model_name](world.config, dataset)
Recmodel = Recmodel.to(world.device)
weight_file = './checkpoints/lgn-gowalla-3-64-continue_train-20240926170142.pth.tar'
world.cprint(f"loaded model weights from {weight_file}")
Recmodel.load_state_dict(torch.load(weight_file,map_location=torch.device('cpu')))
results = Procedure.Test(dataset, Recmodel,0)
rating_items = results['rating_items']
rating_popularity = results['rating_popularity']
rating_count = results['rating_count'].values()

plt.figure()
plt.scatter(rating_count, rating_popularity, alpha=0.5)  
plt.title('Item Popularity vs Recommendation Count')
plt.ylabel('Popularity')
plt.xlabel('Recommendation Count')
# 显示图形
plt.show()
plt.savefig(f"../imgs/{world.dataset}/plot_rating_items_popularity.png")

ground_truth_items = results['ground_truth_items']
gt_popularity = results['gt_popularity']
gt_count = results['gt_count'].values()
plt.figure()
plt.scatter(gt_count, gt_popularity, alpha=0.5)  
plt.title('Item Popularity vs Recommendation Count')
plt.ylabel('Popularity')
plt.xlabel('Recommendation Count')
# 显示图形
plt.show()
plt.savefig(f"../imgs/{world.dataset}/plot_gt_items_popularity.png")


