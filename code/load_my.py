import register
from register import dataset
import world
import torch
from model import Steer_model
import Procedure
from utils import plot_count_popularity_bar, plot_count_popularity
# lgn_path = "./checkpoints/lgn-gowalla-3-64-continue_train-20240926170142.pth.tar"
# rec_path = "./checkpoints/lgn-gowalla-3-64-steer_train-20241021210118.pth.tar"
# Recmodel = register.MODELS[world.model_name](world.config, dataset).to(world.device)
# Recmodel.load_state_dict(torch.load(lgn_path,map_location=torch.device('cpu')))
# item_popularity_labels = dataset.item_popularity_labels * [-1]
# steer_values = torch.Tensor(item_popularity_labels)[:,None]
# if world.config['dummy_steer']:
#     steer_values = torch.cat([steer_values, torch.ones_like(steer_values[:,0])[:,None]],1).to(world.device)
# Recmodel = Steer_model(Recmodel, world.config, dataset, steer_values)
# Recmodel.load_state_dict(torch.load(rec_path,map_location=torch.device('cpu')))
# Recmodel = Recmodel.to(world.device)
# eval_results = Procedure.Test(dataset, Recmodel, 0, None, world.config['multicore'])
# print(f"recall:{eval_results['recall']} ,ndcg:{eval_results['ndcg']},precision:{eval_results['precision']}")
# rating_items = eval_results['rating_items']
# rating_popularity = eval_results['rating_popularity']
# rating_count = eval_results['rating_count'].values()


# plot_count_popularity(rating_popularity, rating_count, "rating_train_after_1000")


# #rating画图
# plot_count_popularity_bar(rating_popularity, rating_count, "rating_train_after_1000")
lgn_path = "./checkpoints/lgn-gowalla-3-64-continue_train-20240926170142.pth.tar"
rec_path = "./checkpoints/lgn-gowalla-3-64-continue_train-20241022151224.pth.tar"
Recmodel = register.MODELS[world.model_name](world.config, dataset).to(world.device)
Recmodel.load_state_dict(torch.load(lgn_path,map_location=torch.device('cpu')))
item_popularity_labels = dataset.item_popularity_labels
steer_values = torch.Tensor(item_popularity_labels)[:,None]
if world.config['dummy_steer']:
    steer_values = torch.cat([steer_values, torch.ones_like(steer_values[:,0])[:,None]],1).to(world.device)
Recmodel = Steer_model(Recmodel, world.config, dataset, steer_values)
Recmodel.load_state_dict(torch.load(rec_path,map_location=torch.device('cpu')))
Recmodel = Recmodel.to(world.device)
eval_results = Procedure.Test(dataset, Recmodel, 0, None, world.config['multicore'])
print(f"recall:{eval_results['recall']} ,ndcg:{eval_results['ndcg']},precision:{eval_results['precision']}")
rating_items = eval_results['rating_items']
rating_popularity = eval_results['rating_popularity']
rating_count = eval_results['rating_count'].values()


plot_count_popularity(rating_popularity, rating_count, "rating_train_after_100")


#rating画图
plot_count_popularity_bar(rating_popularity, rating_count, "rating_train_after_100")
# state_dict = torch.load(rec_path,map_location=torch.device('cpu'))
# print(state_dict.keys())