'''
Created on Mar 1, 2020
Pytorch Implementation of LightGCN in
Xiangnan He et al. LightGCN: Simplifying and Powering Graph Convolution Network for Recommendation

@author: Jianbai Ye (gusye@mail.ustc.edu.cn)
'''
import world
import torch
from torch import nn, optim
import numpy as np
from torch import log
from dataloader import BasicDataset
from time import time
from model import LightGCN
from model import PairWiseModel
from sklearn.metrics import roc_auc_score
import random
import os
from datetime import datetime
import register
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA 
from register import dataset
# try:
#     from cppimport import imp_from_filepath
#     from os.path import join, dirname
#     path = join(dirname(__file__), "sources/sampling.cpp")
#     sampling = imp_from_filepath(path)
#     sampling.seed(world.seed)
#     sample_ext = True
# except:
#     world.cprint("Cpp extension not loaded")
#     sample_ext = False


class BPRLoss:
    def __init__(self,
                 recmodel : PairWiseModel,
                 config : dict):
        self.model = recmodel
        self.weight_decay = config['decay']
        self.lr = config['lr']
        self.opt = optim.Adam(recmodel.parameters(), lr=self.lr)

    def stageOne(self, users, pos, neg):
        loss, reg_loss = self.model.bpr_loss(users, pos, neg)
        reg_loss = reg_loss*self.weight_decay
        loss = loss + reg_loss

        self.opt.zero_grad()
        loss.backward()
        self.opt.step()

        return loss.cpu().item()


class BPR2Loss:
    def __init__(self,
                 Steer_rec_model : PairWiseModel,
                 config : dict):
        self.model = Steer_rec_model
        self.weight_decay = config['decay']
        self.steer_decay = config['steer_decay']
        self.lr = config['lr']
        self.opt = optim.Adam(Steer_rec_model.parameters(), lr=self.lr)

    def stageOne(self, users, high,low, neg):
        #看一下reg_loss,users,pos
        loss, reg_loss_steer = self.model.bpr_loss(users, high, low, neg)
        # reg_loss = reg_loss*self.weight_decay
        reg_loss_steer = reg_loss_steer*self.steer_decay
        # loss = loss + reg_loss + reg_loss_steer
        loss = loss + reg_loss_steer
        
        self.opt.zero_grad()
        loss.backward()
        self.opt.step()

        return loss.cpu().item()
    
# #TODO:infoNCE类
# class InfoNCELoss(nn.Module):
#     def __init__(self,
#                 steer : PairWiseModel,
#                 config : dict):
#         self.model = steer
#         self.lr = config['lr']
#         self.opt = optim.Adam(steer.parameters(), lr=self.lr)

#     def stageOne(self, anchor, pos, neg):
#         loss = self.model.infoNCE_loss(anchor, pos, neg)

#         self.opt.zero_grad()
#         loss.backward()
#         self.opt.step()

#         return loss.cpu().item()


def UniformSample_original(dataset, neg_ratio = 1):
    dataset : BasicDataset
    allPos = dataset.allPos
    start = time()
    # if sample_ext:
    #     S = sampling.sample_negative(dataset.n_users, dataset.m_items,
    #                                  dataset.trainDataSize, allPos, neg_ratio)
    # else:
    S = UniformSample_original_python(dataset)
    return S

def UniformSample_original_python(dataset):
    """
    the original impliment of BPR Sampling in LightGCN
    :return:
        np.array
    """
    total_start = time()
    dataset : BasicDataset
    user_num = dataset.trainDataSize
    users = np.random.randint(0, dataset.n_users, user_num)
    allPos = dataset.allPos
    S = []
    sample_time1 = 0.
    sample_time2 = 0.
    for i, user in enumerate(users):
        start = time()
        posForUser = allPos[user]
        if len(posForUser) == 0:
            continue
        sample_time2 += time() - start
        posindex = np.random.randint(0, len(posForUser))
        positem = posForUser[posindex]
        while True:
            negitem = np.random.randint(0, dataset.m_items)
            if negitem in posForUser:
                continue
            else:
                break
        S.append([user, positem, negitem])
        end = time()
        sample_time1 += end - start
    total = time() - total_start
    return np.array(S)


def UniformSample_original_python_popularity(dataset):
    """
    the original impliment of BPR Sampling in LightGCN
    :return:
        np.array
    """
    total_start = time()
    dataset : BasicDataset
    
    allPos = dataset.allPos
    highItems = dataset.highItems
    lowItems = dataset.lowItems

    user_num = sum(len(sublist) for sublist in highItems)
    # print(user_num)
    users = np.random.randint(0, dataset.n_users, user_num)
    
    S = []
    
    for i, user in enumerate(users):
        posForUser = allPos[user]
        highForUser = highItems[user]
        lowForUser = lowItems[user]
        if len(highForUser) == 0 or len(lowForUser) == 0:
            continue
        
        highindex = np.random.randint(0, len(highForUser))
        highitem = highForUser[highindex]
        lowindex = np.random.randint(0, len(lowForUser))
        lowitem = lowForUser[lowindex]      
        while True:
            negitem = np.random.randint(0, dataset.m_items)
            if negitem in posForUser:
                continue
            else:
                break      
        S.append([user, highitem, lowitem, negitem])
        
        
    total = time() - total_start
    return np.array(S)

# ===================end samplers==========================
# =====================utils====================================

def set_seed(seed):
    np.random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    torch.manual_seed(seed)

def getFileName():
    if world.model_name == 'mf':
        if world.config['steer_train'] == 1:
            if world.config['continue_train'] == 0:
                file = f"mf-{world.dataset}-{world.config['latent_dim_rec']}-steer_train-{datetime.now().strftime('%Y%m%d%H%M%S')}.pth.tar"
            else:
                file = f"mf-{world.dataset}-{world.config['latent_dim_rec']}-continue_train-{datetime.now().strftime('%Y%m%d%H%M%S')}.pth.tar"
        else:
            file = f"mf-{world.dataset}-{world.config['latent_dim_rec']}-{datetime.now().strftime('%Y%m%d%H%M%S')}.pth.tar"
    elif world.model_name == 'lgn':
        if world.config['steer_train'] == 1:
            if world.config['continue_train'] == 0:
                file = f"lgn-{world.dataset}-{world.config['lightGCN_n_layers']}-{world.config['latent_dim_rec']}-steer_train-{datetime.now().strftime('%Y%m%d%H%M%S')}.pth.tar"
            else:
                file = f"lgn-{world.dataset}-{world.config['lightGCN_n_layers']}-{world.config['latent_dim_rec']}-continue_train-{datetime.now().strftime('%Y%m%d%H%M%S')}.pth.tar"
        else:
            file = f"lgn-{world.dataset}-{world.config['lightGCN_n_layers']}-{world.config['latent_dim_rec']}-lgn-{datetime.now().strftime('%Y%m%d%H%M%S')}.pth.tar"
    return os.path.join(world.FILE_PATH,file)

def minibatch(*tensors, **kwargs):

    batch_size = kwargs.get('batch_size', world.config['bpr_batch_size'])

    if len(tensors) == 1:
        tensor = tensors[0]
        for i in range(0, len(tensor), batch_size):
            yield tensor[i:i + batch_size]
    else:
        for i in range(0, len(tensors[0]), batch_size):
            yield tuple(x[i:i + batch_size] for x in tensors)


def shuffle(*arrays, **kwargs):

    require_indices = kwargs.get('indices', False)

    if len(set(len(x) for x in arrays)) != 1:
        raise ValueError('All inputs to shuffle must have '
                         'the same length.')

    shuffle_indices = np.arange(len(arrays[0]))
    np.random.shuffle(shuffle_indices)

    if len(arrays) == 1:
        result = arrays[0][shuffle_indices]
    else:
        result = tuple(x[shuffle_indices] for x in arrays)

    if require_indices:
        return result, shuffle_indices
    else:
        return result

def load_model(weight_file):
    Recmodel = register.MODELS[world.model_name](world.config, dataset)
    Recmodel = Recmodel.to(world.device)
    
    world.cprint(f"loaded model weights from {weight_file}")
    Recmodel.load_state_dict(torch.load(weight_file,map_location=torch.device('cpu')))
    return Recmodel


def plot_item_popularity(dataset):
    plt.figure()
    plt.hist(dataset.item_popularity, bins=100)
    plt.show()
    plt.savefig(f"../imgs/{world.dataset}/plot_item.png")
    print(dataset.item_popularity.shape)

def plot_user_popularity(dataset):
    plt.figure()
    print(len(dataset.highpo_samples))
    print(len(dataset.lowpo_samples))
    print(dataset.user_popularity.shape)
    plt.hist(dataset.user_popularity, bins=100)
    plt.show()
    plt.savefig(f"../imgs/{world.dataset}/plot_user.png")

def PCA_analyse(Recmodel):
    Recmodel.eval()
    with torch.no_grad():
        if world.config['emb_ans_pos'] == 'before':
            users_ebm = Recmodel.embedding_user.weight
            item_emb = Recmodel.embedding_item.weight
        else:
            users_all, item_all, users_layer1, items_layer1, users_layer2, items_layer2, users_layer3, items_layer3 = Recmodel.computer()
            if world.config['emb_ans_pos'] == 'layer_1':
                users_ebm = users_layer1
                item_emb = items_layer1
            elif world.config['emb_ans_pos'] == 'layer_2':
                users_ebm = users_layer2
                item_emb = items_layer2
            elif world.config['emb_ans_pos'] == 'layer_3':
                users_ebm = users_layer3
                item_emb = items_layer3
            elif world.config['emb_ans_pos'] == 'all':
                users_ebm = users_all
                item_emb = item_all
            else:
                raise NotImplementedError
        # item_emb = Recmodel.embedding_item.weight
        item_emb_high = item_emb[Recmodel.dataset.highpo_samples]
        item_emb_low = item_emb[Recmodel.dataset.lowpo_samples]
    pca = PCA(n_components=world.config['n_components'])
    item_emb_high_pca = pca.fit_transform(item_emb_high.cpu().numpy())
    item_emb_low_pca = pca.fit_transform(item_emb_low.cpu().numpy())
    plt.figure()
    plt.scatter(item_emb_high_pca[:,0],item_emb_high_pca[:,1],c='r',label='high popularity')
    plt.show()
    plt.savefig(f"../imgs/{world.dataset}/popularity_high_{world.config['emb_ans_pos']}.png")
    plt.figure()
    plt.scatter(item_emb_low_pca[:,0],item_emb_low_pca[:,1],c='b',label='low popularity')
    plt.show()
    plt.savefig(f"../imgs/{world.dataset}/popularity_low_{world.config['emb_ans_pos']}.png")


def plot_count_popularity_bar(popularity, counts, text):
    plt.figure()
    # 将 popularity 划分为多个区间 (0-200, 200-400, 400-600, ... )
    bins = np.arange(0, 1600, 200)

    # 使用 numpy 的 histogram 统计每个区间内的 count 总和
    bin_indices = np.digitize(popularity, bins)
    binned_counts = [0] * (len(bins) - 1)


    # 对每个 bin 累加相应的 count
    for i, count in enumerate(list(counts)):
        if 0 < bin_indices[i] <= len(binned_counts):
            binned_counts[bin_indices[i] - 1] += count

    total_count = sum(binned_counts)
    binned_probabilities = [count / total_count for count in binned_counts]

    # 绘制柱状图
    plt.bar(range(len(binned_counts)), binned_probabilities, width=0.8, align='center')

    # 添加标签和标题
    plt.xticks(range(len(binned_counts)), [f'{bins[i]}-{bins[i+1]}' for i in range(len(bins) - 1)], rotation=45)
    plt.xlabel('Popularity Range')
    plt.ylabel('Item Count')
    plt.title('Item Counts in Popularity Ranges')

    # 显示图表
    plt.tight_layout()
    plt.show()
    plt.savefig(f"../imgs/{world.dataset}/plot_{text}_items_popularity_bar.png")

def plot_count_popularity(popularity, counts, text):
    plt.figure()
    plt.scatter(counts, popularity, alpha=0.5)  
    plt.title('Item Popularity vs Recommendation Count')
    plt.ylabel('Popularity')
    plt.xlabel('Recommendation Count')
    # 显示图形
    plt.show()
    plt.savefig(f"../imgs/{world.dataset}/plot_{text}_items_popularity.png")


class timer:
    """
    Time context manager for code block
        with timer():
            do something
        timer.get()
    """
    from time import time
    TAPE = [-1]  # global time record
    NAMED_TAPE = {}

    @staticmethod
    def get():
        if len(timer.TAPE) > 1:
            return timer.TAPE.pop()
        else:
            return -1

    @staticmethod
    def dict(select_keys=None):
        hint = "|"
        if select_keys is None:
            for key, value in timer.NAMED_TAPE.items():
                hint = hint + f"{key}:{value:.2f}|"
        else:
            for key in select_keys:
                value = timer.NAMED_TAPE[key]
                hint = hint + f"{key}:{value:.2f}|"
        return hint

    @staticmethod
    def zero(select_keys=None):
        if select_keys is None:
            for key, value in timer.NAMED_TAPE.items():
                timer.NAMED_TAPE[key] = 0
        else:
            for key in select_keys:
                timer.NAMED_TAPE[key] = 0

    def __init__(self, tape=None, **kwargs):
        if kwargs.get('name'):
            timer.NAMED_TAPE[kwargs['name']] = timer.NAMED_TAPE[
                kwargs['name']] if timer.NAMED_TAPE.get(kwargs['name']) else 0.
            self.named = kwargs['name']
            if kwargs.get("group"):
                #TODO: add group function
                pass
        else:
            self.named = False
            self.tape = tape or timer.TAPE

    def __enter__(self):
        self.start = timer.time()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        if self.named:
            timer.NAMED_TAPE[self.named] += timer.time() - self.start
        else:
            self.tape.append(timer.time() - self.start)


# ====================Metrics==============================
# =========================================================
def RecallPrecision_ATk(test_data, r, k):
    """
    test_data should be a list? cause users may have different amount of pos items. shape (test_batch, k)
    pred_data : shape (test_batch, k) NOTE: pred_data should be pre-sorted
    k : top-k
    """
    right_pred = r[:, :k].sum(1)
    precis_n = k
    recall_n = np.array([len(test_data[i]) for i in range(len(test_data))])
    recall = np.sum(right_pred/recall_n)
    precis = np.sum(right_pred)/precis_n
    return {'recall': recall, 'precision': precis}


def MRRatK_r(r, k):
    """
    Mean Reciprocal Rank
    """
    pred_data = r[:, :k]
    scores = np.log2(1./np.arange(1, k+1))
    pred_data = pred_data/scores
    pred_data = pred_data.sum(1)
    return np.sum(pred_data)

def NDCGatK_r(test_data,r,k):
    """
    Normalized Discounted Cumulative Gain
    rel_i = 1 or 0, so 2^{rel_i} - 1 = 1 or 0
    """
    assert len(r) == len(test_data)
    pred_data = r[:, :k]

    test_matrix = np.zeros((len(pred_data), k))
    for i, items in enumerate(test_data):
        length = k if k <= len(items) else len(items)
        test_matrix[i, :length] = 1
    max_r = test_matrix
    idcg = np.sum(max_r * 1./np.log2(np.arange(2, k + 2)), axis=1)
    dcg = pred_data*(1./np.log2(np.arange(2, k + 2)))
    dcg = np.sum(dcg, axis=1)
    idcg[idcg == 0.] = 1.
    ndcg = dcg/idcg
    ndcg[np.isnan(ndcg)] = 0.
    return np.sum(ndcg)

def AUC(all_item_scores, dataset, test_data):
    """
        design for a single user
    """
    dataset : BasicDataset
    r_all = np.zeros((dataset.m_items, ))
    r_all[test_data] = 1
    r = r_all[all_item_scores >= 0]
    test_item_scores = all_item_scores[all_item_scores >= 0]
    return roc_auc_score(r, test_item_scores)

def getLabel(test_data, pred_data):
    r = []
    for i in range(len(test_data)):
        groundTrue = test_data[i]
        predictTopK = pred_data[i]
        #预测的每个值是否在groundtruth当中
        pred = list(map(lambda x: x in groundTrue, predictTopK))
        pred = np.array(pred).astype("float")
        r.append(pred)
    return np.array(r).astype('float')

# ====================end Metrics=============================
# =========================================================
