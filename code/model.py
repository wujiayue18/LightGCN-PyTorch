"""
Created on Mar 1, 2020
Pytorch Implementation of LightGCN in
Xiangnan He et al. LightGCN: Simplifying and Powering Graph Convolution Network for Recommendation

@author: Jianbai Ye (gusye@mail.ustc.edu.cn)

Define models here
"""
import world
import torch
from dataloader import BasicDataset
from torch import nn
import numpy as np
from itertools import chain

class BasicModel(nn.Module):    
    def __init__(self):
        super(BasicModel, self).__init__()
    
    def getUsersRating(self, users):
        raise NotImplementedError
    
class PairWiseModel(BasicModel):
    def __init__(self):
        super(PairWiseModel, self).__init__()
    def bpr_loss(self, users, pos, neg):
        """
        Parameters:
            users: users list 
            pos: positive items for corresponding users
            neg: negative items for corresponding users
        Return:
            (log-loss, l2-loss)
        """
        raise NotImplementedError
    def infoNCE_loss(self, anchor, pos, neg):
        raise NotImplementedError
    
class PureMF(BasicModel):
    def __init__(self, 
                 config:dict, 
                 dataset:BasicDataset):
        super(PureMF, self).__init__()
        self.num_users  = dataset.n_users
        self.num_items  = dataset.m_items
        self.latent_dim = config['latent_dim_rec']
        self.f = nn.Sigmoid()
        self.__init_weight()
        
    def __init_weight(self):
        self.embedding_user = torch.nn.Embedding(
            num_embeddings=self.num_users, embedding_dim=self.latent_dim)
        self.embedding_item = torch.nn.Embedding(
            num_embeddings=self.num_items, embedding_dim=self.latent_dim)
        print("using Normal distribution N(0,1) initialization for PureMF")
        
    def getUsersRating(self, users):
        users = users.long()
        users_emb = self.embedding_user(users)
        items_emb = self.embedding_item.weight
        scores = torch.matmul(users_emb, items_emb.t())
        return self.f(scores)
    
    def bpr_loss(self, users, pos, neg):
        users_emb = self.embedding_user(users.long())
        pos_emb   = self.embedding_item(pos.long())
        neg_emb   = self.embedding_item(neg.long())
        pos_scores= torch.sum(users_emb*pos_emb, dim=1)
        neg_scores= torch.sum(users_emb*neg_emb, dim=1)
        loss = torch.mean(nn.functional.softplus(neg_scores - pos_scores))
        reg_loss = (1/2)*(users_emb.norm(2).pow(2) + 
                          pos_emb.norm(2).pow(2) + 
                          neg_emb.norm(2).pow(2))/float(len(users))
        return loss, reg_loss
        
    def forward(self, users, items):
        users = users.long()
        items = items.long()
        users_emb = self.embedding_user(users)
        items_emb = self.embedding_item(items)
        scores = torch.sum(users_emb*items_emb, dim=1)
        return self.f(scores)

class LightGCN(BasicModel):
    def __init__(self, 
                 config:dict, 
                 dataset:BasicDataset):
        super(LightGCN, self).__init__()
        self.config = config
        self.dataset : dataloader.BasicDataset = dataset
        self.__init_weight()
        

    def __init_weight(self):
        self.num_users  = self.dataset.n_users
        self.num_items  = self.dataset.m_items
        self.latent_dim = self.config['latent_dim_rec']
        self.n_layers = self.config['lightGCN_n_layers']
        self.keep_prob = self.config['keep_prob']
        self.A_split = self.config['A_split']
        self.embedding_user = torch.nn.Embedding(
            num_embeddings=self.num_users, embedding_dim=self.latent_dim)
        self.embedding_item = torch.nn.Embedding(
            num_embeddings=self.num_items, embedding_dim=self.latent_dim)
        if self.config['pretrain'] == 0:
#             nn.init.xavier_uniform_(self.embedding_user.weight, gain=1)
#             nn.init.xavier_uniform_(self.embedding_item.weight, gain=1)
#             print('use xavier initilizer')
# random normal init seems to be a better choice when lightGCN actually don't use any non-linear activation function
            nn.init.normal_(self.embedding_user.weight, std=0.1)
            nn.init.normal_(self.embedding_item.weight, std=0.1)
            world.cprint('use NORMAL distribution initilizer')
        else:
            self.embedding_user.weight.data.copy_(torch.from_numpy(self.config['user_emb']))
            self.embedding_item.weight.data.copy_(torch.from_numpy(self.config['item_emb']))
            print('use pretarined data')
        self.f = nn.Sigmoid()
        self.Graph = self.dataset.getSparseGraph()
        print(f"lgn is already to go(dropout:{self.config['dropout']})")

        # print("save_txt")
    def __dropout_x(self, x, keep_prob):
        #indices,values,size
        size = x.size()
        index = x.indices().t()
        values = x.values()
        random_index = torch.rand(len(values)) + keep_prob
        random_index = random_index.int().bool()
        index = index[random_index]
        values = values[random_index]/keep_prob # # Scale the remaining values to account for dropout
        g = torch.sparse.FloatTensor(index.t(), values, size)
        return g
    
    def __dropout(self, keep_prob):
        if self.A_split:
            graph = []
            for g in self.Graph:
                graph.append(self.__dropout_x(g, keep_prob))
        else:
            graph = self.__dropout_x(self.Graph, keep_prob)
        return graph
    
    def computer(self):
        """
        propagate methods for lightGCN
        """       
        users_emb = self.embedding_user.weight
        items_emb = self.embedding_item.weight
        all_emb = torch.cat([users_emb, items_emb])
        #   torch.split(all_emb , [self.num_users, self.num_items])
        embs = [all_emb]
        if self.config['dropout']:
            if self.training:
                print("droping")
                g_droped = self.__dropout(self.keep_prob)
            else:
                g_droped = self.Graph        
        else:
            g_droped = self.Graph    
        
        for layer in range(self.n_layers):
            if self.A_split:
                temp_emb = []
                for f in range(len(g_droped)):
                    temp_emb.append(torch.sparse.mm(g_droped[f], all_emb))
                side_emb = torch.cat(temp_emb, dim=0)
                all_emb = side_emb
            else:
                all_emb = torch.sparse.mm(g_droped, all_emb)
            embs.append(all_emb)
        #layer1
        users_layer1,items_layer1 = torch.split(embs[1], [self.num_users, self.num_items])
        #layer2
        users_layer2,items_layer2 = torch.split(embs[2], [self.num_users, self.num_items])
        #layer3
        users_layer3,items_layer3 = torch.split(embs[3], [self.num_users, self.num_items])
        
        embs = torch.stack(embs, dim=1)
        #print(embs.size())
        light_out = torch.mean(embs, dim=1) #平均完
        users, items = torch.split(light_out, [self.num_users, self.num_items]) #再切分
        
        return users, items, users_layer1, items_layer1, users_layer2, items_layer2, users_layer3, items_layer3
    
    def getUsersRating(self, users):
        all_users, all_items,_,_,_,_,_,_ = self.computer()
        users_emb = all_users[users.long()]
        items_emb = all_items
        rating = self.f(torch.matmul(users_emb, items_emb.t()))
        return rating
    
    def getEmbedding(self, users, pos_items, neg_items):
        all_users, all_items,_,_,_,_,_,_ = self.computer()
        users_emb = all_users[users]
        pos_emb = all_items[pos_items]
        neg_emb = all_items[neg_items]
        users_emb_ego = self.embedding_user(users)
        pos_emb_ego = self.embedding_item(pos_items)
        neg_emb_ego = self.embedding_item(neg_items)
        return users_emb, pos_emb, neg_emb, users_emb_ego, pos_emb_ego, neg_emb_ego
    
    def bpr_loss(self, users, pos, neg):
        (users_emb, pos_emb, neg_emb, 
        userEmb0,  posEmb0, negEmb0) = self.getEmbedding(users.long(), pos.long(), neg.long())
        reg_loss = (1/2)*(userEmb0.norm(2).pow(2) + 
                         posEmb0.norm(2).pow(2)  +
                         negEmb0.norm(2).pow(2))/float(len(users))
        pos_scores = torch.mul(users_emb, pos_emb)
        pos_scores = torch.sum(pos_scores, dim=1)
        neg_scores = torch.mul(users_emb, neg_emb)
        neg_scores = torch.sum(neg_scores, dim=1)
        
        loss = torch.mean(torch.nn.functional.softplus(neg_scores - pos_scores))
        
        return loss, reg_loss

    # def infoNCE_loss(self, anchor, pos, neg):
    #     #anchor_emb size [batch_size, dim]
    #     all_users, all_items = self.computer()
    #     anchor_emb = all_items[anchor] # [batch_size, dim]
    #     pos_emb = all_items[pos]  # [batch_size, dim]
    #     neg_emb = all_items[neg]  # [batch_size, num_negatives, dim]
    #      # 正样本相似度 (dot product)
    #     anchor_positive_sim = torch.sum(torch.mul(anchor_emb, pos_emb), dim=1)  # [batch_size]
    #     anchor_positive_sim /= self.config['temperature']  # Temperature scaling

    #     # 负样本相似度 (use matrix multiplication for batch dot product)
    #     anchor_negative_sim = torch.bmm(neg_emb, anchor_emb.unsqueeze(2)).squeeze()  # [batch_size, num_negatives]
    #     anchor_negative_sim /= self.config['temperature']  # Temperature scaling

    #     # Concatenate positive and negative similarities
    #     logits = torch.cat([anchor_positive_sim.unsqueeze(1), anchor_negative_sim], dim=1)  # [batch_size, 1 + num_negatives]

    #     # Labels: Positive sample is at index 0
    #     labels = torch.zeros(logits.size(0), dtype=torch.long).to(logits.device)

    #     # Cross entropy loss
    #     loss = torch.nn.functional.cross_entropy(logits, labels)

    #     return loss
       
    def forward(self, users, items):
        # compute embedding
        all_users, all_items,_,_,_,_,_,_ = self.computer()
        # print('forward')
        #all_users, all_items = self.computer()
        users_emb = all_users[users]
        items_emb = all_items[items]
        inner_pro = torch.mul(users_emb, items_emb)
        gamma     = torch.sum(inner_pro, dim=1)
        return gamma


class SteerNet(nn.Module):
    def __init__(self,config, num_users, num_items):
        super(SteerNet, self).__init__()
        assert config['rank'] > 0
        self.adapter_class = config['adapter_class']
        self.epsilon = config['epsilon']
        self.which = config['which'] #user或者是item
        self.rank = config['rank']
        self.latent_dim = config['latent_dim_rec']
        self.num_steers = config['num_steers']
        self.init_var = config['init_var']
        self.num_users = num_users
        self.num_items = num_items
        self.steer_values = torch.zeros(self.num_steers)
        if self.which == 'user':
            self.vocab_size = num_users 
        else:
            self.vocab_size = num_items
        if self.adapter_class == 'multiply':
            self.projector1 = nn.Parameter(torch.randn(
                self.num_steers, self.latent_dim, self.rank  
            ) * self.init_var)
            self.projector2 = nn.Parameter(torch.randn(
                self.num_steers, self.latent_dim, self.rank  
            ) * self.init_var)
        elif self.adapter_class == 'add':
            self.add_vec = nn.Parameter(torch.randn(
                self.num_steers, self.latent_dim
            ))
        else:
            raise NotImplementedError

    def set_value(self, steer_values):
        self.steer_values = steer_values

    def forward(self, state):
        #[batch, latent_dim]
        if self.steer_values.abs().sum() == 0:
            return state.matmul(
                self.lm_head.weight.detach().transpose(0, 1))
        # if self.adapter_class == "multiply":
        #     delta = state[:, None,None,:].matmul(self.projector1[None])
        #     delta = delta * self.steer_values[:, :, None, None]
        #     delta = delta.matmul(
        #         self.projector2.transpose(1, 2)[None]).sum(1).squeeze()
        #     projected_state = state + self.epsilon * delta
        if self.adapter_class == "multiply":
            a = state[:, None,None,:]
            b = self.projector1[None]
            delta = a.matmul(b)
            delta = delta * self.steer_values[:, :, None, None]
            delta = delta.matmul(
                self.projector2.transpose(1, 2)[None]).sum(1).squeeze()
            projected_state = state + self.epsilon * delta
        elif self.adapter_class == "add":
            add_values = self.steer_values.matmul(self.add_vec)
            projected_state = state + self.epsilon * add_values

        return projected_state
    
    def regularization_term(self):
        if self.adapter_class == "multiply":
            return self.projector1.pow(2).sum() + self.projector2.pow(2).sum()
        elif self.adapter_class == "add":
            return self.add_vec.pow(2).sum()
        else:
            raise NotImplementedError
        
    def state_dict(self):
        if self.adapter_class == "multiply":
            return {"projector1": self.projector1,
                    "projector2": self.projector2}
        elif self.adapter_class == "add":
            return {"add_vec": self.add_vec}
        else:
            raise NotImplementedError
        
    def load_state_dict(self, state_dict):
        if self.adapter_class == "multiply":
            self.projector1.data = state_dict["projector1"]
            self.projector2.data = state_dict["projector2"]
        elif self.adapter_class == "add":
            self.add_vec.data = state_dict["add_vec"]
        else:
            raise NotImplementedError

class Steer_model(BasicModel):
    def __init__(self,
                 rec_model,
                 config:dict, 
                 dataset:BasicDataset,
                 steer_values):
        super(Steer_model,self).__init__()
        self.rec_model = rec_model
        self.dataset = dataset
        self.num_users = self.dataset.n_users
        self.num_items = self.dataset.m_items
        self.config = config
        if self.config['continue_train']:
            for _params in self.rec_model.parameters():
                _params.requires_grad = False
        self.init_user_embedding = self.get_parameter_by_name('embedding_user.weight')
        self.init_item_embedding = self.get_parameter_by_name('embedding_item.weight')
        self.steer = SteerNet(config,self.num_users,self.num_items).to(world.device)
        self.steer_values = steer_values
        self.steer.set_value(steer_values)
        self.f = nn.Sigmoid()

    def forward(self, users, items):
        all_users, all_items,_,_,_,_,_,_ = self.computer()
        # print('forward')
        #all_users, all_items = self.computer()
        users_emb = all_users[users]
        items_emb = all_items[items]
        inner_pro = torch.mul(users_emb, items_emb)
        gamma     = torch.sum(inner_pro, dim=1)
        return gamma

    def parameters(self):
        if self.config['continue_train']:
            return self.steer.parameters()
        else:
            return chain(self.rec_model.parameters(), self.steer.parameters())
    
    def state_dict(self):
        if self.config['continue_train']:
            return self.steer.state_dict()
        else:
            state_dict_combined = self.rec_model.state_dict()
            state_dict_combined.update(self.steer.state_dict())
            return state_dict_combined
            
    
    def load_state_dict(self, state_dict):
        if self.config['continue_train']:
            self.steer.load_state_dict(state_dict)
        else:
            rec_model_keys = ['embedding_user.weight', 'embedding_item.weight']
            steer_keys = ['projector1', 'projector2']

            # Filter state_dict for rec_model
            rec_model_state_dict = {k: v for k, v in state_dict.items() if k in rec_model_keys}
            self.rec_model.load_state_dict(rec_model_state_dict)

            # Filter state_dict for steer
            steer_state_dict = {k: v for k, v in state_dict.items() if k in steer_keys}
            self.steer.load_state_dict(steer_state_dict)

    def get_parameter_by_name(self, name):
    # 获取模型的状态字典
        state_dict = self.rec_model.state_dict()
        
        # 从字典中提取对应参数
        if name in state_dict:
            param = state_dict[name]
            # print(param.device)
            return param
        else:
            raise ValueError(f"Parameter '{name}' not found in the rec_model.")
        
    
    def getEmbedding(self, users, high_items, low_items, neg_items):
        all_users, all_items,_,_,_,_,_,_ = self.rec_model.computer()
        #加入steer
        # users_emb_ego = self.rec_model.embedding_user.weight
        # pos_emb_ego = self.rec_model.embedding_user.weight[pos_items]
        # neg_emb_ego = self.rec_model.embedding_user.weight[neg_items]
        if self.config['steer_train']:
            all_items = self.steer(all_items)
        users_emb = all_users[users]
        high_emb = all_items[high_items]
        low_emb = all_items[low_items]
        neg_emb = all_items[neg_items]
        # users_emb_ego = self.rec_model.embedding_user.weight[users]
        # pos_emb_ego = self.rec_model.embedding_item.weight[pos_items]
        # neg_emb_ego = self.rec_model.embedding_item.weight[neg_items]
        return users_emb, high_emb, low_emb, neg_emb
    
    def bpr_loss(self, users, high,low, neg):
        (users_emb, high_emb, low_emb, neg_emb) = self.getEmbedding(users.long(), high.long(),low.long(), neg.long())
        # reg_loss = (1/2)*(userEmb0.norm(2).pow(2) + 
        #                  posEmb0.norm(2).pow(2)  +
        #                  negEmb0.norm(2).pow(2))/float(len(users))
        high_scores = torch.mul(users_emb, high_emb)
        high_scores = torch.sum(high_scores, dim=1)
        low_scores = torch.mul(users_emb, low_emb)
        low_scores = torch.sum(low_scores, dim=1)

        pos_high_scores = torch.mul(users_emb, high_emb)
        pos_high_scores = torch.sum(pos_high_scores, dim=1)
        neg_high_scores = torch.mul(users_emb, neg_emb)
        neg_high_scores = torch.sum(neg_high_scores, dim=1)

        pos_low_scores = torch.mul(users_emb, low_emb)
        pos_low_scores = torch.sum(pos_low_scores, dim=1)
        neg_low_scores = torch.mul(users_emb, neg_emb)
        neg_low_scores = torch.sum(neg_low_scores, dim=1)

        loss1 = torch.mean(torch.nn.functional.softplus(low_scores - high_scores))
        loss2 = torch.mean(torch.nn.functional.softplus(neg_high_scores - pos_high_scores))
        loss3 = torch.mean(torch.nn.functional.softplus(neg_low_scores - pos_low_scores))

        loss = loss1 + loss2 + loss3
        reg_loss_steer = self.steer.regularization_term()
        print(f"loss: {loss.item()},loss1: {loss1.item()},loss2: {loss2.item()},loss3: {loss3.item()}, reg_loss_steer: {reg_loss_steer.item()}")

        return loss, reg_loss_steer

    def getUsersRating(self, users):
        all_users, all_items,_,_,_,_,_,_ = self.rec_model.computer()
        users_emb = all_users[users.long()]
        items_emb_ego = all_items
        
        items_emb = self.steer(all_items)
        
        rating_before = self.f(torch.matmul(users_emb, items_emb_ego.t()))
        
        rating = self.f(torch.matmul(users_emb, items_emb.t()))
        return rating
