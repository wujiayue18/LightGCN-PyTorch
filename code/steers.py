import torch
import torch.nn as nn
import torch.nn.functional as F

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


    
