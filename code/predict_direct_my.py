import world
import utils
from world import cprint
import torch
import numpy as np
from tensorboardX import SummaryWriter
import time
import Procedure
from os.path import join
# ==============================
utils.set_seed(world.seed)
print(">>SEED:", world.seed)
# ==============================
import register
from register import dataset

Recmodel = register.MODELS[world.model_name](world.config, dataset)
Recmodel = Recmodel.to(world.device)
weight_file = './checkpoints/lgn-gowalla-3-64.pth.tar'
world.cprint(f"loaded model weights from {weight_file}")
Recmodel.load_state_dict(torch.load(weight_file,map_location=torch.device('cpu')))
Procedure.Test(dataset, Recmodel,0)



