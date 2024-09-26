import world
import utils
from world import cprint
import torch
import numpy as np
from tensorboardX import SummaryWriter
import time
import Procedure
from os.path import join
from datetime import datetime
# ==============================
utils.set_seed(world.seed)
print(">>SEED:", world.seed)
# ==============================
import register
from register import dataset
import logging
log_file = f'log-{datetime.now().strftime("%Y%m%d%H%M%S")}.txt'
logging.basicConfig(filename=join(world.LOG_PATH, log_file),
                    filemode='w', 
                    level=logging.INFO,
                    format='%(asctime)s - %(name)s - [%(pathname)s:%(lineno)d] - %(levelname)s - %(message)s')
logging.info(f"{world.config}")
Recmodel = register.MODELS[world.model_name](world.config, dataset)
Recmodel = Recmodel.to(world.device)
if world.config['steer_train']:
    bpr = utils.BPR2Loss(Recmodel, world.config)
else:
    bpr = utils.BPRLoss(Recmodel, world.config)

weight_file = utils.getFileName()
weight_file_r = './checkpoints/lgn-gowalla-3-64.pth.tar'
print(f"load from {weight_file_r}")
print(f"save to {weight_file}")
if world.LOAD or world.config['steer_train']:
    try:
        Recmodel.load_state_dict(torch.load(weight_file_r,map_location=torch.device('cpu')))
        world.cprint(f"loaded model weights from {weight_file_r}")
    except FileNotFoundError:
        print(f"{weight_file_r} not exists, start from beginning")
Neg_k = 1

# init tensorboard
if world.tensorboard:
    w : SummaryWriter = SummaryWriter(
                                    join(world.BOARD_PATH, time.strftime("%m-%d-%Hh%Mm%Ss-") + "-" + world.comment)
                                    )
else:
    w = None
    world.cprint("not enable tensorflowboard")

try:
    for epoch in range(world.TRAIN_epochs):
        start = time.time()
        if epoch %10 == 0:
            cprint("[TEST]")
            eval_results = Procedure.Test(dataset, Recmodel, epoch, w, world.config['multicore'])
            logging.info(f"EPOCH[{epoch+1}/{world.TRAIN_epochs}] {eval_results}")
        output_information = Procedure.BPR_train_original(dataset, Recmodel, bpr, epoch, neg_k=Neg_k,w=w)
        print(f'EPOCH[{epoch+1}/{world.TRAIN_epochs}] {output_information}')
        logging.info(f'EPOCH[{epoch+1}/{world.TRAIN_epochs}] {output_information}')
        torch.save(Recmodel.state_dict(), weight_file)
finally:
    if world.tensorboard:
        w.close()