import os
import numpy as np
# os.environ["CUDA_VISIBLE_DEVICES"] = "5"
import math
import argparse
import random
import pprint
import torch
import torch.optim as optim
from torch.utils.data import DataLoader
import torch.backends.cudnn as cudnn

from lib.config.default import config as cfg
from lib.config.default import update_config
from lib.utils.utils import backup_codes

from lib.core.function import train, evaluation

from lib.models.mtlnet1 import MTLNet1
from lib.dataset.ABAW7_dataset import ABAWDataset
from lib.core.results import main_metric


def seed_torch(seed=1029):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed) # if you are using multi-GPU.
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.enabled = True

# for reproduction, same as orig. paper setting
seed_torch()

def main(config):
    best_results = 0
    update_config(config)
    # create output directory
    if cfg.BASIC.CREATE_OUTPUT_DIR:
        out_dir = os.path.join(cfg.BASIC.ROOT_DIR, cfg.TRAIN.MODEL_DIR)
        if not os.path.exists(out_dir):
            os.makedirs(out_dir)
    # copy config file
    if cfg.BASIC.BACKUP_CODES:
        backup_dir = os.path.join(cfg.BASIC.ROOT_DIR, cfg.TRAIN.MODEL_DIR, 'code')
        backup_codes(cfg.BASIC.ROOT_DIR, backup_dir, cfg.BASIC.BACKUP_LISTS)
    if cfg.BASIC.SHOW_CFG:
        pprint.pprint(cfg)

    # model
    model = MTLNet1(cfg)
    model.cuda()

    # optimizer
    optimizer = optim.AdamW(model.parameters(), lr=cfg.TRAIN.LR, weight_decay=cfg.TRAIN.WEIGHT_DECAY)
    warm_up_with_cosine_lr = lambda \
            epoch: epoch / cfg.TRAIN.WARM_UP_EPOCH if epoch <= cfg.TRAIN.WARM_UP_EPOCH else 0.5 * (math.cos(
        (epoch - cfg.TRAIN.WARM_UP_EPOCH) / (cfg.TRAIN.END_EPOCH - cfg.TRAIN.WARM_UP_EPOCH) * math.pi) + 1)
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=warm_up_with_cosine_lr)

    # data loader
    train_data = ABAWDataset(cfg, cfg.DATASET.TRAIN_SPLIT)
    train_loader = DataLoader(train_data, batch_size=cfg.TRAIN.BATCH_SIZE,
                              shuffle=True, drop_last=False, num_workers=cfg.BASIC.WORKERS,
                              pin_memory=cfg.DATASET.PIN_MEMORY)

    val_data = ABAWDataset(cfg, cfg.DATASET.VAL_SPLIT)
    val_loader = DataLoader(val_data, batch_size=cfg.TRAIN.BATCH_SIZE,
                            shuffle=False, drop_last=False, num_workers=cfg.BASIC.WORKERS,
                            pin_memory=cfg.DATASET.PIN_MEMORY)

    all_log_path = os.path.join(cfg.BASIC.ROOT_DIR, cfg.TRAIN.MODEL_DIR, 'experiment_data.log')
    if os.path.exists(all_log_path):
        os.remove(all_log_path)

    device = torch.device('cuda' if torch.cuda.is_available() else "cpu")
    dinov2_model = torch.hub.load('facebookresearch/dinov2', 'dinov2_vitg14').to(device)
    dinov2_model.eval()

    for epoch in range(cfg.TRAIN.END_EPOCH):
        loss_train, au_loss, va_loss, expr_loss = train(train_loader, model, optimizer, dinov2_model)
        scheduler.step()
        output = evaluation(val_loader, model, dinov2_model)

        all_results , f1_expr_avg, f1_au_avg, avg_va = main_metric(output, cfg.DATASET.ANN_PATH)

        # 保存模型
        weight_path = os.path.join(cfg.BASIC.ROOT_DIR, cfg.TRAIN.MODEL_DIR, f'epoch_{epoch}_val.pth')
        if os.path.exists(weight_path):
            os.remove(weight_path)
        torch.save(model.state_dict(), weight_path)

        if all_results > best_results:
            best_results = all_results
        log_path = os.path.join(cfg.BASIC.ROOT_DIR, cfg.TRAIN.MODEL_DIR, f'epoch_{epoch}_val.txt')
        if os.path.exists(log_path):
            os.remove(log_path)
        with open(log_path, 'w') as file:
            for i in output:
                for line in i:
                    line_str = ' '.join(map(str, line)) + '\n'
                    file.write(line_str)
        print('Epoch %d: train_loss: %.4f train_AU loss: %.4f train_EXPR loss: %.4f train_VA loss: %.4f '
              'au: %05f expr: %05f va:%05f all: %05f best: %05f' % (
            epoch, loss_train, au_loss, expr_loss, va_loss, f1_au_avg, f1_expr_avg, avg_va, all_results, best_results))

        with open(all_log_path, 'a') as f:
            f.write('Epoch %d: train_loss: %.4f train_AU loss: %.4f train_EXPR loss: %.4f train_VA loss: %.4f'
                    ' au: %05f expr: %05f va:%05f all: %05f best: %05f\n ' % (
            epoch, loss_train, au_loss, expr_loss, va_loss, f1_au_avg, f1_expr_avg, avg_va, all_results, best_results))



if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='MER SPOT')
    parser.add_argument('--cfg', type=str, help='experiment config file',
                        default="/home/geek/spot/abaw_github/abaw1.yaml")

    args = parser.parse_args()

    new_cfg = args.cfg
    print(args.cfg)
    main(new_cfg)


