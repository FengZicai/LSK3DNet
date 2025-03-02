import torch
from torch.optim.lr_scheduler import ReduceLROnPlateau, StepLR, CosineAnnealingLR, OneCycleLR, LambdaLR
from utils.schedulers import cosine_schedule_with_warmup


def build(configs, model):
    if configs['train_params']['optimizer'] == 'Adam':
        optimizer = torch.optim.Adam(model.parameters(),
                                        lr=configs['train_params']["learning_rate"],
                                        weight_decay=configs['train_params']["weight_decay"])
    elif configs['train_params']['optimizer'] == 'AdamW':
        optimizer = torch.optim.AdamW(model.parameters(),
                                        lr=configs['train_params']["learning_rate"])
    elif configs['train_params']['optimizer'] == 'SGD':
        optimizer = torch.optim.SGD(model.parameters(),
                                    lr=configs['train_params']["learning_rate"],
                                    momentum=configs['train_params']["momentum"],
                                    weight_decay=configs['train_params']["weight_decay"],
                                    nesterov=configs['train_params']["nesterov"])
    else:
        raise NotImplementedError

    if configs['train_params']["lr_scheduler"] == 'StepLR':
        lr_scheduler = StepLR(
            optimizer,
            step_size=configs['train_params']["decay_step"],
            gamma=configs['train_params']["decay_rate"]
        )
    elif configs['train_params']["lr_scheduler"] == 'ReduceLROnPlateau':
        lr_scheduler = ReduceLROnPlateau(
            optimizer,
            mode='max',
            factor=configs['train_params']["decay_rate"],
            patience=configs['train_params']["decay_step"],
            verbose=True
        )
    elif configs['train_params']["lr_scheduler"] == 'CosineAnnealingLR':
        lr_scheduler = CosineAnnealingLR(
            optimizer,
            T_max=configs['train_params']['max_num_epochs'] - 4,
            eta_min=1e-6,
        )
    elif configs['train_params']["lr_scheduler"] == 'OneCycleLR':
        lr_scheduler = OneCycleLR(
                optimizer,
                max_lr=configs['train_params']["learning_rate"],
                total_steps=configs.train_params.total_steps,
                pct_start=0.05,
                anneal_strategy='cos',
                cycle_momentum=True,
                base_momentum=0.85,
                max_momentum=0.95,
                div_factor=10.0,
                final_div_factor=1000.0,
                three_phase=False,
                last_epoch=-1,
                verbose=False)
    elif configs['train_params']["lr_scheduler"] == 'CosineAnnealingWarmRestarts':
        from functools import partial
        lr_scheduler = LambdaLR(
            optimizer, lr_lambda=partial(
                cosine_schedule_with_warmup,
                num_epochs=configs['train_params']['max_num_epochs'],
                batch_size=configs['dataset_params']['train_data_loader']['batch_size'],
                dataset_size=configs['dataset_params']['training_size'],
                num_gpu=configs.train_params.world_size
            )
        )
    elif configs['train_params']["lr_scheduler"] == 'None':
        return optimizer, None
    else:
        raise NotImplementedError

    return optimizer, lr_scheduler

