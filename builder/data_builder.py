import torch
from dataloader.dataset2 import get_dataset_class, get_collate_class
from dataloader.pc_dataset import get_pc_model_class


def build(dataset_config, train_config):

    train_dataloader_config = dataset_config['train_data_loader']
    val_dataloader_config = dataset_config['val_data_loader']
    data_path = train_dataloader_config["data_path"]
    train_imageset = train_dataloader_config["imageset"]
    val_imageset = val_dataloader_config["imageset"]

    label_mapping = dataset_config["label_mapping"]

    SemKITTI = get_pc_model_class(dataset_config['pc_dataset_type'])

    if "nuScenes" not in dataset_config['pc_dataset_type']:
        train_pt_dataset = SemKITTI(data_path, imageset=train_imageset, label_mapping=label_mapping)
        val_pt_dataset = SemKITTI(data_path, imageset=val_imageset, label_mapping=label_mapping)
    elif "nuScenes" in dataset_config['pc_dataset_type']:
        train_pt_dataset = SemKITTI(dataset_config, data_path, imageset=train_imageset)
        val_pt_dataset = SemKITTI(dataset_config, data_path, imageset=val_imageset)

    train_dataset = get_dataset_class(dataset_config['dataset_type'])(
        train_pt_dataset,
        config=dataset_config,
        loader_config=train_dataloader_config
    )

    val_dataset = get_dataset_class(dataset_config['dataset_type'])(
        val_pt_dataset,
        config=dataset_config,
        loader_config=val_dataloader_config
    )

    train_sampler = torch.utils.data.distributed.DistributedSampler(train_dataset, num_replicas=train_config.world_size,
                                                                   rank=train_config.local_rank, shuffle=True)

    train_dataset_loader = torch.utils.data.DataLoader(dataset=train_dataset,
                                                    batch_size=train_dataloader_config["batch_size"],
                                                    collate_fn=get_collate_class(dataset_config['collate_type']),
                                                    num_workers=train_dataloader_config["num_workers"],
                                                    pin_memory=True,
                                                    sampler=train_sampler)
    val_dataset_loader = torch.utils.data.DataLoader(dataset=val_dataset,
                                                    batch_size=val_dataloader_config["batch_size"],
                                                    collate_fn=get_collate_class(dataset_config['collate_type']),
                                                    num_workers=val_dataloader_config["num_workers"],
                                                    pin_memory=True)

    return train_dataset_loader, val_dataset_loader, train_sampler

