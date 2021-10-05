
import torch
import time
from core.log import config_logger
from core.dataloader import DataLoader
from torch.optim.lr_scheduler import StepLR

def run(cfg, create_dataset, create_model, train, test, evaluator=None):
    if cfg.seed is not None:
        set_random_seed(cfg.seed)
        cfg.train.runs = 1 # no need to run same seed multiple times 

    # set num threads
    # torch.set_num_threads(cfg.num_workers)

    # 0. create logger and writer
    writer, logger, config_string = config_logger(cfg)

    # 1. create dataset
    train_dataset, val_dataset, test_dataset = create_dataset(cfg)

    # 2. create loader
    train_loader = DataLoader(train_dataset, cfg.train.batch_size, shuffle=True, num_workers=cfg.num_workers)
    val_loader = DataLoader(val_dataset,  cfg.train.batch_size//cfg.sampling.batch_factor, shuffle=False, num_workers=cfg.num_workers)
    test_loader = DataLoader(test_dataset, cfg.train.batch_size//cfg.sampling.batch_factor, shuffle=False, num_workers=cfg.num_workers)
    
    test_perfs = []
    vali_perfs = []
    for run in range(1, cfg.train.runs+1):
        # 3. create model and opt
        model = create_model(cfg).to(cfg.device)
        model.reset_parameters()
        optimizer = torch.optim.Adam(model.parameters(), lr=cfg.train.lr, weight_decay=cfg.train.wd)
        scheduler = StepLR(optimizer, step_size=cfg.train.lr_patience, gamma=cfg.train.lr_decay)

        # 4. train
        start_outer = time.time()
        best_val_perf = test_perf = float('-inf')
        for epoch in range(1, cfg.train.epochs+1):
            start = time.time()
            model.train()
            train_loss = train(train_loader, model, optimizer, device=cfg.device)
            scheduler.step()
            memory_allocated = torch.cuda.max_memory_allocated(cfg.device) // (1024 ** 2)
            memory_reserved = torch.cuda.max_memory_reserved(cfg.device) // (1024 ** 2)
            # print(f"---{test(train_loader, model, evaluator=evaluator, device=cfg.device) }")

            model.eval()
            val_perf = test(val_loader, model, evaluator=evaluator, device=cfg.device)
            if val_perf > best_val_perf:
                best_val_perf = val_perf
                test_perf = test(test_loader, model, evaluator=evaluator, device=cfg.device) 
            time_per_epoch = time.time() - start 

            # logger here
            print(f'Epoch: {epoch:03d}, Train Loss: {train_loss:.4f}, '
                  f'Val: {val_perf:.4f}, Test: {test_perf:.4f}, Seconds: {time_per_epoch:.4f}, '
                  f'Memory Peak: {memory_allocated} MB allocated, {memory_reserved} MB reserved.')

            # logging training
            writer.add_scalar(f'Run{run}/train-loss', train_loss, epoch)
            writer.add_scalar(f'Run{run}/val-perf', val_perf, epoch)
            writer.add_scalar(f'Run{run}/test-best-perf', test_perf, epoch)
            writer.add_scalar(f'Run{run}/seconds', time_per_epoch, epoch)   
            writer.add_scalar(f'Run{run}/memory', memory_allocated, epoch)   

            torch.cuda.empty_cache() # empty test part memory cost

        time_average_epoch = time.time() - start_outer
        print(f'Run {run}, Vali: {best_val_perf}, Test: {test_perf}, Seconds/epoch: {time_average_epoch/cfg.train.epochs}, Memory Peak: {memory_allocated} MB allocated, {memory_reserved} MB reserved.')
        test_perfs.append(test_perf)
        vali_perfs.append(best_val_perf)

    test_perf = torch.tensor(test_perfs)
    vali_perf = torch.tensor(vali_perfs)
    logger.info("-"*50)
    logger.info(config_string)
    # logger.info(cfg)
    logger.info(f'Final Vali: {vali_perf.mean():.4f} ± {vali_perf.std():.4f}, Final Test: {test_perf.mean():.4f} ± {test_perf.std():.4f},'
                f'Seconds/epoch: {time_average_epoch/cfg.train.epochs}, Memory Peak: {memory_allocated} MB allocated, {memory_reserved} MB reserved.')
    print(f'Final Vali: {vali_perf.mean():.4f} ± {vali_perf.std():.4f}, Final Test: {test_perf.mean():.4f} ± {test_perf.std():.4f},'
                f'Seconds/epoch: {time_average_epoch/cfg.train.epochs}, Memory Peak: {memory_allocated} MB allocated, {memory_reserved} MB reserved.')

import random, numpy as np
import warnings
def set_random_seed(seed=0, cuda_deterministic=True):
    """
    This function is only used for reproducbility, 
    DDP model doesn't need to use same seed for model initialization, 
    as it will automatically send the initialized model from master node to other nodes. 
    Notice this requires no change of model after call DDP(model)
    """
    random.seed(seed)
    torch.manual_seed(seed)
    np.random.seed(seed)
    if cuda_deterministic:
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
        warnings.warn('You have chosen to seed training with CUDNN deterministic setting,'
                      'which can slow down your training considerably! '
                      'You may see unexpected behavior when restarting '
                      'from checkpoints.')
    else:
        torch.backends.cudnn.deterministic = False
        torch.backends.cudnn.benchmark = True
        warnings.warn('You have chosen to seed training WITHOUT CUDNN deterministic. '
                       'This is much faster but less reproducible')
