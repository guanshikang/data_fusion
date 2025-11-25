# -*- encoding: utf-8 -*-
"""
@type: script

@description:

    Train a custom model for Data Fusion task.
    Usage:
        python ./train.py
        [-c[--config] config_path]
        [--checkpoint model_path]
    All intermediate results are saved in [cofig[base][save_dir]].

@author: guanshikang

Created on Mon Oct 20 19:32:56 2025, HONG KONG
"""
import os
import tomllib
import argparse
from Trainer import Trainer
from losses.FusionLoss import FusionLoss

def parse_args():
    parser = argparse.ArgumentParser(description='Triain Model.')
    parser.add_argument("-c", "--config", default='./config.toml', help='Configuration path (toml version).')
    parser.add_argument("--checkpoint", help='If given, this checkpoint will be loaded first.')
    args = parser.parse_args()

    cfg_path = args.config
    checkpoint_path = args.checkpoint

    try:
        with open(cfg_path, 'rb') as f:
            cfg = tomllib.load(f)
        try:
            window_sizes = [tuple(x) for x in cfg['model_params']['window_sizes']]
            cfg['model_params']['window_sizes'] = window_sizes
        except (TypeError, ValueError) as e:
            print(f"Error: Invalid window_sizes format in config: {e}")
            os._exit(1)
        if checkpoint_path:
            cfg['status']['checkpoint'] = True
        elif cfg['status']['checkpoint']:
            checkpoint_path = cfg['base']['checkpoint_path']  # will check in Trainer
    except FileNotFoundError:
        print("Error: Could not find config file.")
        os._exit(1)

    return cfg, checkpoint_path

def main():
    cfg, checkpoint_path = parse_args()
    criterion = FusionLoss(cfg['loss_weights'])
    trainer = Trainer(cfg, checkpoint_path, criterion=criterion)
    trainer.build()
    trainer.run_train()
    trainer.run_test()
    print("Training Completed!")

if __name__ == "__main__":
    main()
