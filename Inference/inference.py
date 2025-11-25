# -*- encoding: utf-8 -*-
"""
@type: script

@description:

    Predict surface reflectance of specified date.
    Usage:
        python ./inference.py
        [-c[--config] config_path]
        [--checkpoint model_path]
        [-o[--outdir] output_path]
    All the used data [landsat / modis] should be in respective diretory with a nc format.
    Output data is saved in the [file_name] directory in [out_dir] with a tif format.
    for example:
        |---input_dir/
        |---|---landsat/
        |---|---|---[file_name].nc
        |---|---modis/
        |---|---|---MOD09A1/
        |---|---|---|---[file_name].nc
        |---|---|---MOD09Q1/
        |---|---|---|---[file_name].nc
        |---output_dir/
        |---|---[file_name]/
        |---|---|---[file_name]_[date].tif

@author: guanshikang

Created on Tue Oct 14 11:21:34 2025, HONG KONG
"""
import os
import re
import sys
sys.path.append(sys.path[0] + "/..")
import tqdm
import torch
import tomllib
import argparse
import torch.utils.data as data_utils
from InferenceDataLoader import InferenceDataLoader
from Backbone.SwinTransformer_v3 import SwinTransformer
from Benchmark.SwinSTFM.models.swinstfm import SwinSTFM
from FunctionalCode.CommonFuncs import CommonFuncs

def main():
    parser = argparse.ArgumentParser(description='Inference with referred model.')
    parser.add_argument("-c", "--config", default='./config.toml', help='Configuration path (toml version).')
    parser.add_argument("--checkpoint", help='Checkpoint path.')
    parser.add_argument("-o", "--outdir", help='Root path for output result.')
    args = parser.parse_args()

    config_path = args.config
    checkpoint_path = args.checkpoint
    out_dir = args.outdir

    try:
        with open(config_path, 'rb') as f:
            config = tomllib.load(f)
        window_sizes = [tuple(x) for x in config['model_params']['window_sizes']]
        config['model_params']['window_sizes'] = window_sizes
        out_dir = config['base']['save_dir'] if out_dir is None else out_dir
        checkpoint_path = config['base']['checkpoint_path'] if checkpoint_path is None else checkpoint_path
        device = torch.device(config['base']['device'] if torch.cuda.is_available() else "cpu")
    except FileNotFoundError:
        print("Error: Could not find config file.")
        os._exit(0)

    dataset = InferenceDataLoader(**config['dataloader'])
    dataloader = data_utils.DataLoader(
        dataset,
        batch_size=1,
        collate_fn=lambda batch: (
            torch.stack([item['landsat'] for item in batch]),
            torch.stack([item['modis_Q'] for item in batch]),
            torch.stack([item['modis_A'] for item in batch]),
            [item['out_name'] for item in batch]
        ),
        pin_memory=True,
        num_workers=2,
        persistent_workers=True,
        prefetch_factor=2
    )

    check_point = torch.load(checkpoint_path, map_location=device, weights_only=False)
    dataset.update_progressive_epoch(check_point['best_epoch'])
    model = SwinTransformer(**config['model_params'])
    # model = SwinSTFM()
    model.load_state_dict(check_point['model'])
    model.eval()
    model.to(device)
    for data in tqdm.tqdm(dataloader):
        if None not in data:
            landsat, modisQ, modisA, file_name = data
            with torch.no_grad():
                landsat = landsat.to(device, non_blocking=True)
                modisQ = modisQ.to(device, non_blocking=True)
                modisA = modisA.to(device, non_blocking=True)
                try:
                    logits = model(landsat, modisQ, modisA)
                    # c0 = modisA[:, :6, 0, ...]
                    # f0 = landsat[:, :6, 0, ...]
                    # c1 = modisA[:, :6, 1, ...]
                    # logits = model(c0, f0, c1)
                    # logits = (logits + 1.0) / 2.0
                except:
                    print("Lack Required Clean Landsat for next season.")
                    continue
            if isinstance(logits, torch.Tensor) and len(logits) > 0:
                logits = logits.cpu().numpy()
                for batch in range(logits.shape[0]):
                    pred = logits[batch]

                    suffix = os.path.splitext(file_name[batch])[1]
                    if len(suffix) == 0:
                        out_name = file_name[batch] + ".tif"
                    else:
                        out_name = file_name[batch]
                    pattern = r"^(?P<site_name>.*)_\d{8}"
                    site_name = re.match(pattern, out_name).group("site_name")
                    site_dir = os.path.join(out_dir, site_name)
                    if not os.path.exists(site_dir):
                        os.makedirs(site_dir)
                    out_path = os.path.join(site_dir, out_name)
                    CommonFuncs.save_image(out_path, pred)


if __name__ == "__main__":
    main()
