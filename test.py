import torch
import datasets
from argparse import ArgumentParser
import utils
import imageio
import time
import os
import re
import numpy as np


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

if __name__ == "__main__":
    parser = ArgumentParser("darts")
    parser.add_argument("--batch-size", default=1, type=int)
    parser.add_argument("--workers", default=1)
    parser.add_argument("--savepath", default=r'./compare_img')

    result_path = 'D:/Code/project/pnas-nni/results/test_results/{}'.format(time.strftime('%Y%m%d-%H'))

    args = parser.parse_args()

    dataset_test = datasets.get_dataset("brats2015", cutout_length=0, test=True)

    model = torch.load('./final model/checkpoint_20210914-024655.json')
    model.to(device)

    test_loader = torch.utils.data.DataLoader(dataset_test,
                                               batch_size=args.batch_size,
                                               shuffle=False,
                                               num_workers=args.workers,
                                               pin_memory=True)

    model.eval()
    results = []
    for step, (x, y) in enumerate(test_loader):
        x, y = x.to(device, non_blocking=True), y.to(device, non_blocking=True)
        path = test_loader.dataset.Flair_path[step]
        logits = model(x)
        result = utils.show_graph(logits)
        result = np.expand_dims(result, axis=0)
        results.append(result)
        array_results = np.concatenate(results, axis=0)
        # test_graph(results, args.savepath)
        if (step+1) % 155 == 0:
            utils.saveItk(array_results, path, result_path)
            results = []


    print('Test complete')





