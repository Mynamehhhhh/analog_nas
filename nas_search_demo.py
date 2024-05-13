# from analogainas.search_spaces.config_space import ConfigSpace
# from analogainas.evaluators.xgboost import XGBoostEvaluator
# from analogainas.search_algorithms.ea_optimized import EAOptimizer
# from analogainas.search_algorithms.worker import Worker
# from
#
# CS = ConfigSpace('CIFAR-10')  # Search Space Definition
# surrogate = XGBoostEvaluator(model_type="XGBRanker", load_weight=True) #
# optimizer = EAOptimizer(surrogate, population_size=20, nb_iter=50) # The default population size is 100.
#
# nb_runs = 2
# worker = Worker(CS, optimizer=optimizer, runs=nb_runs)
#
# worker.search()
# worker.result_summary()
#
# best_config = worker.best_config
# best_model = worker.best_arch
#
# import torchvision
# import torchvision.transforms as transforms
# from torch.utils.data import DataLoader
# from analogainas.search_spaces.dataloaders.cutout import Cutout
#
# import importlib.util
# pyvww = importlib.util.find_spec("pyvww")
# found = pyvww is not None
#
# def load_cifar10(batch_size):
#     transform_train = transforms.Compose([
#         transforms.RandomCrop(32, padding=4),
#         transforms.RandomHorizontalFlip(),
#         transforms.ToTensor(),
#         transforms.Normalize((0.4914, 0.4822, 0.4465),
#                              (0.2023, 0.1994, 0.2010)),
#         Cutout(1, length=8)
#     ])
#
#     transform_test = transforms.Compose([
#         transforms.ToTensor(),
#         transforms.Normalize((0.4914, 0.4822, 0.4465),
#                              (0.2023, 0.1994, 0.2010)),
#     ])
#
#     trainset = torchvision.datasets.CIFAR10(root='./data',
#                                             train=True,
#                                             download=True,
#                                             transform=transform_train)
#
#     trainloader = DataLoader(
#         trainset, batch_size=batch_size, shuffle=True, num_workers=2)
#
#     testset = torchvision.datasets.CIFAR10(
#         root='./data', train=False, download=True, transform=transform_test)
#
#     testloader = DataLoader(
#         testset, batch_size=100, shuffle=False, num_workers=2)
#
#     return trainloader, testloader
#
#
# classes = ('plane', 'car', 'bird', 'cat', 'deer',
#            'dog', 'frog', 'horse', 'ship', 'truck')
#
#
# def load_vww(batch_size, path, annot_path):
#     transform = transforms.Compose([
#         transforms.CenterCrop(100),
#         transforms.ToTensor()
#     ])
#
#     train_dataset = pyvww.pytorch.VisualWakeWordsClassification(
#                     root=path, annFile=annot_path, transform=transform)
#     valid_dataset = pyvww.pytorch.VisualWakeWordsClassification(
#                     root=path, annFile=annot_path, transform=transform)
#
#     train_loader = DataLoader(train_dataset,
#                               batch_size=batch_size,
#                               shuffle=True,
#                               num_workers=1)
#     valid_loader = DataLoader(valid_dataset,
#                               batch_size=batch_size,
#                               shuffle=False,
#                               num_workers=1)
#
#     return train_loader, valid_loader
#
# import random
# import numpy as np
# from analogainas.search_spaces.resnet_macro_architecture import Network
# from analogainas.search_spaces.config_space import ConfigSpace
# from analogainas.search_algorithms.worker import Worker
# from analogainas.search_spaces.train import train
# from analogainas.utils import *
# import csv
#
# EPOCHS = 40
# LEARNING_RATE = 0.05
#
# def latin_hypercube_sample(dataset, n):
#     """Latin Hypercube Sampling of n architectures from ConfigSpace."""
#     cs = ConfigSpace(dataset)
#     num_parameters = len(cs.get_hyperparameters())
#     ranges = np.arange(0, 1, 1/n)
#
#     sampled_architectures = []
#     for _ in range(n):
#         config = {}
#         for i, hyperparameter in enumerate(cs.get_hyperparameters()):
#             min_val, max_val = hyperparameter.lower, hyperparameter.upper
#             val_range = max_val - min_val
#             offset = random.uniform(0, val_range/n)
#             config[hyperparameter.name] = min_val + ranges[_] * val_range + offset
#         sampled_architectures.append(config)
#
#     keys = sampled_architectures[0].keys()
#
#     for config in sampled_architectures:
#         model = Network(config)
#         model_name = "resnet_{}_{}".format(config["M"], get_nb_convs(config))
#
#         with open("./configs/"+model_name+".config",
#                   'w', newline='') as output_file:
#             dict_writer = csv.DictWriter(output_file, keys)
#             dict_writer.writeheader()
#             dict_writer.writerows(config)
#
#         train(model, model_name, LEARNING_RATE, EPOCHS)
#
#
# def random_sample(dataset, n):
#     """Randomly samples n architectures from ConfigSpace."""
#     cs = ConfigSpace(dataset)
#     sampled_architectures = cs.sample_arch_uniformly(n)
#
#     keys = sampled_architectures[0].keys()
#
#     for config in sampled_architectures:
#         model = Network(config)
#         model_name = "resnet_{}_{}".format(config["M"], get_nb_convs(config))
#
#         with open("./configs/"+model_name+".config",
#                   'w', newline='') as output_file:
#             dict_writer = csv.DictWriter(output_file, keys)
#             dict_writer.writeheader()
#             dict_writer.writerows(config)
#
#         train(model, model_name, LEARNING_RATE, EPOCHS)
#
#
# def ea_sample(dataset, n, n_iter):
#     """Samples n architectures from ConfigSpace
#     using an evolutionary algorithm."""
#     cs = ConfigSpace(dataset)
#     worker = Worker(dataset, cs, 3, n_iter)
#     worker.search(population_size=n)


from search_spaces.config_space import ConfigSpace
from analogainas.evaluators.xgboost import XGBoostEvaluator
from analogainas.search_algorithms.ea_optimized import EAOptimizer
from analogainas.search_algorithms.worker import Worker
#
from search_spaces.dataloaders.dataloader import load_cifar10,load_vww
import importlib.util

from torch.utils.data import DataLoader
from search_spaces.dataloaders.dataloader import load_vww

import torch
import pyvww

CS = ConfigSpace('CIFAR-10')  # Search Space Definition
surrogate = XGBoostEvaluator(model_type="XGBRanker", load_weight=True) #
optimizer = EAOptimizer(surrogate, population_size=20, nb_iter=50) # The default population size is 100.

nb_runs = 2
worker = Worker(CS, optimizer=optimizer, runs=nb_runs)

worker.search()
worker.result_summary()

best_config = worker.best_config
best_model = worker.best_arch

#######
evaluator = XGBoostEvaluator(model_type=best_model)
print(evaluator.ranker)
print(evaluator.avm_predictor)
print(evaluator.std_predictor)

#######
# from
# pyvww=importlib.util.find_spec('pyvww')
# found=pyvww is not None
# batchsize = 100
# dataset_path='/root/data1/ZJ/analog-nas-main/analogainas/evaluators/weights/1'
# annotation_path='/root/data1/ZJ/analog-nas-main/analogainas/evaluators/weights/surrogate_xgboost_avm.json'

# dataloader = DataLoader(load_cifar10, batch_size=batchsize, shuffle=True)
# train_loader = DataLoader(load_cifar10, batch_size=batchsize, shuffle=True)
# valid_loader = DataLoader(load_cifar10, batch_size=batchsize, shuffle=True)
# train_loader,valid_loader=load_vww(batchsize,dataset_path,annotation_path)
