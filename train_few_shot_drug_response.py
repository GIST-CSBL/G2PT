import argparse
import os
import random
import shutil
import time
import warnings
from enum import Enum
import pandas as pd

import argparse
import torch
import torch.nn.parallel
import torch.distributed as dist
import torch.optim
import torch.multiprocessing as mp
import torch.utils.data
import torch.utils.data.distributed

from prettytable import PrettyTable

from src.model.compound import ECFPCompoundModel, ChemBERTaCompoundModel
from src.model import DrugResponseModel, DrugResponseFewShotTransformer


from src.utils.data import CompoundEncoder
from src.utils.tree import MutTreeParser
from src.utils.data.dataset import DrugResponseDataset, DrugResponseCollator, DrugResponseSampler, DrugBatchSampler, DrugDataset, CellLineBatchSampler
from src.utils.trainer.drug_response_fewshot_learner import DrugResponseFewShotLearner
import numpy as np
import torch.nn as nn



from torch.utils.data.dataloader import DataLoader

def count_parameters(model):
    table = PrettyTable(["Modules", "Parameters", "Trainable"])
    total_params = 0
    for name, parameter in model.named_parameters():
        params = parameter.numel()
        #print(name, params, parameter.requires_grad)
        table.add_row([name, params, parameter.requires_grad])
        if parameter.requires_grad:
            total_params+=params
    print(table)
    print(f"Total Trainable Params: {total_params}")
    return total_params


def main():
    parser = argparse.ArgumentParser(description='Some beautiful description')
    parser.add_argument('--onto', help='Ontology file used to guide the neural network', type=str)
    parser.add_argument('--subtree_order', help='Subtree cascading order', nargs='+', default=['default'])
    parser.add_argument('--train', help='Training dataset', type=str)
    parser.add_argument('--few-shot', help='Training dataset', type=str)

    parser.add_argument('--epochs', help='Training epochs for training', type=int, default=300)
    parser.add_argument('--compound_epochs', type=int, default=10)
    parser.add_argument('--lr', help='Learning rate', type=float, default=0.001)

    parser.add_argument('--hidden_dims', help='hidden dimension for model', default=256, type=int)
    parser.add_argument('--dropout', help='dropout ratio', type=float, default=0.2)
    parser.add_argument('--batch_size', help='Batch size', type=int, default=128)
    parser.add_argument('--val_step', help='Batch size', type=int, default=1)

    parser.add_argument('--cuda', help='Specify GPU', type=int, default=None)
    parser.add_argument('--gene2id', help='Gene to ID mapping file', type=str)


    parser.add_argument('--cell2id', help='Cell to ID mapping file', type=str)
    parser.add_argument('--genotypes', help='Mutation information for cell lines', type=str)

    parser.add_argument('--few_shot_train', help='Cell to ID mapping file', type=str)
    parser.add_argument('--few_shot_test', help='Cell to ID mapping file', type=str)
    parser.add_argument('--few_shot_cell2id', help='Cell to ID mapping file', type=str)
    parser.add_argument('--few_shot_genotypes', help='Mutation information for cell lines', type=str)

    parser.add_argument('--bert', help='huggingface repository for smiles parsing', default=None)
    parser.add_argument('--radius', help='ECFP radius', type=int, default=2)
    parser.add_argument('--n_bits', help='ECFP number of bits', type=int, default=512)
    parser.add_argument('--compound_layers', help='Compound_dense_layer', nargs='+', default=[256], type=int)
    parser.add_argument('--l1_lambda', help='l1 lambda for l1 loss', type=float, default=0.001)
    parser.add_argument('--l2_lambda', help='l1 lambda for l1 loss', type=float, default=0.001)

    parser.add_argument('--model', help='model trained', default=None)

    parser.add_argument('--jobs', help="The number of threads", type=int, default=0)
    parser.add_argument('--out', help="output model path")


    parser.add_argument('--sys2cell', action='store_true', default=False)
    parser.add_argument('--cell2sys', action='store_true', default=False)
    parser.add_argument('--sys2gene', action='store_true', default=False)
    args = parser.parse_args()
    torch.cuda.empty_cache()

    args.gpu = args.cuda
    print("ECFP with radius %d, and %d bits used for compound encoding" % (args.radius, args.n_bits))
    compound_encoder = CompoundEncoder('Morgan', args.radius, args.n_bits)

    tree_parser = MutTreeParser(args.onto, args.gene2id)
    train_dataset = pd.read_csv(args.train, header=None, sep='\t')
    device = torch.device("cuda:%d" % args.gpu)
    drug_response_model = torch.load(args.model, map_location=device)
    few_shot_model = DrugResponseFewShotTransformer(args.hidden_dims, n_heads=8)

    print("Summary of trainable parameters")
    count_parameters(few_shot_model)
    if args.sys2cell:
        print("Model will use Sys2Cell")
    if args.cell2sys:
        print("Model will use Cell2Sys")
    if args.sys2gene:
        print("Model will use Sys2Gene")

    args.genotypes = {genotype.split(":")[0]: genotype.split(":")[1] for genotype in args.genotypes.split(',')}
    drug_response_dataset = DrugResponseDataset(train_dataset, args.cell2id, args.genotypes, compound_encoder,
                                                tree_parser)
    drug_response_collator = DrugResponseCollator(tree_parser, list(args.genotypes.keys()), compound_encoder)
    train_response_dataloader = DataLoader(drug_response_dataset, shuffle=False, batch_size=args.batch_size,
                                              num_workers=args.jobs, collate_fn=drug_response_collator)

    #few_shot_dataset = pd.read_csv(args.few_shot, header=None, sep='\t')

    #few_shot_df = pd.read_csv()#.sample(frac=1)

    few_shot_train = pd.read_csv(args.few_shot_train, header=None, sep='\t')#few_shot_dataset.iloc[:args.n_shot]
    few_shot_test = pd.read_csv(args.few_shot_test, header=None, sep='\t')#few_shot_dataset.iloc[args.n_shot:]
    args.few_shot_genotypes = {genotype.split(":")[0]: genotype.split(":")[1] for genotype in args.few_shot_genotypes.split(',')}
    few_shot_train_drug_response_dataset = DrugResponseDataset(few_shot_train, args.few_shot_cell2id, args.few_shot_genotypes, compound_encoder, tree_parser)
    few_shot_train_response_dataloader = DataLoader(few_shot_train_drug_response_dataset, shuffle=False, batch_size=args.batch_size,
                                           num_workers=args.jobs, collate_fn=drug_response_collator)

    few_shot_test_drug_response_dataset = DrugResponseDataset(few_shot_test, args.few_shot_cell2id,
                                                               args.few_shot_genotypes, compound_encoder, tree_parser)
    few_shot_test_response_dataloader = DataLoader(few_shot_test_drug_response_dataset, shuffle=False,
                                                    batch_size=args.batch_size,
                                                    num_workers=args.jobs, collate_fn=drug_response_collator)

    few_shot_learner = DrugResponseFewShotLearner(drug_response_model, few_shot_model, train_response_dataloader, few_shot_train_response_dataloader, few_shot_test_response_dataloader, device, args=args)
    few_shot_learner.train_few_shot(args.epochs, args.out)

if __name__ == '__main__':
    main()
