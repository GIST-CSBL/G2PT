import pandas as pd
import numpy as np
from torch.utils.data.dataset import Dataset
from torch.utils.data.sampler import Sampler, BatchSampler
from scipy.stats import zscore, skewnorm
import torch
from src.utils.tree import TreeParser, SNPTreeParser
from torch.nn.utils.rnn import pad_sequence
from torch.nn.functional import one_hot
import time
import networkx as nx
from random import shuffle
from ast import literal_eval
from torch.utils.data.distributed import DistributedSampler


class SNP2PDataset(Dataset):

    def __init__(self, genotype_phenotype, snp_data, tree_parser:SNPTreeParser, effective_allele='heterozygous', return_indice=True):
        self.g2p_df = genotype_phenotype
        self.tree_parser = TreeParser
        self.snp_df = snp_data
        self.tree_parser = tree_parser
        self.effective_allele = effective_allele
        self.return_indice=return_indice


    def __len__(self):
        return self.g2p_df.shape[0]

    def __getitem__(self, index):
        start = time.time()
        sample_ind, phenotype, sex, age, age_sq, *covariates = self.g2p_df.iloc[index].values
        #print(index, sample_ind, phenotype, sex, covariates)
        sample2snp_dict = {}
        homozygous_a1 = self.snp_df.loc[sample_ind, 'homozygous_a1']
        if type(homozygous_a1)==str:
            homozygous_a1 = [int(i) for i in homozygous_a1.split(",")]
        else:
            homozygous_a1 = []
        #homozygous_a2 = [int(i) for i in (self.snp_df.loc[sample_ind, 'homozygous_a2']).split(",")]
        #heterozygous = [int(i) for i in (self.snp_df.loc[sample_ind, 'heterozygous']).split(",")]
        heterozygous = self.snp_df.loc[sample_ind, 'heterozygous']
        if type(heterozygous) == str:
            heterozygous = [int(i) for i in heterozygous.split(",")]
        else:
            heterozygous = []
        '''
        type_indices = {1.0:homozygous}
        if self.effective_allele=='heterozygous':
            heterozygous = [int(i) for i in (self.snp_df.loc[sample_ind, 'heterozygous']).split(
                ",")]  # np.where(self.snp_df.loc[sample_ind].values == 1.0)[0]
            total_snps = np.concatenate([heterozygous,homozygous])
            type_indices[2.0] = homozygous
            type_indices[1.0] = heterozygous
        else:
            total_snps = homozygous
        sample2snp_dict['embedding'] = self.tree_parser.get_snp2gene(total_snps, type_indices=type_indices )
        '''
        snp_type_dict = {}
        if self.return_indice:
            snp_type_dict['homozygous_a1'] = self.tree_parser.get_snp2gene(homozygous_a1, {1.0: homozygous_a1})
            #snp_type_dict['homozygous_a0'] = self.tree_parser.get_snp2gene(homozygous_a2, {1.0: homozygous_a2})
            snp_type_dict['heterozygous'] = self.tree_parser.get_snp2gene(heterozygous, {1.0: heterozygous})
            sample2snp_dict['embedding'] = snp_type_dict
            # homozygous_a2_gene_indices = torch.unique(snp_type_dict['homozygous_a0']['gene']).tolist()
            # homozygous_a1_gene_indices = self.tree_parser.get_snp2gene_indices(homozygous_a1)
            # homozygous_a2_gene_indices = self.tree_parser.get_snp2gene_indices(homozygous_a2)
            # heterozygous_gene_indices = self.tree_parser.get_snp2gene_indices(heterozygous)
            # gene2sys_mask_for_gene[:, homozygous_a2_gene_indices] = 1
            # if self.effective_allele=='heterozygous':
            '''
            heterozygous_gene_indices = torch.unique(snp_type_dict['heterozygous']['gene']).tolist()
            homozygous_a1_gene_indices = torch.unique(snp_type_dict['homozygous_a1']['gene']).tolist()
            gene2sys_mask_for_gene = torch.zeros((self.tree_parser.n_systems, self.tree_parser.n_genes),
                                                 dtype=torch.bool)
            gene2sys_mask_for_gene[:, homozygous_a1_gene_indices] = 1
            gene2sys_mask_for_gene[:, heterozygous_gene_indices] = 1
            
            sample2snp_dict["gene2sys_mask"] = torch.tensor(self.tree_parser.gene2sys_mask, dtype=torch.bool) # & gene2sys_mask_for_gene
            '''
            sample2snp_dict["gene2sys_mask"] = torch.tensor(self.tree_parser.gene2sys_mask, dtype=torch.bool)
        else:
            homozygous_a1_binary = torch.tensor([1 if i in homozygous_a1 else 0 for i in range(self.tree_parser.n_snps)])
            sample2snp_dict['homozygous_a1'] = self.tree_parser.get_gene2snp_mask(homozygous_a1_binary)
            heterozygous_binary = torch.tensor([1 if i in homozygous_a1 else 0 for i in range(self.tree_parser.n_snps)])
            sample2snp_dict['heterozygous'] = self.tree_parser.get_gene2snp_mask(heterozygous_binary)
            #sample2snp_dict["gene2sys_mask"] = torch.tensor(self.tree_parser.gene2sys_mask, dtype=torch.bool)
        result_dict = dict()
        result_dict['phenotype'] = phenotype
        #sex_age_tensor = [0, 0, 0, 0]
        sex_age_tensor = [0, 0]
        if int(sex)==-9:
            pass
        else:
            sex_age_tensor[int(sex)] = 1
        #sex_age_tensor[2] = age
        #sex_age_tensor[3] = age_sq
        sex_age_tensor = torch.tensor(sex_age_tensor, dtype=torch.float32)
        covariates = sex_age_tensor#torch.cat([sex_age_tensor, torch.tensor(covariates, dtype=torch.float32)])

        result_dict['genotype'] = sample2snp_dict
        end = time.time()
        result_dict["datatime"] = torch.tensor(end-start)
        result_dict["covariates"] = covariates

        return result_dict

class SNP2PCollator(object):

    def __init__(self, tree_parser:SNPTreeParser, sampling=1, return_indices=True, subtree_order=['default']):
        self.tree_parser = tree_parser
        self.padding_index = {"snp":self.tree_parser.n_snps, "gene":self.tree_parser.n_genes}
        self.sampling_ratio = sampling
        self.return_indices = return_indices
        self.nested_subtrees_forward = self.tree_parser.get_nested_subtree_mask(subtree_order, direction='forward')
        self.nested_subtrees_backward = self.tree_parser.get_nested_subtree_mask(subtree_order, direction='backward')
        self.gene2sys_mask = torch.tensor(self.tree_parser.gene2sys_mask, dtype=torch.bool)
        self.sys2gene_mask = torch.tensor(self.tree_parser.sys2gene_mask, dtype=torch.bool)
        self.subtree_order = subtree_order

    def __call__(self, data):
        start = time.time()
        result_dict = dict()
        genotype_dict = dict()
        if self.sampling_ratio!=1.:
            def find_root_node(G):
                # This function finds the root node in a directed graph (DAG)
                for node in G.nodes():
                    if G.in_degree(node) == 0:
                        return node
                return None
            root_node = find_root_node(self.tree_parser.sys_graph)
            first_children = list(self.tree_parser.sys_graph.successors(root_node))
            n_children = int(len(first_children) * self.sampling_ratio)
            sampled_first_children = np.random.permutation(first_children)[:n_children]
            sampled_systems = list(set(sum([list(nx.descendants(self.tree_parser.sys_graph, first_child)) for first_child in sampled_first_children], [])))
            sampled_systems.append(root_node)
            sampled_sys_inds = sorted([self.tree_parser.sys2ind[sys] for sys in sampled_systems])
            sampled_gene_inds = sorted(list(set(sum([self.tree_parser.sys2gene_full_dict[sys_ind] for sys_ind in sampled_sys_inds ], []))))
            sampled_snp_inds = sorted(list(set(sum([self.tree_parser.sys2gene_full_dict[sys_ind] for sys_ind in sampled_sys_inds ], []))))
        else:
            sampled_sys_inds = list(self.tree_parser.sys2ind.values())
            sampled_gene_inds = list(self.tree_parser.gene2ind.values())
            sampled_snp_inds = list(self.tree_parser.snp2ind.values())

        if self.return_indices:
            snp_type_dict = {}
            for snp_type in ['heterozygous', 'homozygous_a1']:
                embedding_dict = {}
                for embedding_type in ['snp', 'gene']:
                    embedding_dict[embedding_type] = pad_sequence(
                            [d['genotype']['embedding'][snp_type][embedding_type] for d in data], batch_first=True,
                            padding_value=self.padding_index[embedding_type]).to(torch.long)
                gene_max_len = embedding_dict["gene"].size(1)
                snp_max_len = embedding_dict["snp"].size(1)
                mask = torch.stack(
                        [d["genotype"]["embedding"][snp_type]['mask'] for d in data])[:, :gene_max_len, :snp_max_len]
                embedding_dict['mask'] = mask
                #print(mask.sum())
                snp_type_dict[snp_type] = embedding_dict

            '''
            #genotype_dict['embedding'] = snp_type_dict
            embedding_dict = {}
            for embedding_type in ['snp', 'gene']:
                embedding_dict[embedding_type] = pad_sequence(
                    [d['genotype']['embedding'][embedding_type] for d in data], batch_first=True,
                    padding_value=self.padding_index[embedding_type]).to(torch.long)
            gene_max_len = embedding_dict["gene"].size(1)
            snp_max_len = embedding_dict["snp"].size(1)
            mask = torch.stack(
                [d["genotype"]["embedding"]['mask'] for d in data])[:, :gene_max_len, :snp_max_len]
            embedding_dict['mask'] = mask
            # print(mask.sum())
            '''
            genotype_dict['embedding'] = snp_type_dict
            #genotype_dict['gene2sys_mask'] = torch.stack([d['genotype']['gene2sys_mask'] for d in data])
        else:
            homozygous = torch.stack([d['genotype']['homozygous_a1'] for d in data])[:, :, sampled_snp_inds]
            homozygous = homozygous[:, sampled_gene_inds, :]
            heterozygous = torch.stack([d['genotype']['heterozygous'] for d in data])[:, :, sampled_snp_inds]
            heterozygous = heterozygous[:, sampled_gene_inds, :]
            genotype_dict['homozygous_a1'] = homozygous
            genotype_dict['heterozygous'] = heterozygous
            #gene2sys_mask = torch.stack([d['genotype']['gene2sys_mask'] for d in data])[:, sampled_sys_inds, :]
            #gene2sys_mask = gene2sys_mask[:, :, sampled_gene_inds]
            #genotype_dict['gene2sys_mask'] = gene2sys_mask
        result_dict['genotype'] = genotype_dict
        result_dict['covariates'] = torch.stack([d['covariates'] for d in data])
        result_dict['phenotype'] = torch.tensor([d['phenotype'] for d in data], dtype=torch.float32)
        if self.sampling_ratio != 1.:
            result_dict['nested_subtrees_forward'] = self.tree_parser.get_nested_subtree_mask(self.subtree_order, direction='forward', sys_list=sampled_sys_inds)
            result_dict['nested_subtrees_backward'] =  self.tree_parser.get_nested_subtree_mask(self.subtree_order, direction='backward', sys_list=sampled_sys_inds)
            gene2sys_mask = self.gene2sys_mask[sampled_sys_inds, :]
            gene2sys_mask = gene2sys_mask[:, sampled_gene_inds]
            sys2gene_mask = self.sys2gene_mask[:, sampled_sys_inds]
            sys2gene_mask = sys2gene_mask[sampled_gene_inds, :]

            result_dict['gene2sys_mask'] = gene2sys_mask
            result_dict['sys2gene_mask'] = sys2gene_mask
            result_dict['sys_inds'] = torch.tensor(sampled_sys_inds)
            result_dict['gene_inds'] = torch.tensor(sampled_gene_inds)
            result_dict['snp_inds'] = torch.tensor(sampled_snp_inds)

        else:
            result_dict['nested_subtrees_forward'] = self.nested_subtrees_forward
            result_dict['nested_subtrees_backward'] = self.nested_subtrees_backward
            result_dict['gene2sys_mask'] = self.gene2sys_mask
            result_dict['sys2gene_mask'] = self.sys2gene_mask
            result_dict['sys_inds'] = torch.tensor([value for key, value in self.tree_parser.sys2ind.items()])
            result_dict['gene_inds'] = torch.tensor([value for key, value in self.tree_parser.gene2ind.items()])
            result_dict['snp_inds'] = torch.tensor([value for key, value in self.tree_parser.snp2ind.items()])
        end = time.time()
        result_dict['datatime'] = torch.mean(torch.stack([d['datatime'] for d in data]))
        result_dict["time"] = torch.tensor(end - start)
        #print(genotype_dict)
        return result_dict

class CohortSampler(Sampler):

    def __init__(self, dataset, n_samples=None, phenotype_index='phenotype', z_weight=1):
        #super(DrugResponseSampler, self).__init__()
        self.indices = dataset.index
        self.num_samples = dataset.shape[0]

        phenotype_values = dataset[phenotype_index]
        #phenotype_mean = np.mean(phenotype_values)
        #phenotype_std = np.std(phenotype_values)
        #weights = np.array(z_weights*np.abs((phenotype_values-phenotype_mean)/np.std(phenotype_std)), dtype=np.int)
        self.weights = np.abs(zscore(phenotype_values)*z_weight)
        #self.dataset = result_df.reset_index()[["cellline", "drug", "response", "source", "zscore"]]
        self.weights = torch.tensor(self.weights, dtype=torch.double)

    def __iter__(self):
        count = 0
        index = [i for i in torch.multinomial(self.weights*10, self.num_samples, replacement=True)]
        while count < self.num_samples:
            #print(index[count], type(index[count]))
            #result = index[count].item()
            #print(result, type(result))
            yield index[count].item()
            count += 1

    def __len__(self):
        return self.num_samples



class DistributedCohortSampler(DistributedSampler):

    def __init__(self, dataset, num_replicas=None, rank=None, shuffle=False, seed = 0, phenotype_index='phenotype', z_weight=1):
        #super(DrugResponseSampler, self).__init__()
        super().__init__(dataset, num_replicas, rank, shuffle, seed, drop_last=False)
        self.indices = dataset.index
        self.num_samples = int(dataset.shape[0]/num_replicas)

        phenotype_values = dataset[phenotype_index].values
        a, loc, scale = skewnorm.fit(phenotype_values)
        #weights = np.array(z_weights*np.abs((phenotype_values-phenotype_mean)/np.std(phenotype_std)), dtype=np.int)
        self.weights = skewnorm(a, loc, scale*z_weight).pdf(phenotype_values)
        #self.dataset = result_df.reset_index()[["cellline", "drug", "response", "source", "zscore"]]
        self.weights = torch.tensor(self.weights, dtype=torch.double)
    def __iter__(self):
        count = 0
        index = [i for i in torch.multinomial(self.weights*10, self.num_samples, replacement=True)]
        while count < self.num_samples:
            #print(index[count], type(index[count]))
            #result = index[count].item()
            #print(result, type(result))
            yield index[count].item()
            count += 1

    def __len__(self):
        return self.num_samples
