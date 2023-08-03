import gc
import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import  StepLR
from sklearn import metrics
from scipy.stats import spearmanr, pearsonr
from tqdm import tqdm
import numpy as np
import copy
from transformers import get_linear_schedule_with_warmup
from src.utils.trainer import CCCLoss
from src.utils.data import move_to
from src.utils.trainer.loss import spearman
import copy



class DrugResponseFewShotLearner(object):

    def __init__(self, train_drug_response_model, few_shot_model, train_drug_response_dataloader,
                 few_shot_train_drug_response_dataloader, few_shot_test_drug_response_dataloader, device, args):
        self.device = device
        self.train_drug_response_model = train_drug_response_model.to(self.device)
        for param in self.train_drug_response_model.parameters():
            param.requires_grad = False

        self.few_shot_model = few_shot_model.to(self.device)
        '''
        for name, param in self.drug_response_model.named_parameters():
            if "compound_encoder" in name:
                param.requires_grad = False
                print(name, param.requires_grad)
        '''
        self.train_drug_response_dataloader = train_drug_response_dataloader
        self.few_shot_train_drug_response_dataloader = few_shot_train_drug_response_dataloader
        self.few_shot_test_drug_response_dataloader = few_shot_test_drug_response_dataloader
        self.ccc_loss = CCCLoss()
        self.optimizer = optim.SGD(filter(lambda p: p.requires_grad, self.few_shot_model.parameters()),
                                     lr=args.lr, weight_decay=0)
        self.nested_subtrees_forward = self.train_drug_response_dataloader.dataset.tree_parser.get_nested_subtree_mask(
            args.subtree_order, direction='forward')
        self.nested_subtrees_forward = move_to(self.nested_subtrees_forward, device)
        self.nested_subtrees_backward = self.train_drug_response_dataloader.dataset.tree_parser.get_nested_subtree_mask(
            args.subtree_order, direction='backward')
        self.nested_subtrees_backward = move_to(self.nested_subtrees_backward, device)
        self.system2gene_mask = move_to(torch.tensor(self.train_drug_response_dataloader.dataset.tree_parser.sys2gene_mask, dtype=torch.bool), device)
        self.args = args
        train_system_embedding, train_gene_embedding = self.get_train_embedding(self.train_drug_response_model, self.train_drug_response_dataloader)
        self.train_system_embedding = train_system_embedding
        self.train_gene_embedding = train_gene_embedding

    def get_train_embedding(self, model, dataloader):
        dataloader_with_tqdm = tqdm(dataloader)
        model.to(self.device)
        model.eval()
        perturbed_systems_total = []
        perturbed_genes_total = []
        print("Get Training Embedding...")
        with torch.no_grad():
            for i, batch in enumerate(dataloader_with_tqdm):
                batch = move_to(batch, self.device)
                perturbed_systems, perturbed_genes = model.get_perturbed_embedding(batch['genotype'], batch['drug'],
                                                                                   self.nested_subtrees_forward,
                                                                                   self.nested_subtrees_backward,
                                                                                   self.system2gene_mask,
                                                                                   sys2cell=self.args.sys2cell,
                                                                                   cell2sys=self.args.cell2sys,
                                                                                   sys2gene=self.args.sys2gene)
                perturbed_systems_total.append(perturbed_systems)
                perturbed_genes_total.append(perturbed_genes)
        perturbed_systems_total = torch.cat(perturbed_systems_total, dim=0)
        perturbed_genes_total = torch.cat(perturbed_genes_total, dim=0)
        return perturbed_systems_total, perturbed_genes_total

    def train_few_shot(self, epochs, output_path):
        self.best_model = self.few_shot_model
        best_performance = 0
        for epoch in range(epochs):
            self.train_epoch(epoch + 1)
            gc.collect()
            torch.cuda.empty_cache()
            if (epoch % self.args.val_step)==0 & (epoch != 0):
                performance = self.evaluate(self.few_shot_test_drug_response_dataloader, epoch+1, name="Validation")
                if performance > best_performance:
                    self.best_model = copy.deepcopy(self.few_shot_model).to('cpu')
                torch.cuda.empty_cache()
                gc.collect()
                if output_path:
                    output_path_epoch = output_path + ".%d"%epoch
                    torch.save(self.few_shot_model, output_path_epoch)
            #self.lr_scheduler.step()


    def evaluate(self, dataloader, train_predictor, epoch, name="Validation"):
        trues = []
        results = []
        dataloader_with_tqdm = tqdm(dataloader)

        with torch.no_grad():
            for i, batch in enumerate(dataloader_with_tqdm):
                batch = move_to(batch, self.device)
                query_sys_embedding, query_gene_embedding, = self.train_drug_response_model.get_perturbed_embedding(
                    batch['genotype'], batch['drug'],
                    self.nested_subtrees_forward, self.nested_subtrees_backward, self.system2gene_mask,
                    sys2cell=self.args.sys2cell,
                    cell2sys=self.args.cell2sys,
                    sys2gene=self.args.sys2gene)
                updated_sys_embedding, updated_gene_embedding = self.few_shot_model(query_sys_embedding,
                                                                                    query_gene_embedding,
                                                                                    self.train_system_embedding,
                                                                                    self.train_gene_embedding)

                compound_embedding = self.train_drug_response_model.get_compound_embedding(batch['drug'],
                                                                                           unsqueeze=True)

                if train_predictor:
                    predictor = self.few_shot_model.predictor
                else:
                    predictor = self.train_drug_response_model.predictor

                prediction = self.train_drug_response_model(predictor, compound_embedding, updated_sys_embedding,
                                                            updated_gene_embedding)
                trues.append(batch['response'].detach().cpu().numpy())
                prediction = prediction.detach().cpu().numpy()
                results.append(prediction)
                dataloader_with_tqdm.set_description("%s epoch: %d" % (name, epoch))
        trues = np.concatenate(trues)
        results = np.concatenate(results)[:, 0]
        r_square = metrics.r2_score(trues, results)
        spearman = spearmanr(trues, results)
        print("R_square: ", r_square)
        print("Spearman Rho: ", spearman)
        return r_square[0]

    def iter_minibatches(self, dataloader, epoch, name="", train_predictor=False):
        mean_spearman_loss = 0.
        dataloader_with_tqdm = tqdm(dataloader)
        for i, batch in enumerate(dataloader_with_tqdm):
            batch = move_to(batch, self.device)
            query_sys_embedding, query_gene_embedding, = self.train_drug_response_model.get_perturbed_embedding(batch['genotype'], batch['drug'],
                                                           self.nested_subtrees_forward, self.nested_subtrees_backward, self.system2gene_mask,
                                                               sys2cell=self.args.sys2cell,
                                                               cell2sys=self.args.cell2sys,
                                                               sys2gene=self.args.sys2gene)
            updated_sys_embedding, updated_gene_embedding = self.few_shot_model(query_sys_embedding, query_gene_embedding, self.train_system_embedding, self.train_gene_embedding)

            compound_embedding = self.train_drug_response_model.get_compound_embedding(batch['drug'], unsqueeze=True)

            if train_predictor:
                predictor = self.few_shot_model.predictor
            else:
                predictor = self.train_drug_response_model.predictor

            prediction = self.train_drug_response_model(predictor, compound_embedding, updated_sys_embedding, updated_gene_embedding)
            spearman_loss = spearman((batch['response']).to(torch.float32), prediction[:, 0], regularization_strength=self.args.l2_lambda)
            loss = spearman_loss
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()
            mean_spearman_loss += float(spearman_loss)

            dataloader_with_tqdm.set_description("%s Train epoch: %d, loss %.3f" % ( name, epoch, mean_spearman_loss / (i + 1)))