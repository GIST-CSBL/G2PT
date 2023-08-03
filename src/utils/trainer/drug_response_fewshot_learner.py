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
from src.utils.trainer.loss import SpearmanLoss
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
        self.loss = nn.L1Loss()#CCCLoss()#SpearmanLoss(regularization_strength=args.l2_lambda, device=self.device)
        self.ccc_loss = CCCLoss()
        self.optimizer = optim.Adam(filter(lambda p: p.requires_grad, self.few_shot_model.parameters()),
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
        self.l1_lambda = args.l1_lambda
        self.l2_lambda = args.l2_lambda

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
                perturbed_systems, perturbed_genes = model.get_perturbed_embedding(batch['genotype'],
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
                performance = self.evaluate(self.few_shot_test_drug_response_dataloader, epoch+1, name="Validation", train_predictor=False, invert_prediction=True)
                if performance > best_performance:
                    self.best_model = copy.deepcopy(self.few_shot_model).to('cpu')
                torch.cuda.empty_cache()
                gc.collect()
                if output_path:
                    output_path_epoch = output_path + ".%d"%epoch
                    torch.save(self.few_shot_model, output_path_epoch)
            #self.lr_scheduler.step()

    def train_epoch(self, epoch):
        self.few_shot_model.train()
        self.iter_minibatches(self.few_shot_train_drug_response_dataloader, epoch, name="Batch", train_predictor=False, invert_prediction=True)


    def evaluate(self, dataloader, epoch, name="Validation", train_predictor=False, invert_prediction=True):
        trues = []
        results = []
        dataloader_with_tqdm = tqdm(dataloader)
        self.few_shot_model.eval()
        with torch.no_grad():
            for i, batch in enumerate(dataloader_with_tqdm):
                batch = move_to(batch, self.device)
                query_sys_embedding, query_gene_embedding, = self.train_drug_response_model.get_perturbed_embedding(
                    batch['genotype'],
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
                    predictor = self.train_drug_response_model.drug_response_predictor

                prediction = self.train_drug_response_model.prediction(predictor, compound_embedding, updated_sys_embedding,
                                                            updated_gene_embedding)
                if invert_prediction:
                    prediction = 1 - prediction
                trues.append((batch['response_mean'] + batch['response_residual']).detach().cpu().numpy())
                prediction = prediction.detach().cpu().numpy()
                results.append(prediction)
                dataloader_with_tqdm.set_description("%s epoch: %d" % (name, epoch))
        trues = np.concatenate(trues)
        results = np.concatenate(results)[:, 0]
        print(trues[:10])
        print(results[:10])
        r_square = metrics.r2_score(trues, results)
        spearman = spearmanr(trues, results)
        print("R_square: ", r_square)
        print("Spearman Rho: ", spearman)
        return spearman[0]

    def iter_minibatches(self, dataloader, epoch, name="", train_predictor=False, invert_prediction=True):
        mean_ccc_loss = 0.
        mean_l1_loss = 0.
        dataloader_with_tqdm = tqdm(dataloader)
        for i, batch in enumerate(dataloader_with_tqdm):
            batch = move_to(batch, self.device)
            query_sys_embedding, query_gene_embedding, = self.train_drug_response_model.get_perturbed_embedding(batch['genotype'],
                                                           self.nested_subtrees_forward, self.nested_subtrees_backward, self.system2gene_mask,
                                                               sys2cell=self.args.sys2cell,
                                                               cell2sys=self.args.cell2sys,
                                                               sys2gene=self.args.sys2gene)
            updated_sys_embedding, updated_gene_embedding = self.few_shot_model(query_sys_embedding, query_gene_embedding, self.train_system_embedding, self.train_gene_embedding)

            compound_embedding = self.train_drug_response_model.get_compound_embedding(batch['drug'], unsqueeze=True)

            if train_predictor:
                predictor = self.few_shot_model.predictor
            else:
                predictor = self.train_drug_response_model.drug_response_predictor

            prediction = self.train_drug_response_model.prediction(predictor, compound_embedding, updated_sys_embedding, updated_gene_embedding)
            if invert_prediction:
                prediction = 1 - prediction
            prediction = prediction[:, 0]
            l1_loss = self.loss((batch['response_mean'] + batch['response_residual']).to(torch.float32), prediction)
            ccc_loss = self.ccc_loss((batch['response_mean'] + batch['response_residual']).to(torch.float32), prediction)
            l1_reg = 0
            l2_reg = 0
            for param in self.few_shot_model.parameters():
                l1_reg += torch.norm(param, p=1)
                l2_reg += torch.norm(param, p=2)

            loss =  ccc_loss + self.l1_lambda * l1_reg + self.l2_lambda * l2_reg
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()
            mean_l1_loss += float(l1_loss)
            mean_ccc_loss += float(ccc_loss)

            dataloader_with_tqdm.set_description("%s Train epoch: %d, l1 loss: %.3f , ccc loss: %.3f" % ( name, epoch, mean_l1_loss / (i + 1), mean_ccc_loss / (i + 1)))
