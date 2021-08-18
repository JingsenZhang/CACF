import torch
import torch.nn as nn
from torch.nn.init import xavier_normal_, constant_
import torch.nn.functional as F

from model.abstract_recommender import ContextRecommender
from model.layers import BaseFactorizationMachine, MLPLayers, AttLayer


class CACF(ContextRecommender):
    def __init__(self, config, dataset):
        super(CACF, self).__init__(config, dataset)

        self.mlp_hidden_size = config['mlp_hidden_size']
        self.dropout_prob = config['dropout_prob']
        self.loss_weight1 = config['loss_weight1']
        self.loss_weight2 = config['loss_weight2']
        self.t = config['t']
        self.attention_size = config['attention_size']
        self.distance = config['distance']
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        self.fm = BaseFactorizationMachine(reduce_sum=True)
        size_list = [self.embedding_size * self.num_feature_field] + self.mlp_hidden_size
        self.mlp_layers = MLPLayers(size_list, self.dropout_prob)
        self.deep_predict_layer = nn.Linear(self.mlp_hidden_size[-1], 1)
        self.attlayer = AttLayer(self.embedding_size, self.attention_size)
        self.sigmoid = nn.Sigmoid()
        self.loss = nn.BCELoss()
        self.softmax = nn.Softmax(dim=1)
        self.kl = torch.nn.KLDivLoss(reduction='batchmean')
        self.apply(self._init_weights)

    def _init_weights(self, module):
        if isinstance(module, nn.Embedding):
            xavier_normal_(module.weight.data)
        elif isinstance(module, nn.Linear):
            xavier_normal_(module.weight.data)
            if module.bias is not None:
                constant_(module.bias.data, 0)

    def forward(self, interaction):
        feature_emb = self.concat_embed_input_fields(interaction)
        attention = self.calculate_alpha(feature_emb)
        first_order_output = self.first_order_linear(interaction, -1)
        score = self.predict_layer(feature_emb, attention, first_order_output)
        return score, feature_emb, attention

    def predict_layer(self, feature_emb, attention, first_order_output):
        feature_atten_emb = torch.mul(feature_emb, attention.unsqueeze(-1))
        batch_size = feature_emb.shape[0]
        y_fm = first_order_output + self.fm(feature_atten_emb)
        y_deep = self.deep_predict_layer(self.mlp_layers(feature_atten_emb.view(batch_size, -1)))
        y = self.sigmoid(y_fm + y_deep).squeeze()
        return y

    def calculate_loss(self, interaction):
        label = interaction[self.LABEL]
        score, feature_emb, attention = self.forward(interaction)
        main_loss = self.loss(score, label)
        beta = self.calculate_beta(score, feature_emb, attention, interaction).to(device=self.device)
        if self.distance == 1:
            distance_loss = self.kl(attention.log(), beta)
        elif self.distance == 2:
            distance_loss = F.pairwise_distance(attention, beta, p=2).mean()
        elif self.distance == 3:
            distance_loss = -torch.cosine_similarity(attention.view(-1), beta.view(-1), dim=0)
        elif self.distance == 4:
            distance_loss = -torch.mul(attention, beta).sum(dim=-1).mean()
        elif self.distance == 5:
            distance_loss = self.pearson(attention, beta)
        else:
            distance_loss = self.kl(attention.log(), beta)
        l1norm_loss = torch.norm(attention, p=1, dim=1, keepdim=False).sum(dim=0)
        loss = main_loss + distance_loss * self.loss_weight1 + l1norm_loss * self.loss_weight2
        return loss

    def calculate_alpha(self, feature_emb):
        alpha = self.attlayer(feature_emb)
        return alpha

    def calculate_beta(self, score_true, feature_emb, attention, interaction):
        batch_size = feature_emb.shape[0]
        delta = torch.Tensor().to(device=self.device)
        for i in range(self.num_feature_field):
            index = torch.full((batch_size, 1, self.embedding_size), fill_value=i, dtype=torch.int64).to(device=self.device)
            value = torch.zeros(batch_size, 1, self.embedding_size).to(device=self.device)
            emb_change = feature_emb.scatter(1, index, value)
            first_order_output = self.first_order_linear(interaction, i)
            score_change = self.predict_layer(emb_change, attention, first_order_output)
            difference = abs(score_true - score_change).unsqueeze(-1)
            delta = torch.cat((delta, difference), dim=1)
        beta = self.softmax(delta / self.t)
        return beta

    def predict(self, interaction):
        return self.forward(interaction)[0]

    def pearson(self, x, y):
        vx = x - torch.mean(x)
        vy = y - torch.mean(y)
        cost = torch.sum(vx * vy) / (torch.sqrt(torch.sum(vx ** 2)) * torch.sqrt(torch.sum(vy ** 2)))
        return cost
