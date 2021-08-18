import torch.nn as nn
from torch.nn.init import xavier_normal_

from model.abstract_recommender import ContextRecommender
from model.layers import BaseFactorizationMachine


class FM(ContextRecommender):
    """Factorization Machine considers the second-order interaction with features to predict the final score.

    """

    def __init__(self, config, dataset):

        super(FM, self).__init__(config, dataset)

        # define layers and loss
        self.fm = BaseFactorizationMachine(reduce_sum=True)
        self.sigmoid = nn.Sigmoid()
        self.loss = nn.BCELoss()

        # parameters initialization
        self.apply(self._init_weights)

    def _init_weights(self, module):
        if isinstance(module, nn.Embedding):
            xavier_normal_(module.weight.data)

    def forward(self, interaction):
        fm_all_embeddings = self.concat_embed_input_fields(interaction)  # [batch_size, num_field, embed_dim]
        y = self.sigmoid(self.first_order_linear(interaction, -1) + self.fm(fm_all_embeddings))
        return y.squeeze()

    def calculate_loss(self, interaction):
        label = interaction[self.LABEL]

        output = self.forward(interaction)
        return self.loss(output, label)

    def predict(self, interaction):
        return self.forward(interaction)
