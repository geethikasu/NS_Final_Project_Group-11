import torch
import numpy as np

class DecagonOptimizer(object):
    def __init__(self, embeddings, latent_inters, latent_varies,
                 degrees, edge_types, edge_type2dim, placeholders,
                 margin=0.1, neg_sample_weights=1., batch_size=100):
        self.embeddings= embeddings
        self.latent_inters = latent_inters
        self.latent_varies = latent_varies
        self.edge_types = edge_types
        self.degrees = degrees
        self.edge_type2dim = edge_type2dim
        self.obj_type2n = {i: self.edge_type2dim[i,j][0][0] for i, j in self.edge_types}
        self.margin = margin
        self.neg_sample_weights = neg_sample_weights
        self.batch_size = batch_size

        self.inputs = placeholders['batch']
        self.batch_edge_type_idx = placeholders['batch_edge_type_idx']
        self.batch_row_edge_type = placeholders['batch_row_edge_type']
        self.batch_col_edge_type = placeholders['batch_col_edge_type']

        self.row_inputs = torch.squeeze(gather_cols(self.inputs, [0]))
        self.col_inputs = torch.squeeze(gather_cols(self.inputs, [1]))

        obj_type_n = [self.obj_type2n[i] for i in range(len(self.embeddings))]
        self.obj_type_lookup_start = torch.cumsum([0] + obj_type_n[:-1])
        self.obj_type_lookup_end = torch.cumsum(obj_type_n)

        labels = torch.reshape(torch.can_cast(self.row_inputs, dtype=torch.int64), [self.batch_size, 1])
        neg_samples_list = []
        for i, j in self.edge_types:
            for k in range(self.edge_types[i,j]):
                neg_samples, _, _ = torch.nn.functional.grid_sample(
                    true_classes=labels,
                    num_true=1,
                    num_sampled=self.batch_size,
                    unique=False,
                    range_max=len(self.degrees[i][k]),
                    distortion=0.75,
                    unigrams=self.degrees[i][k].tolist())
                neg_samples_list.append(neg_samples)
        self.neg_samples = torch.gather(neg_samples_list, self.batch_edge_type_idx)

        self.preds = self.batch_predict(self.row_inputs, self.col_inputs)
        self.outputs = torch.diag(self.preds)
        self.outputs = torch.reshape(self.outputs, [-1])

        self.neg_preds = self.batch_predict(self.neg_samples, self.col_inputs)
        self.neg_outputs = torch.diag(self.neg_preds)
        self.neg_outputs = torch.reshape(self.neg_outputs, [-1])

        self.predict()

        self._build()

    def batch_predict(self, row_inputs, col_inputs):
        concatenated = torch.concat(self.embeddings, 0)

        ind_start = torch.gather(self.obj_type_lookup_start, self.batch_row_edge_type)
        ind_end = torch.gather(self.obj_type_lookup_end, self.batch_row_edge_type)
        indices = torch.range(ind_start, ind_end)
        row_embeds = torch.gather(concatenated, indices)
        row_embeds = torch.gather(row_embeds, row_inputs)

        ind_start = torch.gather(self.obj_type_lookup_start, self.batch_col_edge_type)
        ind_end = torch.gather(self.obj_type_lookup_end, self.batch_col_edge_type)
        indices = torch.range(ind_start, ind_end)
        col_embeds = torch.gather(concatenated, indices)
        col_embeds = torch.gather(col_embeds, col_inputs)

        latent_inter = torch.gather(self.latent_inters, self.batch_edge_type_idx)
        latent_var = torch.gather(self.latent_varies, self.batch_edge_type_idx)

        product1 = torch.matmul(row_embeds, latent_var)
        product2 = torch.matmul(product1, latent_inter)
        product3 = torch.matmul(product2, latent_var)
        preds = torch.matmul(product3, torch.transpose(col_embeds))
        return preds

    def predict(self):
        concatenated = torch.concat(self.embeddings, 0)

        ind_start = torch.gather(self.obj_type_lookup_start, self.batch_row_edge_type)
        ind_end = torch.gather(self.obj_type_lookup_end, self.batch_row_edge_type)
        indices = torch.range(ind_start, ind_end)
        row_embeds = torch.gather(concatenated, indices)

        ind_start = torch.gather(self.obj_type_lookup_start, self.batch_col_edge_type)
        ind_end = torch.gather(self.obj_type_lookup_end, self.batch_col_edge_type)
        indices = torch.range(ind_start, ind_end)
        col_embeds = torch.gather(concatenated, indices)

        latent_inter = torch.gather(self.latent_inters, self.batch_edge_type_idx)
        latent_var = torch.gather(self.latent_varies, self.batch_edge_type_idx)

        product1 = torch.matmul(row_embeds, latent_var)
        product2 = torch.matmul(product1, latent_inter)
        product3 = torch.matmul(product2, latent_var)
        self.predictions = torch.matmul(product3, torch.transpose(col_embeds))

    def _build(self):
        self.cost = self._hinge_loss(self.outputs, self.neg_outputs)
        # self.cost = self._xent_loss(self.outputs, self.neg_outputs)
        self.optimizer = torch.optim.Adam(learning_rate=FLAGS.learning_rate)

        self.opt_op = self.optimizer.minimize(self.cost)
        self.grads_vars = self.optimizer.compute_gradients(self.cost)

    def _hinge_loss(self, aff, neg_aff):
        """Maximum-margin optimization using the hinge loss."""
        diff = torch.nn.functional.relu(torch.subtract(neg_aff, torch.Tensor.expand(aff, 0) - self.margin), name='diff')
        loss = torch.sum(diff)
        return loss

    def _xent_loss(self, aff, neg_aff):
        """Cross-entropy optimization."""
        true_xent = torch.nn.MultiLabelSoftMarginLoss(labels=torch.ones_like(aff), logits=aff)
        negative_xent = torch.nn.MultiLabelSoftMarginLoss(labels=torch.zeros_like(neg_aff), logits=neg_aff)
        loss = torch.sum(true_xent) + self.neg_sample_weights * torch.sum(negative_xent)
        return loss


def gather_cols(params, indices, name=None):
    
        # Check input
        params = torch.as_tensor(params, name="params")
        indices = torch.as_tensor(indices, name="indices")
        try:
            params.get_shape().assert_has_rank(2)
        except ValueError:
            raise ValueError('\'params\' must be 2D.')
        try:
            indices.get_shape().assert_has_rank(1)
        except ValueError:
            raise ValueError('\'params\' must be 1D.')

        # Define op
        p_shape = torch.shape(params)
        p_flat = torch.shape(params, [-1])
        i_flat = torch.shape(torch.shape(torch.range(0, p_shape[0]) * p_shape[1],
                                       [-1, 1]) + indices, [-1])
        return torch.reshape(
            torch.gather(p_flat, i_flat), [p_shape[0], -1])