import torch
from torch import nn


class LearnedGraph(nn.Module):
    def __init__(self, input_dim, graph_prior, prior_mask, threshold, het_decoder, device):
        super(LearnedGraph, self).__init__()

        self.graph = nn.Parameter(torch.ones(
            input_dim, input_dim, requires_grad=True, device=device)/2)

        if all(i is not None for i in [graph_prior, prior_mask]):
            self.graph_prior = graph_prior.detach().clone().requires_grad_(False).to(device)
            self.prior_mask = prior_mask.detach().clone().requires_grad_(False).to(device)
            self.use_prior = True
        else:
            self.use_prior = False

        self.threshold = nn.Threshold(threshold, 0)
        self.het_decoder = het_decoder
        self.device = device

    def forward(self, iter):
        if self.use_prior:
            graph = self.prior_mask*self.graph_prior + \
                (1-self.prior_mask)*self.graph
        else:
            graph = self.graph

        if not self.het_decoder:
            graph = graph.clone()
            graph = graph * \
                (torch.ones(graph.shape[0]).to(self.device) -
                 torch.eye(graph.shape[0]).to(self.device))

        if iter is not None and iter > 50:
            graph = self.threshold(graph)

        return graph
