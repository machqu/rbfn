#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
This module implements efficient radial basis function networks (RBFN's)
and the pruning method from the paper

  J. Määttä, V. Bazaliy, J. Kimari, F. Djurabekova, K. Nordlund, T. Roos. (2020).
  "Gradient-Based Training and Pruning of Radial Basis Function Networks
   with an Application in Materials Physics."

The parameter names are in line with the paper's notation.
"""


import copy
import math
import torch
import torch.nn
import numpy as np


class RBFN(torch.nn.Module):
    def __init__(self, num_inputs, num_centroids):
        """
        :param num_inputs: Number of dimensions in the input space
        :param num_centroids: Number of centroids in RBFN
        """

        super(RBFN, self).__init__()

        self.num_inputs = num_inputs
        self.num_centroids = num_centroids

        self.alpha = torch.nn.Parameter(torch.Tensor(1))
        self.log_gamma = torch.nn.Parameter(torch.Tensor(1))
        self.beta = torch.nn.Parameter(torch.Tensor(num_centroids))

        # Centroids
        self.Z = torch.nn.Parameter(torch.Tensor(num_inputs, num_centroids))

        self.initialize()

    def initialize(self, source=None):
        """
        Initializes parameters either from a uniform distributions or
        from randomly chosen centroids of a source RBFN if provided.

        :param source: Another RBFN
        """

        if source is None:
            self.alpha.data.uniform_(-2.0, 2.0)
            self.log_gamma.data.uniform_(-0.1 - math.log(self.num_inputs),
                                         0.1 - math.log(self.num_inputs))
            self.beta.data.uniform_(-2.0, 2.0)
            self.Z.data.uniform_(-2.0, 2.0)
        elif isinstance(source, RBFN):
            if self.num_inputs != source.num_inputs:
                raise RuntimeError(
                    "cannot initialize from an RBFN with different number of inputs")

            # We can simply copy scalar parameters since they don't depent on
            # the input dimensionality.
            self.alpha.data.copy_(source.alpha.data)
            self.log_gamma.data.copy_(source.log_gamma.data)

            # Randomly pick a subset of the source RBFN's centroids
            idx = np.random.choice(source.num_centroids,
                                   size=self.num_centroids,
                                   replace=False)
            for i, k in enumerate(idx):
                self.beta[i].data.copy_(source.beta.data[k])
                for j in range(self.num_inputs):
                    self.Z[j, i].data.copy_(source.Z.data[j, k])
        else:
            raise RuntimeError("invalid value for source: {}".format(source))

    def forward(self, X):
        # Sanity check
        if X.dim() != 2 or X.shape[1] != self.num_inputs:
            raise ValueError("X should be a 2d tensor with {} columns".format(
                self.num_inputs))

        # Compute the distance matrix D.
        # This vectorized version (based on expanding the square)
        # is much faster than a bunch of nested for loops.
        B = torch.mm(X, self.Z)
        D = (X**2).sum(1).unsqueeze(1).expand_as(B) \
            + (self.Z**2).sum(0).expand_as(B) \
            - 2.0 * B

        # Apply the kernel
        gamma = torch.exp(self.log_gamma)
        A = torch.exp(-gamma * D)

        # Finalize the predictions and return
        pred = self.alpha + torch.mv(A, self.beta)
        return pred


class RBFNPruningLoss:
    def __init__(self, old_rbfn):
        self.D = old_rbfn.num_inputs
        self.N = old_rbfn.num_centroids

        self.old_alpha = old_rbfn.alpha.data.detach()
        self.old_beta = old_rbfn.beta.data.detach()
        self.old_gamma = torch.exp(old_rbfn.log_gamma.data).detach()
        self.old_Z = old_rbfn.Z.data.detach()

        # Cache constant terms
        self.old_single_sum_term_val = self.single_sum_term(
            self.old_gamma, self.old_beta, self.old_Z)
        self.old_double_sum_term_val = self.double_sum_term(
            self.old_gamma, self.old_beta, self.old_Z)

    def expectation_1(self, gamma, Z):
        # Depends on the data distribution.
        # Z should have shape (num_inputs, num_centroids).
        # The return value should be a tensor of shape (num_centroids).
        # Subclasses may provide optimized implementations or use this one.
        return self.expectation_2(gamma, Z, 0, torch.zeros_like(Z)).reshape((-1,))

    def expectation_2(self,
                      gamma_1, Z_1,
                      gamma_2, Z_2
                      ):
        # Depends on the data distribution.
        # Z_1 should have shape (num_inputs, num_centroids_1) and
        # Z_2 should have shape (num_inputs, num_centroids_2).
        # The return value should be a tensor of shape (num_centroids_1, num_centroids_2).
        # The exact implementation is provided in child classes.
        raise NotImplementedError

    def single_sum_term(self, gamma, beta, Z):
        return torch.dot(beta, self.expectation_1(gamma, Z))

    def double_sum_term(self, gamma, beta, Z):
        return torch.chain_matmul(
            beta.unsqueeze(0),
            self.expectation_2(gamma, Z, gamma, Z),
            beta.unsqueeze(1)
        ).squeeze()

    def cross_sum_term(self,
                       gamma_1, beta_1, Z_1,
                       gamma_2, beta_2, Z_2
                       ):
        return torch.chain_matmul(
            beta_1.unsqueeze(0),
            self.expectation_2(gamma_1, Z_1, gamma_2, Z_2),
            beta_2.unsqueeze(1)
        ).squeeze()

    def __call__(self, new_rbfn):
        # Extract parameters from the new RBFN
        new_alpha = new_rbfn.alpha
        new_beta = new_rbfn.beta
        new_gamma = torch.exp(new_rbfn.log_gamma)
        new_Z = new_rbfn.Z

        loss = (new_alpha - self.old_alpha) ** 2
        loss += self.old_double_sum_term_val
        loss += self.double_sum_term(new_gamma, new_beta, new_Z)

        loss += 2.0 * (self.old_alpha - new_alpha) * \
            (self.old_single_sum_term_val -
             self.single_sum_term(new_gamma, new_beta, new_Z))

        loss -= 2.0 * self.cross_sum_term(self.old_gamma, self.old_beta, self.old_Z,
                                          new_gamma,      new_beta,      new_Z)

        return loss


class UnitNormalPruningLoss(RBFNPruningLoss):
    def expectation_1(self, gamma, Z):
        log_E = -gamma * torch.sum(Z**2, 0) / (1.0 + 2.0 * gamma)
        log_E -= 0.5 * Z.shape[0] * torch.log1p(2.0 * gamma)
        return torch.exp(log_E)

    def expectation_2(self, gamma_1, Z_1, gamma_2, Z_2):
        log_E = -4.0 * gamma_1 * gamma_2 * torch.mm(Z_1.t(), Z_2)
        log_E += gamma_1 * (1.0 + 2.0*gamma_2) * \
            torch.sum(Z_1**2, 0).unsqueeze(1).expand_as(log_E)
        log_E += gamma_2 * (1.0 + 2.0*gamma_1) * \
            torch.sum(Z_2**2, 0).unsqueeze(0).expand_as(log_E)
        log_E /= -(1.0 + 2.0 * (gamma_1 + gamma_2))
        log_E -= 0.5 * Z_1.shape[0] * torch.log1p(2.0 * (gamma_1 + gamma_2))
        return torch.exp(log_E)


class UniformPruningLoss(RBFNPruningLoss):
    def __init__(self, old_rbfn, a=-1, b=1, *args, **kwargs):
        """
        :param a: lower limit of the interval
        :param b: upper limit of the interval
        """

        self.a = a
        self.b = b
        if a >= b:
            raise ValueError("must have a<b")

        super(UniformPruningLoss, self).__init__(old_rbfn, *args, **kwargs)

    def expectation_1(self, gamma, Z):
        num_inputs = Z.shape[0]
        E = (torch.erf(torch.sqrt(gamma) * (self.b - Z))
             - torch.erf(torch.sqrt(gamma) * (self.a - Z)))
        E = torch.prod(E, 0)
        E = E * torch.exp(
            -num_inputs * math.log(2 * (self.b - self.a))
            - 0.5 * num_inputs * torch.log(gamma / math.pi)
        )
        return E

    def expectation_2(self, gamma_1, Z_1, gamma_2, Z_2):
        num_inputs = Z_1.shape[0]
        log_E = -2.0 * torch.mm(Z_1.t(), Z_2)
        log_E += torch.sum(Z_1**2, 0).unsqueeze(1).expand_as(log_E)
        log_E += torch.sum(Z_2**2, 0).unsqueeze(0).expand_as(log_E)
        log_E *= -gamma_1 * gamma_2 / (gamma_1 + gamma_2)
        log_E += (
            -num_inputs * math.log(2 * (self.b - self.a))
            - 0.5 * num_inputs * torch.log((gamma_1 + gamma_2) / math.pi)
        )

        const_sqrt = torch.sqrt(gamma_1 + gamma_2)
        const_a = torch.div(self.a * (gamma_1 + gamma_2), const_sqrt)
        const_b = torch.div(self.b * (gamma_1 + gamma_2), const_sqrt)
        multipliers = []
        for k in range(num_inputs):
            B = (gamma_1 * Z_1[k].unsqueeze(1).expand_as(log_E)
                 + gamma_2 * Z_2[k].unsqueeze(0).expand_as(log_E)
                 ) / const_sqrt
            B = torch.erf(const_b - B) - torch.erf(const_a - B)
            multipliers.append(B)
        Q = torch.prod(torch.stack(multipliers, 0), 0)

        E = torch.exp(log_E) * Q
        return E


class BinaryPruningLoss(RBFNPruningLoss):
    def expectation_1(self, gamma, Z):
        log_E = -Z.shape[0] * math.log(2) - gamma * torch.sum((Z+1)**2, 0)
        log_E += torch.sum(torch.log1p(torch.exp(4.0 * gamma * Z)), 0)
        return torch.exp(log_E)

    def expectation_2(self, gamma_1, Z_1, gamma_2, Z_2):
        num_inputs = Z_1.shape[0]
        num_centroids_1 = Z_1.shape[1]
        num_centroids_2 = Z_2.shape[1]

        Q_1 = torch.exp(4.0 * gamma_1 * Z_1)
        Q_2 = torch.exp(4.0 * gamma_2 * Z_2)
        log_E = torch.zeros((num_centroids_1, num_centroids_2))
        for k in range(num_inputs):
            log_E += torch.log1p(torch.mm(
                Q_1[k].unsqueeze(1),
                Q_2[k].unsqueeze(0)
            ))

        log_E -= gamma_1 * \
            torch.sum((Z_1 + 1) ** 2, 0).unsqueeze(1).expand_as(log_E)
        log_E -= gamma_2 * \
            torch.sum((Z_2 + 1) ** 2, 0).unsqueeze(0).expand_as(log_E)

        log_E -= num_inputs * math.log(2)

        return torch.exp(log_E)
