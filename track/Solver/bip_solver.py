import itertools
from collections import defaultdict

import numpy as np
from cvxopt import glpk, matrix, spmatrix

glpk.options = {"msg_lev": "GLP_MSG_ERR"}


FROZEN_POS_EDGE = -1
FROZEN_NEG_EDGE = -2
INVALID_EDGE = -100


class _BIPSolver:
    def __init__(self, min_affinity=-np.inf, max_affinity=np.inf, create_bip=None):
        self.min_affinity = min_affinity
        self.max_affinity = max_affinity

    @staticmethod
    def _create_bip(affinity_matrix, min_affinity, max_affinity):
        n_nodes = affinity_matrix.shape[0]

        # mask for selecting pairs of nodes
        triu_mask = np.triu(np.ones_like(affinity_matrix, dtype=bool), 1)

        affinities = affinity_matrix[triu_mask]
        frozen_pos_mask = affinities >= max_affinity
        frozen_neg_mask = affinities <= min_affinity
        unfrozen_mask = np.logical_not(frozen_pos_mask | frozen_neg_mask)

        # generate objective coefficients
        objective_coefficients = affinities[unfrozen_mask]

        if len(objective_coefficients) == 0:  # nio unfrozen edges
            objective_coefficients = np.asarray([affinity_matrix[0, -1]])
            unfrozen_mask = np.zeros_like(unfrozen_mask, dtype=np.bool)
            unfrozen_mask[affinity_matrix.shape[1] - 1] = 1

        # create matrix whose rows are the indices of the three edges in a
        # constraint x_ij + x_ik - x_jk <= 1
        constraints_edges_idx = []
        if n_nodes >= 3:
            edges_idx = np.empty_like(affinities, dtype=int)
            edges_idx[frozen_pos_mask] = FROZEN_POS_EDGE
            edges_idx[frozen_neg_mask] = FROZEN_NEG_EDGE
            edges_idx[unfrozen_mask] = np.arange(len(objective_coefficients))
            nodes_to_edge_matrix = np.empty_like(affinity_matrix, dtype=int)
            nodes_to_edge_matrix.fill(INVALID_EDGE)
            nodes_to_edge_matrix[triu_mask] = edges_idx

            triplets = np.asarray(
                tuple(itertools.combinations(range(n_nodes), 3)), dtype=int
            )
            constraints_edges_idx = np.zeros_like(triplets)
            constraints_edges_idx[:, 0] = nodes_to_edge_matrix[
                (triplets[:, 0], triplets[:, 1])
            ]
            constraints_edges_idx[:, 1] = nodes_to_edge_matrix[
                (triplets[:, 0], triplets[:, 2])
            ]
            constraints_edges_idx[:, 2] = nodes_to_edge_matrix[
                (triplets[:, 1], triplets[:, 2])
            ]
            constraints_edges_idx = constraints_edges_idx[
                np.any(constraints_edges_idx >= 0, axis=1)
            ]

        if len(constraints_edges_idx) == 0:  # no constraints
            constraints_edges_idx = np.asarray([0, 0, 0], dtype=int).reshape(-1, 3)

        # add remaining constraints by permutation
        constraints_edges_idx = np.vstack(
            (
                constraints_edges_idx,
                np.roll(constraints_edges_idx, 1, axis=1),
                np.roll(constraints_edges_idx, 2, axis=1),
            )
        )

        # clean redundant constraints
        # x1 + x2 <= 2
        constraints_edges_idx = constraints_edges_idx[
            constraints_edges_idx[:, 2] != FROZEN_POS_EDGE
        ]
        # x1 - x2 <= 1
        constraints_edges_idx = constraints_edges_idx[
            np.all(constraints_edges_idx[:, 0:2] != FROZEN_NEG_EDGE, axis=1)
        ]
        if len(constraints_edges_idx) == 0:  # no constraints
            constraints_edges_idx = np.asarray([0, 0, 0], dtype=int).reshape(-1, 3)

        # generate constraint coefficients
        constraints_coefficients = np.ones_like(constraints_edges_idx)
        constraints_coefficients[:, 2] = -1

        # generate constraint upper bounds
        upper_bounds = np.ones(len(constraints_coefficients), dtype=np.float)
        upper_bounds -= np.sum(
            constraints_coefficients * (constraints_edges_idx == FROZEN_POS_EDGE),
            axis=1,
        )

        # flatten constraints data into sparse matrix format
        constraints_idx = np.repeat(np.arange(len(constraints_edges_idx)), 3)
        constraints_edges_idx = constraints_edges_idx.reshape(-1)
        constraints_coefficients = constraints_coefficients.reshape(-1)

        unfrozen_edges = constraints_edges_idx >= 0
        constraints_idx = constraints_idx[unfrozen_edges]
        constraints_edges_idx = constraints_edges_idx[unfrozen_edges]
        constraints_coefficients = constraints_coefficients[unfrozen_edges]

        return (
            objective_coefficients,
            unfrozen_mask,
            frozen_pos_mask,
            frozen_neg_mask,
            (constraints_coefficients, constraints_idx, constraints_edges_idx),
            upper_bounds,
        )

    @staticmethod
    def _solve_bip(objective_coefficients, sparse_constraints, upper_bounds):
        raise NotImplementedError

    @staticmethod
    def solution_mat_clusters(solution_mat):
        n = solution_mat.shape[0]
        labels = np.arange(1, n + 1)
        for i in range(n):
            for j in range(i + 1, n):
                if solution_mat[i, j] > 0:
                    labels[j] = labels[i]

        clusters = defaultdict(list)
        for i, label in enumerate(labels):
            clusters[label].append(i)
        return list(clusters.values())

    def solve(self, affinity_matrix, rtn_matrix=False):
        n_nodes = affinity_matrix.shape[0]
        if n_nodes <= 1:
            solution_x, sol_matrix = (
                np.asarray([], dtype=int),
                np.asarray([0] * n_nodes, dtype=int),
            )
            sol_matrix = sol_matrix[:, None]
        elif n_nodes == 2:
            solution_matrix = np.zeros_like(affinity_matrix, dtype=int)
            solution_matrix[0, 1] = affinity_matrix[0, 1] > 0
            solution_matrix += solution_matrix.T
            solution_x = (
                [solution_matrix[0, 1]]
                if self.min_affinity < affinity_matrix[0, 1] < self.max_affinity
                else []
            )
            solution_x, sol_matrix = np.asarray(solution_x), solution_matrix
        else:
            # create BIP problem
            (
                objective_coefficients,
                unfrozen_mask,
                frozen_pos_mask,
                frozen_neg_mask,
                sparse_constraints,
                upper_bounds,
            ) = self._create_bip(affinity_matrix, self.min_affinity, self.max_affinity)
            #print(sparse_constraints)
            # solve
            solution_x = self._solve_bip(
                objective_coefficients, sparse_constraints, upper_bounds
            )

            # solution to matrix
            all_sols = np.zeros_like(unfrozen_mask, dtype=int)
            all_sols[unfrozen_mask] = np.array(solution_x, dtype=int).reshape(-1)
            all_sols[frozen_neg_mask] = 0
            all_sols[frozen_pos_mask] = 1
            sol_matrix = np.zeros_like(affinity_matrix, dtype=int)
            sol_matrix[
                np.triu(np.ones([n_nodes, n_nodes], dtype=int), 1) > 0
            ] = all_sols
            sol_matrix += sol_matrix.T

        clusters = self.solution_mat_clusters(sol_matrix)
        if not rtn_matrix:
            return clusters
        return clusters, sol_matrix


class GLPKSolver(_BIPSolver):
    def __init__(self, min_affinity=-np.inf, max_affinity=np.inf):
        super(GLPKSolver, self).__init__(min_affinity, max_affinity)

    @staticmethod
    def _solve_bip(objective_coefficients, sparse_constraints, upper_bounds):
        c = matrix(-objective_coefficients)  # max -> min
        G = spmatrix(
            *sparse_constraints, size=(len(upper_bounds), len(objective_coefficients))
        )  # G * x <= h
        h = matrix(upper_bounds)

        status, solution = glpk.ilp(c, G, h, B=set(range(len(c))))

        assert solution is not None, "Solver error: {}".format(status)

        return np.asarray(solution, int).reshape(-1)