# -*- coding: utf-8 -*-
"""
XTreme.py  (single-file, NumPy-only)

Self-contained reference implementation of the main algorithms from the paper:

  "eXtreme Top-k Contextual Bandits with IGW" (TOPkExtreme.pdf)

Included:
- Algorithm 1: IGW distribution + Top-k selection (greedy (k-r) + sequential IGW for r slots)
- Algorithm 2: Beamsearch over a hierarchy to produce effective arms A_x = S_x ∪ I_x
- Algorithm 3: eXtreme Top-k Contextual Bandits with IGW (epochs + regression oracle on effective arms)
- Adapter XtremeAlg3OnPaths for non-contextual harnesses expecting:
      done, top_m = algo.select_and_update(oracle)

Reward-bounding option for eXtreme only:
- Optional sigmoid squashing of observed rewards inside Algorithm 3:
      r' = sigmoid(alpha * (r - beta))
  This is controlled by squash_rewards/sigmoid_alpha/sigmoid_beta.

Dependencies: numpy
"""

from __future__ import annotations

from dataclasses import dataclass, replace
from typing import Callable, Dict, Hashable, List, Optional, Tuple, Union

import math
import numpy as np


# ---------------------------------------------------------------------
# Types
# ---------------------------------------------------------------------

NodeId = Hashable
Context = np.ndarray
Reward = float


# ---------------------------------------------------------------------
# Ridge regression oracle (per effective arm / node)
# ---------------------------------------------------------------------

class Ridge:
    """Simple ridge regression with sufficient statistics (V, b)."""

    def __init__(self, d: int, lam: float = 1.0):
        self.d = int(d)
        self.lam = float(lam)
        self.V = self.lam * np.eye(self.d)
        self.b = np.zeros(self.d)

    def update(self, x: Context, y: float) -> None:
        x = np.asarray(x, dtype=float).reshape(-1)
        if x.size != self.d:
            raise ValueError(f"Ridge.update: expected x dim {self.d}, got {x.size}")
        self.V += np.outer(x, x)
        self.b += float(y) * x

    def theta(self) -> np.ndarray:
        return np.linalg.solve(self.V, self.b)


# ---------------------------------------------------------------------
# Hierarchy representation required by Beamsearch (Algorithm 2)
# ---------------------------------------------------------------------

@dataclass(frozen=True)
class HierarchyNode:
    """
    A node in the hierarchy tree T.

    - If arm_id is not None -> leaf node representing a singleton arm.
    - children contains child node IDs (internal nodes).
    - leaf_arm_ids caches all leaf arms under this node (filled by finalize_leaf_sets()).
    """
    node_id: NodeId
    level: int                     # root at level 0, leaves at level H
    children: Tuple[NodeId, ...] = ()
    leaf_arm_ids: Tuple[int, ...] = ()
    arm_id: Optional[int] = None   # singleton arm id if leaf


class ArmHierarchy:
    """Hierarchy T used by Algorithm 2/3."""

    def __init__(self, nodes: Dict[NodeId, HierarchyNode], root_id: NodeId, depth: int):
        self.nodes: Dict[NodeId, HierarchyNode] = dict(nodes)
        self.root_id: NodeId = root_id
        self.depth: int = int(depth)  # H (leaves at level H)

        if self.root_id not in self.nodes:
            raise ValueError("ArmHierarchy: root_id not present in nodes")
        if self.nodes[self.root_id].level != 0:
            raise ValueError("ArmHierarchy: root node must have level 0")

        self.arm_to_leaf_node: Dict[int, NodeId] = {}
        for nid, n in self.nodes.items():
            if n.arm_id is not None:
                self.arm_to_leaf_node[int(n.arm_id)] = nid

    def is_leaf(self, node_id: NodeId) -> bool:
        n = self.nodes[node_id]
        return n.arm_id is not None or len(n.children) == 0

    def sample_leaf_arm(self, node_id: NodeId, rng: np.random.Generator) -> int:
        """Uniformly sample a singleton arm from the subtree of node_id."""
        n = self.nodes[node_id]
        if n.arm_id is not None:
            return int(n.arm_id)
        if not n.leaf_arm_ids:
            raise ValueError("ArmHierarchy.sample_leaf_arm: leaf_arm_ids is empty. Call finalize_leaf_sets().")
        idx = int(rng.integers(0, len(n.leaf_arm_ids)))
        return int(n.leaf_arm_ids[idx])

    def finalize_leaf_sets(self) -> "ArmHierarchy":
        """
        Return a new ArmHierarchy with leaf_arm_ids populated for all nodes,
        computed bottom-up.
        """
        nodes = dict(self.nodes)

        # Leaves: leaf_arm_ids = (arm_id,)
        for nid, n in nodes.items():
            if n.arm_id is not None:
                nodes[nid] = replace(n, leaf_arm_ids=(int(n.arm_id),))

        # Bottom-up fill for internals
        by_level_desc = sorted(nodes.values(), key=lambda x: x.level, reverse=True)
        for n in by_level_desc:
            if n.arm_id is not None:
                continue
            if not n.children:
                continue
            leafs: List[int] = []
            for c in n.children:
                leafs.extend(list(nodes[c].leaf_arm_ids))
            nodes[n.node_id] = replace(n, leaf_arm_ids=tuple(leafs))

        return ArmHierarchy(nodes=nodes, root_id=self.root_id, depth=self.depth)

    @staticmethod
    def build_balanced_kary(num_arms: int, branching: int = 2) -> "ArmHierarchy":
        """
        Convenience builder: balanced k-ary tree whose leaves are singleton arms 0..num_arms-1.
        """
        if num_arms <= 0:
            raise ValueError("build_balanced_kary: num_arms must be positive")
        if branching < 2:
            raise ValueError("build_balanced_kary: branching must be >= 2")

        tmp_nodes: Dict[NodeId, HierarchyNode] = {}

        leaf_ids: List[NodeId] = [("L", i) for i in range(num_arms)]
        for i, nid in enumerate(leaf_ids):
            tmp_nodes[nid] = HierarchyNode(node_id=nid, level=0, arm_id=i)

        current = leaf_ids
        tmp_level = 0
        while len(current) > 1:
            tmp_level += 1
            nxt: List[NodeId] = []
            for j in range(0, len(current), branching):
                chunk = current[j:j + branching]
                pid = ("I", tmp_level, j // branching)
                tmp_nodes[pid] = HierarchyNode(node_id=pid, level=tmp_level, children=tuple(chunk))
                nxt.append(pid)
            current = nxt

        root_id = current[0]
        depth = tmp_level  # root at tmp_level, leaves at 0

        # invert levels so root level 0, leaves level H=depth
        nodes: Dict[NodeId, HierarchyNode] = {}
        for nid, n in tmp_nodes.items():
            nodes[nid] = replace(n, level=depth - n.level)

        return ArmHierarchy(nodes=nodes, root_id=root_id, depth=depth).finalize_leaf_sets()


# ---------------------------------------------------------------------
# Algorithm 1: IGW distribution and Top-k selection
# ---------------------------------------------------------------------

def igw_distribution(scores: np.ndarray, gamma: float) -> np.ndarray:
    """
    IGW(A; yhat) from the paper.

    For A0 = |A| and best arm a*:
      for a != a*:   p(a) = 1 / (A0 + gamma * (yhat(a*) - yhat(a)))
      p(a*) = 1 - sum_{a != a*} p(a)
    """
    scores = np.asarray(scores, dtype=float).reshape(-1)
    A0 = int(scores.size)
    if A0 <= 0:
        raise ValueError("igw_distribution: empty scores")
    if A0 == 1:
        return np.array([1.0], dtype=float)

    best = int(np.argmax(scores))
    gaps = scores[best] - scores  # >= 0 (for non-best)

    denom = A0 + gamma * np.maximum(gaps, 0.0)
    p = np.zeros(A0, dtype=float)

    for i in range(A0):
        if i == best:
            continue
        p[i] = 1.0 / float(denom[i])

    p[best] = 1.0 - float(p.sum())

    # Numerical safety
    if p[best] < 0.0:
        p = np.maximum(p, 0.0)
        s = float(p.sum())
        p = p / s if s > 0 else np.ones(A0, dtype=float) / A0

    return p


def topk_select_greedy_plus_igw(
    scores: np.ndarray,
    *,
    k: int,
    r: int,
    rng: np.random.Generator,
    gamma_fn: Callable[[int], float],
) -> List[int]:
    """
    Top-k selection used in Algorithm 3:

    1) Choose top-(k-r) arms greedily.
    2) For c = 1..r:
         compute IGW distribution over remaining arms
         sample one arm, add it, recompute next time.
    """
    scores = np.asarray(scores, dtype=float).reshape(-1)
    Z = int(scores.size)
    if Z <= 0:
        raise ValueError("topk_select: empty scores")
    k = int(min(k, Z))
    r = int(min(r, k))
    greedy_cnt = k - r

    order = np.argsort(-scores)
    chosen: List[int] = [int(z) for z in order[:greedy_cnt]]
    chosen_set = set(chosen)

    for _ in range(r):
        remaining = [int(z) for z in range(Z) if z not in chosen_set]
        if not remaining:
            break
        rem_scores = scores[np.array(remaining, dtype=int)]
        gamma = float(gamma_fn(len(remaining)))
        p = igw_distribution(rem_scores, gamma=gamma)
        j = int(rng.choice(len(remaining), p=p))
        z = remaining[j]
        chosen.append(z)
        chosen_set.add(z)

    return chosen


# ---------------------------------------------------------------------
# Algorithm 2: Beamsearch
# ---------------------------------------------------------------------

def beam_search(
    hierarchy: ArmHierarchy,
    x: Context,
    *,
    routing_score: Callable[[NodeId, Context], float],
    beam_size: int,
) -> List[NodeId]:
    """
    Algorithm 2 Beamsearch.

    Returns ordered effective arms A_x = S_x ∪ I_x (as a list of node IDs).
    """
    x = np.asarray(x, dtype=float).reshape(-1)
    b = int(beam_size)
    if b <= 0:
        raise ValueError("beam_search: beam_size must be positive")

    codes: List[NodeId] = [hierarchy.root_id]
    I_x: List[NodeId] = []

    # for h = 1..H-1  (hierarchy.depth == H)
    for _h in range(1, hierarchy.depth):
        labels: List[NodeId] = []
        for nid in codes:
            labels.extend(list(hierarchy.nodes[nid].children))
        if not labels:
            break

        scored = [(float(routing_score(nid, x)), nid) for nid in labels]
        scored.sort(key=lambda t: (t[0], str(t[1])), reverse=True)

        top = [nid for _, nid in scored[:b]]
        for _, nid in scored[b:]:
            I_x.append(nid)
        codes = top

    # S_x is union of children of nodes in codes
    S_x: List[NodeId] = []
    for nid in codes:
        S_x.extend(list(hierarchy.nodes[nid].children))

    return S_x + I_x


# ---------------------------------------------------------------------
# Algorithm 3: eXtreme Top-k Contextual Bandits with IGW
# ---------------------------------------------------------------------

class XtremeAlg3IGW:
    """
    Algorithm 3: eXtreme Top-k Contextual Bandits with IGW.
    """

    def __init__(
        self,
        *,
        hierarchy: ArmHierarchy,
        routing_score: Callable[[NodeId, Context], float],
        context_dim: int,
        k: int,
        r: int = 1,
        beam_size: int = 8,
        lam: float = 1.0,
        # gamma schedule: in paper experiments gamma_l = sqrt(C * N_{l-1} * |A0|)
        gamma_C: float = 1.0,
        gamma_schedule: Optional[Callable[[int, int, int], float]] = None,
        # epoch schedule: n_l (default doubling 2^{l-1})
        epoch_len_fn: Optional[Callable[[int], int]] = None,
        # partial feedback (optional simulation): each played arm observed w.p. feedback_prob
        feedback_prob: float = 1.0,
        # also train the leaf model of the played singleton arm
        update_leaf_models: bool = True,

        # --- eXtreme-only reward squashing (sigmoid) ---
        squash_rewards: bool = False,
        sigmoid_alpha: float = 1.0,
        sigmoid_beta: float = 0.0,
        sigmoid_clip: float = 35.0,

        seed: Optional[int] = None,
    ):
        self.T = hierarchy
        self.g = routing_score
        self.d = int(context_dim)

        self.k = int(k)
        self.r = int(r)
        if not (1 <= self.r <= self.k):
            raise ValueError("XtremeAlg3IGW: require 1 <= r <= k")

        self.beam_size = int(beam_size)
        if self.beam_size <= 0:
            raise ValueError("XtremeAlg3IGW: beam_size must be positive")

        self.lam = float(lam)
        self.gamma_C = float(gamma_C)

        self.epoch_len_fn = epoch_len_fn or (lambda l: 2 ** (l - 1))
        self.gamma_schedule = gamma_schedule  # overrides default if provided

        self.feedback_prob = float(feedback_prob)
        if not (0.0 < self.feedback_prob <= 1.0):
            raise ValueError("XtremeAlg3IGW: feedback_prob must be in (0,1]")

        self.update_leaf_models = bool(update_leaf_models)
        self.rng = np.random.default_rng(seed)

        # --- store squashing params (fixes your AttributeError) ---
        self.squash_rewards = bool(squash_rewards)
        self.sigmoid_alpha = float(sigmoid_alpha)
        self.sigmoid_beta = float(sigmoid_beta)
        self.sigmoid_clip = float(sigmoid_clip)

        # Online models and epoch-frozen parameters
        self._models: Dict[NodeId, Ridge] = {}
        self._theta_epoch: Dict[NodeId, np.ndarray] = {}

        # Epoch bookkeeping
        self.epoch: int = 1
        self.N_prev: int = 0                    # N_{l-1}
        self._epoch_remaining: int = int(self.epoch_len_fn(self.epoch))
        if self._epoch_remaining <= 0:
            raise ValueError("epoch_len_fn must return positive lengths")

        # Optional progress fields (expected by your plotting harness)
        self.t: int = 0
        self.total_comparisons: int = 0
        self.best_G_history: List[float] = []
        self.min_lcb_history: List[float] = []

    def _get_model(self, node_id: NodeId) -> Ridge:
        m = self._models.get(node_id)
        if m is None:
            m = Ridge(d=self.d, lam=self.lam)
            self._models[node_id] = m
        return m

    def _refresh_epoch_oracle(self) -> None:
        """Fit/freeze regression oracle at epoch start."""
        self._theta_epoch = {nid: model.theta() for nid, model in self._models.items()}

    def _predict_node(self, x: Context, node_id: NodeId) -> float:
        theta = self._theta_epoch.get(node_id)
        if theta is None:
            return 0.0
        return float(np.dot(theta, x))

    def _gamma_l(self, A0: int) -> float:
        """Default gamma_l: sqrt(C * N_{l-1} * A0)."""
        if self.gamma_schedule is not None:
            return float(self.gamma_schedule(self.epoch, self.N_prev, A0))
        return float(math.sqrt(self.gamma_C * float(self.N_prev) * float(max(1, A0))))

    def _squash_reward(self, r: float) -> float:
        """Optional sigmoid mapping to (0,1), used only if squash_rewards=True."""
        if not self.squash_rewards:
            return float(r)
        z = self.sigmoid_alpha * (float(r) - self.sigmoid_beta)
        z = float(np.clip(z, -self.sigmoid_clip, self.sigmoid_clip))
        return float(1.0 / (1.0 + np.exp(-z)))

    def effective_arms(self, x: Context) -> List[NodeId]:
        """Return A_x via Algorithm 2 beam search."""
        x = np.asarray(x, dtype=float).reshape(-1)
        if x.size != self.d:
            raise ValueError(f"effective_arms: expected context_dim {self.d}, got {x.size}")
        return beam_search(self.T, x, routing_score=self.g, beam_size=self.beam_size)

    def select_and_update(
        self,
        x: Context,
        oracle: Callable[..., Union[Reward, None]],
    ) -> Tuple[bool, List[int]]:
        """
        One round of Algorithm 3.

        Returns (done, played_singleton_arms).
        """
        x = np.asarray(x, dtype=float).reshape(-1)
        if x.size != self.d:
            raise ValueError(f"select_and_update: expected context_dim {self.d}, got {x.size}")

        # Epoch boundary
        if self._epoch_remaining <= 0:
            # advance epoch: update N_{l-1} by the length of the previous epoch
            self.N_prev += int(self.epoch_len_fn(self.epoch))
            self.epoch += 1
            self._epoch_remaining = int(self.epoch_len_fn(self.epoch))
            if self._epoch_remaining <= 0:
                raise ValueError("epoch_len_fn must return positive lengths")

        # Refresh oracle at start of each epoch
        if self._epoch_remaining == int(self.epoch_len_fn(self.epoch)):
            self._refresh_epoch_oracle()

        # (7) compute effective arms A_x
        A_x = self.effective_arms(x)
        if not A_x:
            raise RuntimeError("effective_arms returned empty list")

        # predicted reward for each effective arm
        yhat = np.array([self._predict_node(x, nid) for nid in A_x], dtype=float)

        chosen_z = topk_select_greedy_plus_igw(
            yhat,
            k=self.k,
            r=self.r,
            rng=self.rng,
            gamma_fn=lambda A0: self._gamma_l(A0),
        )

        # Substitute effective arms by singleton arms if needed
        played: List[int] = []
        z_to_arm: Dict[int, int] = {}
        for z in chosen_z:
            nid = A_x[z]
            if self.T.is_leaf(nid):
                n = self.T.nodes[nid]
                arm = int(n.arm_id) if n.arm_id is not None else self.T.sample_leaf_arm(nid, self.rng)
            else:
                arm = self.T.sample_leaf_arm(nid, self.rng)
            played.append(arm)
            z_to_arm[int(z)] = int(arm)

        # De-duplicate while preserving order (set-like semantics)
        seen = set()
        played_unique: List[int] = []
        for a in played:
            if a not in seen:
                seen.add(a)
                played_unique.append(a)
        played = played_unique

        # Play and observe rewards (possibly partial)
        observed: Dict[int, float] = {}
        for a in played:
            if self.feedback_prob < 1.0 and self.rng.random() > self.feedback_prob:
                continue

            try:
                r = oracle(x, a)
            except TypeError:
                r = oracle(a)

            if r is None:
                continue

            r2 = self._squash_reward(float(r))
            observed[a] = r2
            self.total_comparisons += 1

        # Map rewards back to effective arms (update only observed)
        for z, a in z_to_arm.items():
            if a not in observed:
                continue
            r2 = observed[a]
            eff_node = A_x[z]
            self._get_model(eff_node).update(x, r2)

            if self.update_leaf_models:
                leaf_node = self.T.arm_to_leaf_node.get(a)
                if leaf_node is not None:
                    self._get_model(leaf_node).update(x, r2)

        # bookkeeping
        self.t += 1
        self._epoch_remaining -= 1

        self.best_G_history.append(float(np.max(yhat)))
        self.min_lcb_history.append(float(np.min(yhat)))

        return False, played


# ---------------------------------------------------------------------
# Adapter: run Algorithm 3 in a non-contextual harness (paths only)
# ---------------------------------------------------------------------

class XtremeAlg3OnPaths:
    """
    Wrapper to run Algorithm 3 in a harness that does NOT provide contexts and expects:

        done, top_m = algo.select_and_update(oracle)

    We use constant context x=[1.0] and a balanced hierarchy over arm IDs 0..num_paths-1.
    """

    def __init__(
        self,
        *,
        num_paths: int,
        m: int,
        k: int = 1,
        r: int = 1,
        beam_size: Optional[int] = None,
        branching: int = 2,
        lam: float = 1.0,
        gamma_C: float = 1.0,
        seed: Optional[int] = None,
        name: str = "eXtreme(Alg3-IGW)",

        # pass-through for reward squashing (affects only eXtreme core)
        squash_rewards: bool = False,
        sigmoid_alpha: float = 1.0,
        sigmoid_beta: float = 0.0,
        sigmoid_clip: float = 35.0,
    ):
        self.num_paths = int(num_paths)
        self.m = int(m)
        self.k = int(k)
        self.r = int(r)

        if self.num_paths <= 0:
            raise ValueError("XtremeAlg3OnPaths: num_paths must be positive")
        if not (1 <= self.m <= self.num_paths):
            raise ValueError("XtremeAlg3OnPaths: m must be in [1, num_paths]")

        self._name = str(name)

        # harness-friendly fields used by your plot_metrics.py
        self.t = 0
        self.total_comparisons = 0
        self.best_G_history: List[float] = []
        self.min_lcb_history: List[float] = []
        self._true_top_m: List[int] = []

        self._x = np.array([1.0], dtype=float)
        self.hierarchy = ArmHierarchy.build_balanced_kary(self.num_paths, branching=branching)

        if beam_size is None:
            beam_size = self.num_paths

        def routing_score(node_id: NodeId, x: Context) -> float:
            return 0.0

        self.core = XtremeAlg3IGW(
            hierarchy=self.hierarchy,
            routing_score=routing_score,
            context_dim=1,
            k=self.k,
            r=self.r,
            beam_size=int(beam_size),
            lam=float(lam),
            gamma_C=float(gamma_C),
            feedback_prob=1.0,
            update_leaf_models=True,

            # IMPORTANT: forward squashing params to core
            squash_rewards=bool(squash_rewards),
            sigmoid_alpha=float(sigmoid_alpha),
            sigmoid_beta=float(sigmoid_beta),
            sigmoid_clip=float(sigmoid_clip),

            seed=seed,
        )

    def _recommend_top_m(self) -> List[int]:
        scores = np.zeros(self.num_paths, dtype=float)
        for a in range(self.num_paths):
            leaf = self.hierarchy.arm_to_leaf_node[a]
            scores[a] = float(self.core._predict_node(self._x, leaf))
        return np.argsort(-scores)[: self.m].tolist()

    def select_and_update(self, oracle: Callable[[int], Reward]) -> Tuple[bool, List[int]]:
        _done, _played = self.core.select_and_update(self._x, oracle)

        # mirror fields expected by harness
        self.t = self.core.t
        self.total_comparisons = self.core.total_comparisons
        self.best_G_history = self.core.best_G_history
        self.min_lcb_history = self.core.min_lcb_history

        top_m = self._recommend_top_m()
        return False, top_m


# ---------------------------------------------------------------------
# Quick sanity test
# ---------------------------------------------------------------------

if __name__ == "__main__":
    rng = np.random.default_rng(0)
    true_means = rng.normal(size=50)

    def oracle(a: int) -> float:
        return float(true_means[a] + 0.1 * rng.normal())

    alg = XtremeAlg3OnPaths(
        num_paths=50,
        m=5,
        k=1,
        r=1,
        seed=1,
        squash_rewards=True,     # turn on sigmoid here if you want
        sigmoid_alpha=5.0,
        sigmoid_beta=0.0,
    )

    for _ in range(200):
        _, topm = alg.select_and_update(oracle)

    print("Estimated top-m:", topm)
    print("True top-m:", np.argsort(-true_means)[:5].tolist())
