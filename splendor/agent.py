"""PPO 학습용 신경망 정책 & 상대 자동 진행 wrapper.

핵심 구성
---------
- ``ActorCritic`` : MLP 기반 정책-가치 공유 네트워크. 마스킹된
  ``Categorical`` 분포로 행동을 샘플링한다.
- ``PPOOpponentEnv`` : ``SplendorEnv`` 위에서 *PPO 시점의 step* 만 노출하는
  wrapper. 상대 차례는 내부에서 고정 정책으로 자동 진행되며, 보상은
  종료 시 ``final_rewards[ppo_player]`` 만 반환한다 (sparse).
- ``make_ppo_policy_fn`` : 학습된 모델을 ``evaluate_vs_policy`` 가 기대하는
  ``(obs, mask, rng) -> action`` 형태로 감싸주는 헬퍼.
"""
from __future__ import annotations

from typing import Callable, Optional

import numpy as np

try:
    import torch
    import torch.nn as nn
    from torch.distributions import Categorical
except ImportError as e:  # pragma: no cover
    raise ImportError(
        "PyTorch 가 필요합니다. venv 활성화 후 `pip install torch` 하세요."
    ) from e

from splendor.env import N_ACTIONS, SplendorEnv
from splendor.metrics import MetricsWrapper, random_policy


# ──────────────────────────────────────────────────────────────────────────
# 네트워크
# ──────────────────────────────────────────────────────────────────────────
class ActorCritic(nn.Module):
    """MLP 기반 actor-critic. 정책과 가치는 마지막 hidden 을 공유."""

    def __init__(self, obs_dim: int, n_actions: int = N_ACTIONS, hidden: int = 256):
        super().__init__()
        self.shared = nn.Sequential(
            nn.Linear(obs_dim, hidden),
            nn.Tanh(),
            nn.Linear(hidden, hidden),
            nn.Tanh(),
        )
        self.policy_head = nn.Linear(hidden, n_actions)
        self.value_head = nn.Linear(hidden, 1)
        # 직교 초기화 — PPO 표준
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.orthogonal_(m.weight, gain=np.sqrt(2))
                nn.init.zeros_(m.bias)
        nn.init.orthogonal_(self.policy_head.weight, gain=0.01)
        nn.init.orthogonal_(self.value_head.weight, gain=1.0)

    def forward(self, obs: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        h = self.shared(obs)
        return self.policy_head(h), self.value_head(h).squeeze(-1)

    @staticmethod
    def _masked_logits(logits: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        # mask: bool 텐서, True=합법
        very_neg = torch.finfo(logits.dtype).min
        return torch.where(mask, logits, torch.full_like(logits, very_neg))

    def act(
        self, obs: torch.Tensor, mask: torch.Tensor, deterministic: bool = False
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        logits, value = self.forward(obs)
        logits = self._masked_logits(logits, mask)
        dist = Categorical(logits=logits)
        if deterministic:
            action = torch.argmax(logits, dim=-1)
        else:
            action = dist.sample()
        log_prob = dist.log_prob(action)
        entropy = dist.entropy()
        return action, log_prob, entropy, value

    def evaluate_actions(
        self, obs: torch.Tensor, mask: torch.Tensor, action: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        logits, value = self.forward(obs)
        logits = self._masked_logits(logits, mask)
        dist = Categorical(logits=logits)
        return dist.log_prob(action), dist.entropy(), value


# ──────────────────────────────────────────────────────────────────────────
# PPO 시점 wrapper
# ──────────────────────────────────────────────────────────────────────────
PolicyFn = Callable[[np.ndarray, np.ndarray, np.random.Generator], int]


class PPOOpponentEnv:
    """PPO 한 명의 시점에서 step 을 정의. 상대 차례는 내부 자동 진행.

    학습 측은 ``reset()`` 후 ``step(action)`` 만 호출하면, 자기 행동의
    *그리고 이어지는 상대 행동까지* 한 step 으로 묶여 결과가 나온다.

    보상은 sparse — 게임이 종결되지 않으면 0, 종결되면 PPO 시점의
    ``final_rewards[ppo_player]`` (±1).
    """

    def __init__(
        self,
        env: Optional[SplendorEnv] = None,
        opponent_policy: PolicyFn = random_policy,
        ppo_player: int = 0,
        max_steps: int = 400,
        seed: Optional[int] = None,
    ):
        self.env = env or SplendorEnv(max_steps=max_steps, seed=seed)
        if not isinstance(self.env, MetricsWrapper):
            self.env = MetricsWrapper(self.env)
        self.opponent_policy = opponent_policy
        self.ppo_player = ppo_player
        self.rng = np.random.default_rng(seed)
        self.action_space = self.env.action_space
        self.observation_space = self.env.observation_space

    @property
    def observation_size(self) -> int:
        return self.env.observation_size

    def reset(self, *, seed: Optional[int] = None):
        if seed is not None:
            self.rng = np.random.default_rng(seed)
        obs, info = self.env.reset(seed=seed)
        obs, info, _term, _trunc = self._advance_to_ppo(obs, info)
        return obs, info

    def step(self, action: int):
        obs, _r, term, trunc, info = self.env.step(int(action))
        if not (term or trunc):
            obs, info, term, trunc = self._advance_to_ppo(obs, info)

        if term or trunc:
            final = info.get("final_rewards", (0.0, 0.0))
            return obs, float(final[self.ppo_player]), term, trunc, info
        return obs, 0.0, term, trunc, info

    def _advance_to_ppo(self, obs, info):
        while info["current_player"] != self.ppo_player:
            a = self.opponent_policy(obs, info["action_mask"], self.rng)
            obs, _r, term, trunc, info = self.env.step(a)
            if term or trunc:
                return obs, info, term, trunc
        return obs, info, False, False


# ──────────────────────────────────────────────────────────────────────────
# 평가용 정책 헬퍼
# ──────────────────────────────────────────────────────────────────────────
def make_ppo_policy_fn(
    model: ActorCritic,
    device: torch.device | str = "cpu",
    deterministic: bool = True,
) -> PolicyFn:
    """학습된 모델을 ``evaluate_vs_policy`` 에 넘길 수 있는 PolicyFn 으로 감쌈."""
    device = torch.device(device)

    def _fn(obs: np.ndarray, mask: np.ndarray, _rng: np.random.Generator) -> int:
        with torch.no_grad():
            obs_t = torch.as_tensor(obs, dtype=torch.float32, device=device).unsqueeze(0)
            mask_t = torch.as_tensor(mask, dtype=torch.bool, device=device).unsqueeze(0)
            action, _, _, _ = model.act(obs_t, mask_t, deterministic=deterministic)
        return int(action.item())

    return _fn
