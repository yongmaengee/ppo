"""에피소드 단위 지표 수집과 상대 에이전트 평가 유틸.

크게 두 부분.

1. ``MetricsWrapper`` — ``SplendorEnv`` 를 감싸 행동 분포 / 티어별 구매 /
   노블 획득 / 불법 행동 시도 등을 누적. 에피소드가 끝나면
   ``info["episode_stats"]`` 에 dict 로 채워준다. SB3/CleanRL 학습 루프 어디든
   info 만 보면 그대로 로깅 가능.

2. ``evaluate_vs_policy`` — 학습 정책과 상대 정책을 두 자리로 번갈아 앉혀
   N 게임 두고 승률·평균 점수·평균 step 등을 반환. ``random_policy`` 와
   휴리스틱 베이스라인 ``greedy_buy_policy`` 도 함께 제공.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable, Optional

import numpy as np

from splendor.env import (
    A_BUY_BOARD_START,
    A_BUY_RESERVED_START,
    A_PASS,
    A_RESERVE_BOARD_START,
    A_RESERVE_DECK_START,
    A_TAKE2_START,
    A_TAKE3_START,
    MAX_RESERVED,
    N_BOARD_CARDS,
    N_BOARD_CARDS_PER_TIER,
    N_TIERS,
    SplendorEnv,
)


# ──────────────────────────────────────────────────────────────────────────
# 액션 카테고리화
# ──────────────────────────────────────────────────────────────────────────
ACTION_CATEGORIES = ("take3", "take2", "reserve_board", "reserve_deck", "buy_board", "buy_reserved", "pass")


def action_category(action: int) -> str:
    if A_TAKE3_START <= action < A_TAKE3_START + 10:
        return "take3"
    if A_TAKE2_START <= action < A_TAKE2_START + 5:
        return "take2"
    if A_RESERVE_BOARD_START <= action < A_RESERVE_BOARD_START + N_BOARD_CARDS:
        return "reserve_board"
    if A_RESERVE_DECK_START <= action < A_RESERVE_DECK_START + N_TIERS:
        return "reserve_deck"
    if A_BUY_BOARD_START <= action < A_BUY_BOARD_START + N_BOARD_CARDS:
        return "buy_board"
    if A_BUY_RESERVED_START <= action < A_BUY_RESERVED_START + MAX_RESERVED:
        return "buy_reserved"
    if action == A_PASS:
        return "pass"
    raise ValueError(f"unknown action {action}")


def buy_board_tier(action: int) -> Optional[int]:
    """보드 구매 행동이면 티어(0~2) 반환, 아니면 None."""
    if A_BUY_BOARD_START <= action < A_BUY_BOARD_START + N_BOARD_CARDS:
        slot = action - A_BUY_BOARD_START
        return slot // N_BOARD_CARDS_PER_TIER
    return None


# ──────────────────────────────────────────────────────────────────────────
# 통계 누적용 dataclass
# ──────────────────────────────────────────────────────────────────────────
@dataclass
class _PlayerEpisodeStats:
    action_counts: dict[str, int] = field(default_factory=lambda: {c: 0 for c in ACTION_CATEGORIES})
    tier_buys: list[int] = field(default_factory=lambda: [0] * N_TIERS)
    illegal_attempts: int = 0
    gold_used: int = 0  # 구매 시점에 사용한 골드 토큰 수
    nobles_gained: int = 0


# ──────────────────────────────────────────────────────────────────────────
# Wrapper
# ──────────────────────────────────────────────────────────────────────────
class MetricsWrapper:
    """SplendorEnv 를 감싸 에피소드 통계를 누적.

    PEP 8 명칭은 다르지만 gym.Wrapper 가 없어도 동작하도록 단순 위임으로 구현.
    필요한 메서드/속성만 노출한다.
    """

    def __init__(self, env: SplendorEnv):
        self.env = env
        self._stats: list[_PlayerEpisodeStats] = []
        self._steps: int = 0
        self._prev_nobles_remaining: int = 0
        # 패스-스루
        self.action_space = env.action_space
        self.observation_space = env.observation_space

    # ── 패스-스루 속성 ──
    @property
    def observation_size(self) -> int:
        return self.env.observation_size

    def render(self):
        return self.env.render()

    def legal_actions(self) -> np.ndarray:
        return self.env.legal_actions()

    # ── reset ──
    def reset(self, *, seed: Optional[int] = None, options=None):
        obs, info = self.env.reset(seed=seed, options=options)
        self._stats = [_PlayerEpisodeStats() for _ in range(2)]
        self._steps = 0
        self._prev_nobles_remaining = sum(1 for n in self.env.nobles if n is not None)
        return obs, info

    # ── step ──
    def step(self, action: int):
        actor = self.env.current_player
        mask = self.env.legal_actions()
        illegal = not bool(mask[int(action)])

        # 골드 사용 계산을 위해 구매 직전 골드 보유량 기록
        gold_before = int(self.env.players[actor].tokens[5])
        nobles_before = sum(1 for n in self.env.nobles if n is not None)

        obs, reward, terminated, truncated, info = self.env.step(action)

        ps = self._stats[actor]
        if illegal:
            ps.illegal_attempts += 1
        # 실제 환경 내부에서 불법 행동은 PASS 로 강제됨 — 카테고리는 의도된 액션 기준
        ps.action_counts[action_category(int(action))] += 1

        tier = buy_board_tier(int(action))
        if tier is not None and not illegal:
            ps.tier_buys[tier] += 1
        if A_BUY_RESERVED_START <= int(action) < A_BUY_RESERVED_START + MAX_RESERVED and not illegal:
            # 예약 카드 구매 — 티어 추적은 어려우니 별도 카운트만 (buy_reserved 액션 카운트로 대신)
            pass

        gold_after = int(self.env.players[actor].tokens[5])
        # 구매 시점 외에는 골드가 변하지 않거나 +1(예약), 0(토큰). 감소분만 사용량.
        if gold_after < gold_before:
            ps.gold_used += gold_before - gold_after

        nobles_after = sum(1 for n in self.env.nobles if n is not None)
        if nobles_after < nobles_before:
            ps.nobles_gained += nobles_before - nobles_after

        self._steps += 1

        if terminated or truncated:
            info["episode_stats"] = self._summarize(terminated, truncated, info)

        return obs, reward, terminated, truncated, info

    # ── 종료 시 요약 ──
    def _summarize(self, terminated: bool, truncated: bool, info: dict) -> dict:
        s0, s1 = info["scores"]
        final_rewards = info.get("final_rewards", (0.0, 0.0))
        winner: Optional[int]
        if final_rewards[0] > final_rewards[1]:
            winner = 0
        elif final_rewards[1] > final_rewards[0]:
            winner = 1
        else:
            winner = None

        per_player = []
        for i in (0, 1):
            ps = self._stats[i]
            total_actions = sum(ps.action_counts.values())
            per_player.append(
                {
                    "score": int(info["scores"][i]),
                    "bonuses": int(self.env.players[i].bonuses.sum()),
                    "reserved": len(self.env.players[i].reserved),
                    "nobles_gained": ps.nobles_gained,
                    "gold_used": ps.gold_used,
                    "illegal_attempts": ps.illegal_attempts,
                    "action_counts": dict(ps.action_counts),
                    "action_fractions": {
                        k: (v / total_actions if total_actions else 0.0)
                        for k, v in ps.action_counts.items()
                    },
                    "tier_buys": list(ps.tier_buys),
                }
            )

        return {
            "steps": self._steps,
            "terminated": terminated,
            "truncated": truncated,
            "winner": winner,  # 0 / 1 / None(draw)
            "scores": (int(s0), int(s1)),
            "per_player": per_player,
        }


# ──────────────────────────────────────────────────────────────────────────
# 기본 정책들
# ──────────────────────────────────────────────────────────────────────────
PolicyFn = Callable[[np.ndarray, np.ndarray, np.random.Generator], int]


def random_policy(obs: np.ndarray, mask: np.ndarray, rng: np.random.Generator) -> int:
    del obs
    legal = np.flatnonzero(mask)
    return int(rng.choice(legal))


def greedy_buy_policy(obs: np.ndarray, mask: np.ndarray, rng: np.random.Generator) -> int:
    """살 수 있으면 산다. 그 외엔 take-3 → take-2 → 예약 → 패스 우선순위.

    학습 정책의 베이스라인용. obs 는 사용하지 않고 mask 만 본다.
    """
    del obs
    priority = (
        list(range(A_BUY_BOARD_START, A_BUY_BOARD_START + N_BOARD_CARDS)),
        list(range(A_BUY_RESERVED_START, A_BUY_RESERVED_START + MAX_RESERVED)),
        list(range(A_TAKE3_START, A_TAKE3_START + 10)),
        list(range(A_TAKE2_START, A_TAKE2_START + 5)),
        list(range(A_RESERVE_BOARD_START, A_RESERVE_BOARD_START + N_BOARD_CARDS)),
        list(range(A_RESERVE_DECK_START, A_RESERVE_DECK_START + N_TIERS)),
        [A_PASS],
    )
    for group in priority:
        candidates = [a for a in group if mask[a]]
        if candidates:
            return int(rng.choice(candidates))
    # mask 가 비어있을 일은 없지만 안전망
    return A_PASS


# ──────────────────────────────────────────────────────────────────────────
# 평가
# ──────────────────────────────────────────────────────────────────────────
def evaluate_vs_policy(
    learner_policy: PolicyFn,
    opponent_policy: PolicyFn = random_policy,
    n_episodes: int = 100,
    seed: int = 0,
    max_steps: int = 400,
    swap_sides: bool = True,
) -> dict:
    """두 정책을 N 게임 두게 하고 학습 정책 기준 지표 반환.

    ``swap_sides=True`` 면 절반은 학습 정책이 P0, 나머지 절반은 P1 으로 배치.
    """
    rng = np.random.default_rng(seed)
    n_wins = 0
    n_draws = 0
    total_steps = 0
    learner_scores: list[int] = []
    opponent_scores: list[int] = []
    ep_stats_all: list[dict] = []

    for ep in range(n_episodes):
        learner_is_p0 = (ep % 2 == 0) or (not swap_sides)
        env = MetricsWrapper(SplendorEnv(max_steps=max_steps, seed=seed + ep))
        obs, info = env.reset(seed=seed + ep)

        while True:
            actor = info["current_player"]
            mask = info["action_mask"]
            policy = (
                learner_policy
                if (actor == 0) == learner_is_p0
                else opponent_policy
            )
            action = policy(obs, mask, rng)
            obs, _r, terminated, truncated, info = env.step(action)
            if terminated or truncated:
                stats = info["episode_stats"]
                ep_stats_all.append(stats)
                total_steps += stats["steps"]
                learner_idx = 0 if learner_is_p0 else 1
                opp_idx = 1 - learner_idx
                learner_scores.append(stats["scores"][learner_idx])
                opponent_scores.append(stats["scores"][opp_idx])
                if stats["winner"] is None:
                    n_draws += 1
                elif stats["winner"] == learner_idx:
                    n_wins += 1
                break

    # 학습 정책 시점의 행동 분포 / 티어 분포 평균
    action_frac_sum = {k: 0.0 for k in ACTION_CATEGORIES}
    tier_buy_sum = [0.0] * N_TIERS
    nobles_sum = 0
    gold_used_sum = 0
    illegal_sum = 0
    for ep, stats in enumerate(ep_stats_all):
        learner_is_p0 = (ep % 2 == 0) or (not swap_sides)
        learner_idx = 0 if learner_is_p0 else 1
        pstats = stats["per_player"][learner_idx]
        for k, v in pstats["action_fractions"].items():
            action_frac_sum[k] += v
        for i, v in enumerate(pstats["tier_buys"]):
            tier_buy_sum[i] += v
        nobles_sum += pstats["nobles_gained"]
        gold_used_sum += pstats["gold_used"]
        illegal_sum += pstats["illegal_attempts"]

    n = max(n_episodes, 1)
    return {
        "n_episodes": n_episodes,
        "win_rate": n_wins / n,
        "draw_rate": n_draws / n,
        "loss_rate": (n_episodes - n_wins - n_draws) / n,
        "mean_steps": total_steps / n,
        "mean_learner_score": float(np.mean(learner_scores)) if learner_scores else 0.0,
        "mean_opponent_score": float(np.mean(opponent_scores)) if opponent_scores else 0.0,
        "mean_action_fractions": {k: v / n for k, v in action_frac_sum.items()},
        "mean_tier_buys": [v / n for v in tier_buy_sum],
        "mean_nobles_gained": nobles_sum / n,
        "mean_gold_used": gold_used_sum / n,
        "total_illegal_attempts": illegal_sum,
    }
