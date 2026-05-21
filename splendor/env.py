"""Splendor 2인용 Gymnasium 환경.

설계 요약
---------
- 행동 공간 : Discrete(46), 적법성은 `info["action_mask"]` 로 전달
- 관측      : Flat float32 벡터 (길이 ``SplendorEnv.observation_size``)
- 보상      : ``reward_mode`` 로 선택
        * "sparse" — 종료 시 +1 / -1 / 0, 그 외 0
        * "dense"  — 직전 액션을 한 플레이어의 점수 증가량 + 종료 보너스
- 시점      : 매 step 의 obs/reward 는 "방금 행동한 플레이어" 관점
              (self-play 학습에서 trajectory 를 알아서 분리하기 좋게)

행동 인덱스 (총 46)
~~~~~~~~~~~~~~~~~
    0..9    : 서로 다른 3색 토큰 픽업
    10..14  : 같은 색 2개 토큰 픽업
    15..26  : 보드 12 슬롯 예약 (티어 0~2 × 슬롯 0~3)
    27..29  : 덱 블라인드 예약 (티어 0/1/2)
    30..41  : 보드 12 슬롯 카드 구매
    42..44  : 자기 예약 카드 0/1/2 구매
    45      : 패스
"""
from __future__ import annotations

from dataclasses import dataclass, field
from itertools import combinations
from typing import Optional

import numpy as np

try:
    import gymnasium as gym
    from gymnasium import spaces
except ImportError:  # pragma: no cover
    # gymnasium 미설치 환경에서도 게임 로직 자체는 동작하도록 최소 stub 제공.
    # 실제 PPO 학습 시에는 `pip install gymnasium` 필수.
    class _StubEnv:
        metadata: dict = {}

        def reset(self, *_args, **_kwargs):
            raise NotImplementedError

        def step(self, _action):
            raise NotImplementedError

    class _StubBox:
        def __init__(self, low, high, shape, dtype):
            self.low = low
            self.high = high
            self.shape = shape
            self.dtype = dtype

    class _StubDiscrete:
        def __init__(self, n):
            self.n = n

    class _SpacesModule:
        Box = _StubBox
        Discrete = _StubDiscrete

    class _GymModule:
        Env = _StubEnv

    gym = _GymModule()  # type: ignore[assignment]
    spaces = _SpacesModule()  # type: ignore[assignment]

from splendor.data import (
    CARD_FEATURE_SIZE,
    GOLD,
    N_GEM_COLORS,
    NOBLE_FEATURE_SIZE,
    NOBLES,
    TIER1,
    TIER2,
    TIER3,
    empty_card_features,
    empty_noble_features,
    encode_card,
    encode_noble,
)

# ──────────────────────────────────────────────────────────────────────────
# 상수
# ──────────────────────────────────────────────────────────────────────────
N_PLAYERS = 2
TOKEN_BANK_2P = 4          # 2인 룰에서 색깔별 토큰 수
GOLD_BANK = 5
N_BOARD_CARDS_PER_TIER = 4
N_TIERS = 3
N_BOARD_CARDS = N_TIERS * N_BOARD_CARDS_PER_TIER  # 12
N_NOBLES = 3
MAX_RESERVED = 3
MAX_TOKENS_HELD = 10
WIN_SCORE = 15

# 3색 조합 (오름차순)
THREE_COLOR_COMBOS: list[tuple[int, int, int]] = list(combinations(range(N_GEM_COLORS), 3))
assert len(THREE_COLOR_COMBOS) == 10

# 액션 인덱스 경계
A_TAKE3_START = 0                               # 0..9
A_TAKE2_START = 10                              # 10..14
A_RESERVE_BOARD_START = 15                      # 15..26
A_RESERVE_DECK_START = 27                       # 27..29
A_BUY_BOARD_START = 30                          # 30..41
A_BUY_RESERVED_START = 42                       # 42..44
A_PASS = 45
N_ACTIONS = 46


# ──────────────────────────────────────────────────────────────────────────
# 내부 상태
# ──────────────────────────────────────────────────────────────────────────
@dataclass
class PlayerState:
    tokens: np.ndarray = field(
        default_factory=lambda: np.zeros(N_GEM_COLORS + 1, dtype=np.int32)
    )  # 6 : 5색 + gold
    bonuses: np.ndarray = field(
        default_factory=lambda: np.zeros(N_GEM_COLORS, dtype=np.int32)
    )
    score: int = 0
    reserved: list[tuple[int, int, int, int, int, int, int]] = field(default_factory=list)


# ──────────────────────────────────────────────────────────────────────────
# 환경 본체
# ──────────────────────────────────────────────────────────────────────────
class SplendorEnv(gym.Env):
    metadata = {"render_modes": ["human", "ansi"]}

    def __init__(
        self,
        reward_mode: str = "sparse",
        max_steps: int = 200,
        seed: Optional[int] = None,
        render_mode: Optional[str] = None,
    ):
        super().__init__()
        if reward_mode not in {"sparse", "dense"}:
            raise ValueError(f"reward_mode must be 'sparse' or 'dense', got {reward_mode!r}")
        self.reward_mode = reward_mode
        self.max_steps = max_steps
        self.render_mode = render_mode

        self.action_space = spaces.Discrete(N_ACTIONS)

        # 관측 크기 계산
        # 6 (bank) + 12*11 (board cards) + 3*6 (nobles)
        # + 2 * (6 tokens + 5 bonus + 1 score + 3*11 reserved) = 2 * 45
        # + 3 (deck sizes) + 1 (current player flag) + 1 (final round flag)
        self._obs_size = (
            (N_GEM_COLORS + 1)
            + N_BOARD_CARDS * CARD_FEATURE_SIZE
            + N_NOBLES * NOBLE_FEATURE_SIZE
            + N_PLAYERS
            * (
                (N_GEM_COLORS + 1)
                + N_GEM_COLORS
                + 1
                + MAX_RESERVED * CARD_FEATURE_SIZE
            )
            + N_TIERS
            + 1
            + 1
        )
        self.observation_space = spaces.Box(
            low=-1.0,
            high=50.0,
            shape=(self._obs_size,),
            dtype=np.float32,
        )

        self._np_random: np.random.Generator
        self._seed_initial: Optional[int] = seed
        # 게임 상태 (reset 에서 채움)
        self.bank: np.ndarray
        self.decks: list[list[tuple[int, int, int, int, int, int, int]]]
        self.board: list[list[Optional[tuple[int, int, int, int, int, int, int]]]]
        self.nobles: list[tuple[int, int, int, int, int]]
        self.players: list[PlayerState]
        self.current_player: int
        self.final_round: bool
        self.final_round_trigger: Optional[int]
        self.step_count: int
        self.done: bool

    # ── 사이즈 ──
    @property
    def observation_size(self) -> int:
        return self._obs_size

    # ── reset ──
    def reset(self, *, seed: Optional[int] = None, options=None):  # noqa: ARG002
        del options  # 현재 환경에서는 options 사용하지 않음
        if seed is not None:
            self._np_random = np.random.default_rng(seed)
        elif not hasattr(self, "_np_random"):
            self._np_random = np.random.default_rng(self._seed_initial)

        # 토큰 뱅크
        self.bank = np.array(
            [TOKEN_BANK_2P] * N_GEM_COLORS + [GOLD_BANK], dtype=np.int32
        )

        # 덱 셔플
        self.decks = [list(TIER1), list(TIER2), list(TIER3)]
        for d in self.decks:
            self._np_random.shuffle(d)

        # 보드 카드 4장씩 배치
        self.board = []
        for t in range(N_TIERS):
            slots: list[Optional[tuple]] = []
            for _ in range(N_BOARD_CARDS_PER_TIER):
                slots.append(self.decks[t].pop() if self.decks[t] else None)
            self.board.append(slots)

        # 노블 3장 선택
        noble_idx = self._np_random.choice(len(NOBLES), size=N_NOBLES, replace=False)
        self.nobles = [NOBLES[i] for i in noble_idx]

        self.players = [PlayerState() for _ in range(N_PLAYERS)]
        self.current_player = 0
        self.final_round = False
        self.final_round_trigger = None
        self.step_count = 0
        self.done = False

        return self._obs(), self._info()

    # ── step ──
    def step(self, action: int):
        if self.done:
            raise RuntimeError("이미 종료된 에피소드입니다. reset() 호출 필요.")
        action = int(action)
        actor = self.current_player
        prev_score = self.players[actor].score

        mask = self._action_mask()
        if not mask[action]:
            # 불법 행동 — 강제로 PASS 처리 + 페널티
            illegal = True
            action = A_PASS
        else:
            illegal = False

        self._apply_action(actor, action)
        # 턴 종료 시 노블 체크
        self._check_nobles(actor)

        # 승리 조건 트리거
        if self.players[actor].score >= WIN_SCORE and not self.final_round:
            self.final_round = True
            self.final_round_trigger = actor

        # 다음 플레이어
        next_player = 1 - actor
        terminated = False

        # 마지막 라운드 종결 체크 : trigger 플레이어가 다시 자기 차례가 되는 시점
        if self.final_round and next_player == self.final_round_trigger:
            terminated = True

        self.step_count += 1
        truncated = (not terminated) and self.step_count >= self.max_steps

        # 보상 — truncated 시점에는 sparse 승패 보상 안 줌 (게임이 미완)
        reward = self._compute_reward(actor, prev_score, terminated)
        if illegal:
            reward -= 0.5  # 불법 행동 페널티

        if terminated or truncated:
            self.done = True
        else:
            self.current_player = next_player

        info = self._info(last_actor=actor)
        if terminated or truncated:
            info["final_rewards"] = self._final_rewards(truncated)
        return self._obs(), float(reward), terminated, truncated, info

    # ── 보상 ──
    def _compute_reward(
        self, actor: int, prev_score: int, terminated: bool
    ) -> float:
        r = 0.0
        if self.reward_mode == "dense":
            r += float(self.players[actor].score - prev_score)
        if terminated:
            s0, s1 = self.players[0].score, self.players[1].score
            # 동점 시 카드 수가 적은 쪽 승 (스플렌더 공식 타이브레이커)
            if s0 == s1:
                c0 = int(self.players[0].bonuses.sum()) + len(self.players[0].reserved)
                c1 = int(self.players[1].bonuses.sum()) + len(self.players[1].reserved)
                winner = 0 if c0 < c1 else (1 if c1 < c0 else -1)
            else:
                winner = 0 if s0 > s1 else 1
            if winner == -1:
                r += 0.0
            elif winner == actor:
                r += 1.0
            else:
                r += -1.0
        return r

    # ── render ──
    # ── 양쪽 플레이어 관점의 종료 보상 (self-play 후처리용) ──
    def _final_rewards(self, truncated: bool) -> tuple[float, float]:
        s0, s1 = self.players[0].score, self.players[1].score
        if truncated:
            return (0.0, 0.0)
        if s0 == s1:
            c0 = int(self.players[0].bonuses.sum()) + len(self.players[0].reserved)
            c1 = int(self.players[1].bonuses.sum()) + len(self.players[1].reserved)
            if c0 < c1:
                return (1.0, -1.0)
            if c1 < c0:
                return (-1.0, 1.0)
            return (0.0, 0.0)
        return (1.0, -1.0) if s0 > s1 else (-1.0, 1.0)

    def render(self):
        if self.render_mode is None:
            return None
        lines = []
        lines.append(
            f"[step {self.step_count}] current={self.current_player} "
            f"final_round={self.final_round}"
        )
        names = "W B G R K Au".split()
        lines.append("Bank   : " + " ".join(f"{n}={v}" for n, v in zip(names, self.bank)))
        for t in range(N_TIERS):
            row = []
            for c in self.board[t]:
                if c is None:
                    row.append("[--]")
                else:
                    bonus, pts = c[0], c[1]
                    cost = c[2:7]
                    row.append(
                        f"[{'WBGRK'[bonus]}{pts} c=({','.join(str(x) for x in cost)})]"
                    )
            lines.append(f"Tier{t + 1} : " + " ".join(row))
        for i, p in enumerate(self.players):
            tok = " ".join(f"{n}={v}" for n, v in zip(names, p.tokens))
            bon = " ".join(f"{n}={v}" for n, v in zip("WBGRK", p.bonuses))
            lines.append(
                f"P{i} sc={p.score} tokens=({tok}) bonus=({bon}) "
                f"reserved={len(p.reserved)}"
            )
        text = "\n".join(lines)
        if self.render_mode == "human":
            print(text)
            return None
        return text

    # ──────────────────────────────────────────────────────────────────
    # 액션 적법성 / 적용
    # ──────────────────────────────────────────────────────────────────
    def _action_mask(self) -> np.ndarray:
        mask = np.zeros(N_ACTIONS, dtype=bool)
        p = self.players[self.current_player]
        held_total = int(p.tokens.sum())

        # take-3
        for i, combo in enumerate(THREE_COLOR_COMBOS):
            avail = [c for c in combo if self.bank[c] > 0]
            if not avail:
                continue
            if held_total + len(avail) <= MAX_TOKENS_HELD:
                mask[A_TAKE3_START + i] = True

        # take-2 (같은 색) — 룰: 해당 색 ≥4 남아있어야 함
        for c in range(N_GEM_COLORS):
            if self.bank[c] >= 4 and held_total + 2 <= MAX_TOKENS_HELD:
                mask[A_TAKE2_START + c] = True

        # 예약 (보드 12 슬롯) — 자기 예약 < 3, 슬롯에 카드 있어야 함
        if len(p.reserved) < MAX_RESERVED:
            for slot in range(N_BOARD_CARDS):
                tier, idx = divmod(slot, N_BOARD_CARDS_PER_TIER)
                if self.board[tier][idx] is not None:
                    mask[A_RESERVE_BOARD_START + slot] = True
            # 덱 블라인드 예약
            for t in range(N_TIERS):
                if self.decks[t]:
                    mask[A_RESERVE_DECK_START + t] = True

        # 구매 (보드)
        for slot in range(N_BOARD_CARDS):
            tier, idx = divmod(slot, N_BOARD_CARDS_PER_TIER)
            card = self.board[tier][idx]
            if card is not None and self._can_afford(p, card):
                mask[A_BUY_BOARD_START + slot] = True

        # 구매 (예약된 카드)
        for i, card in enumerate(p.reserved):
            if self._can_afford(p, card):
                mask[A_BUY_RESERVED_START + i] = True

        # 어떤 행동도 못 하면 패스
        if not mask.any():
            mask[A_PASS] = True

        return mask

    def _can_afford(
        self, p: PlayerState, card: tuple[int, int, int, int, int, int, int]
    ) -> bool:
        cost = card[2:7]
        deficit = 0
        for c in range(N_GEM_COLORS):
            need = max(0, cost[c] - int(p.bonuses[c]))
            short = need - int(p.tokens[c])
            if short > 0:
                deficit += short
        return deficit <= int(p.tokens[GOLD])

    def _pay_card(
        self, p: PlayerState, card: tuple[int, int, int, int, int, int, int]
    ) -> None:
        cost = card[2:7]
        for c in range(N_GEM_COLORS):
            need = max(0, cost[c] - int(p.bonuses[c]))
            use = min(need, int(p.tokens[c]))
            p.tokens[c] -= use
            self.bank[c] += use
            short = need - use
            if short > 0:
                p.tokens[GOLD] -= short
                self.bank[GOLD] += short

    def _apply_action(self, actor: int, action: int) -> None:
        p = self.players[actor]

        if A_TAKE3_START <= action < A_TAKE3_START + 10:
            combo = THREE_COLOR_COMBOS[action - A_TAKE3_START]
            for c in combo:
                if self.bank[c] > 0:
                    self.bank[c] -= 1
                    p.tokens[c] += 1
            return

        if A_TAKE2_START <= action < A_TAKE2_START + N_GEM_COLORS:
            c = action - A_TAKE2_START
            self.bank[c] -= 2
            p.tokens[c] += 2
            return

        if A_RESERVE_BOARD_START <= action < A_RESERVE_BOARD_START + N_BOARD_CARDS:
            slot = action - A_RESERVE_BOARD_START
            tier, idx = divmod(slot, N_BOARD_CARDS_PER_TIER)
            card = self.board[tier][idx]
            assert card is not None
            p.reserved.append(card)
            # 슬롯 보충
            self.board[tier][idx] = self.decks[tier].pop() if self.decks[tier] else None
            # 골드 토큰 1개 (가능하면)
            if self.bank[GOLD] > 0 and int(p.tokens.sum()) < MAX_TOKENS_HELD:
                self.bank[GOLD] -= 1
                p.tokens[GOLD] += 1
            return

        if A_RESERVE_DECK_START <= action < A_RESERVE_DECK_START + N_TIERS:
            tier = action - A_RESERVE_DECK_START
            card = self.decks[tier].pop()
            p.reserved.append(card)
            if self.bank[GOLD] > 0 and int(p.tokens.sum()) < MAX_TOKENS_HELD:
                self.bank[GOLD] -= 1
                p.tokens[GOLD] += 1
            return

        if A_BUY_BOARD_START <= action < A_BUY_BOARD_START + N_BOARD_CARDS:
            slot = action - A_BUY_BOARD_START
            tier, idx = divmod(slot, N_BOARD_CARDS_PER_TIER)
            card = self.board[tier][idx]
            assert card is not None
            self._pay_card(p, card)
            p.bonuses[card[0]] += 1
            p.score += int(card[1])
            self.board[tier][idx] = self.decks[tier].pop() if self.decks[tier] else None
            return

        if A_BUY_RESERVED_START <= action < A_BUY_RESERVED_START + MAX_RESERVED:
            i = action - A_BUY_RESERVED_START
            card = p.reserved.pop(i)
            self._pay_card(p, card)
            p.bonuses[card[0]] += 1
            p.score += int(card[1])
            return

        if action == A_PASS:
            return

        raise ValueError(f"알 수 없는 액션 {action}")

    def _check_nobles(self, actor: int) -> None:
        p = self.players[actor]
        for i, noble in enumerate(self.nobles):
            if noble is None:
                continue
            if all(int(p.bonuses[c]) >= noble[c] for c in range(N_GEM_COLORS)):
                p.score += 3
                self.nobles[i] = None  # type: ignore[index]
                break  # 한 턴에 1장만 (단순화)

    # ──────────────────────────────────────────────────────────────────
    # 관측 / 정보
    # ──────────────────────────────────────────────────────────────────
    def _obs(self) -> np.ndarray:
        feats: list[float] = []
        feats.extend(float(x) for x in self.bank)

        for t in range(N_TIERS):
            for c in self.board[t]:
                feats.extend(encode_card(c) if c is not None else empty_card_features())

        for n in self.nobles:
            feats.extend(encode_noble(n) if n is not None else empty_noble_features())

        # current player 부터 상대 순으로 인코딩 — self-play 학습에 유리
        order = [self.current_player, 1 - self.current_player]
        for idx in order:
            p = self.players[idx]
            feats.extend(float(x) for x in p.tokens)
            feats.extend(float(x) for x in p.bonuses)
            feats.append(float(p.score))
            for r_idx in range(MAX_RESERVED):
                if r_idx < len(p.reserved):
                    feats.extend(encode_card(p.reserved[r_idx]))
                else:
                    feats.extend(empty_card_features())

        feats.extend(float(len(d)) for d in self.decks)
        feats.append(float(self.current_player))
        feats.append(1.0 if self.final_round else 0.0)

        arr = np.asarray(feats, dtype=np.float32)
        assert arr.shape[0] == self._obs_size, (
            f"obs size mismatch: got {arr.shape[0]}, expected {self._obs_size}"
        )
        return arr

    def _info(self, last_actor: Optional[int] = None) -> dict:
        return {
            "action_mask": self._action_mask(),
            "current_player": self.current_player,
            "last_actor": last_actor,
            "scores": (self.players[0].score, self.players[1].score),
        }

    # ── 편의 ──
    def legal_actions(self) -> np.ndarray:
        return self._action_mask()
