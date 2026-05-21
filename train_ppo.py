"""마스킹된 PPO 로 SplendorEnv 의 P0 자리를 학습.

상대(P1)는 고정 정책 (기본 random, ``--opponent greedy`` 로 변경 가능).
한 번에 한 환경, CPU 기준 — 가벼운 검증 용도. CleanRL 단일 파일 스타일.

실행 예:
    python train_ppo.py --updates 200 --opponent random
    python train_ppo.py --updates 500 --opponent greedy --eval-episodes 200

각 update 마다 출력되는 지표:
    rollout_return       : 최근 rollout 에서 종료된 게임들의 평균 PPO 보상
    policy_loss / vloss  : PPO 손실
    entropy / approx_kl  : 정책 분산 / 이전 정책과의 차이
    eval_win_rate_*      : --opponent 대비 승률 (매 eval_interval 마다)
"""
from __future__ import annotations

import argparse
import time
from dataclasses import dataclass

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

from splendor import (
    SplendorEnv,
    evaluate_vs_policy,
    greedy_buy_policy,
    random_policy,
)
from splendor.agent import ActorCritic, PPOOpponentEnv, make_ppo_policy_fn
from splendor.env import N_ACTIONS


# ──────────────────────────────────────────────────────────────────────────
@dataclass
class Args:
    updates: int = 200
    n_steps: int = 1024       # rollout 길이 (PPO 시점 step 수)
    epochs: int = 4
    minibatch_size: int = 256
    gamma: float = 0.99
    gae_lambda: float = 0.95
    clip_eps: float = 0.2
    entropy_coef: float = 0.01
    value_coef: float = 0.5
    max_grad_norm: float = 0.5
    lr: float = 3e-4
    opponent: str = "random"   # 'random' or 'greedy'
    eval_interval: int = 10
    eval_episodes: int = 60
    seed: int = 0
    device: str = "cpu"
    log_interval: int = 1


def parse_args() -> Args:
    p = argparse.ArgumentParser()
    for k in Args.__dataclass_fields__:
        default = getattr(Args, k)
        t = type(default)
        if t is bool:
            p.add_argument(f"--{k.replace('_', '-')}", action="store_true")
        else:
            p.add_argument(f"--{k.replace('_', '-')}", type=t, default=default)
    raw = p.parse_args()
    return Args(**{k: getattr(raw, k) for k in Args.__dataclass_fields__})


# ──────────────────────────────────────────────────────────────────────────
def _get_opponent(name: str):
    if name == "random":
        return random_policy
    if name == "greedy":
        return greedy_buy_policy
    raise ValueError(f"unknown opponent {name!r}")


# ──────────────────────────────────────────────────────────────────────────
class RolloutBuffer:
    def __init__(self, n_steps: int, obs_dim: int, n_actions: int, device: torch.device):
        self.obs = torch.zeros((n_steps, obs_dim), dtype=torch.float32, device=device)
        self.masks = torch.zeros((n_steps, n_actions), dtype=torch.bool, device=device)
        self.actions = torch.zeros(n_steps, dtype=torch.long, device=device)
        self.logprobs = torch.zeros(n_steps, dtype=torch.float32, device=device)
        self.values = torch.zeros(n_steps, dtype=torch.float32, device=device)
        self.rewards = torch.zeros(n_steps, dtype=torch.float32, device=device)
        self.dones = torch.zeros(n_steps, dtype=torch.float32, device=device)
        self.advantages = torch.zeros(n_steps, dtype=torch.float32, device=device)
        self.returns = torch.zeros(n_steps, dtype=torch.float32, device=device)
        self.n_steps = n_steps

    def compute_gae(self, last_value: torch.Tensor, gamma: float, lam: float):
        gae = torch.zeros(1, device=self.values.device)
        for t in reversed(range(self.n_steps)):
            next_value = last_value if t == self.n_steps - 1 else self.values[t + 1]
            next_nonterminal = 1.0 - self.dones[t]
            delta = self.rewards[t] + gamma * next_value * next_nonterminal - self.values[t]
            gae = delta + gamma * lam * next_nonterminal * gae
            self.advantages[t] = gae.squeeze()
        self.returns = self.advantages + self.values


# ──────────────────────────────────────────────────────────────────────────
def collect_rollout(env: PPOOpponentEnv, model: ActorCritic, buf: RolloutBuffer,
                    device: torch.device, ep_returns: list[float]):
    """rollout 한 번 채우고, 종료된 에피소드들의 PPO 시점 누적 보상을 ep_returns 에 추가."""
    obs, info = (env.last_obs, env.last_info) if hasattr(env, "last_obs") else env.reset()
    cur_return = 0.0
    for t in range(buf.n_steps):
        obs_t = torch.as_tensor(obs, dtype=torch.float32, device=device).unsqueeze(0)
        mask_t = torch.as_tensor(info["action_mask"], dtype=torch.bool, device=device).unsqueeze(0)
        with torch.no_grad():
            action, log_prob, _ent, value = model.act(obs_t, mask_t)
        buf.obs[t] = obs_t.squeeze(0)
        buf.masks[t] = mask_t.squeeze(0)
        buf.actions[t] = action.squeeze()
        buf.logprobs[t] = log_prob.squeeze()
        buf.values[t] = value.squeeze()

        next_obs, reward, term, trunc, next_info = env.step(int(action.item()))
        done = term or trunc
        buf.rewards[t] = float(reward)
        buf.dones[t] = float(done)
        cur_return += float(reward)
        if done:
            ep_returns.append(cur_return)
            cur_return = 0.0
            next_obs, next_info = env.reset()
        obs, info = next_obs, next_info

    # bootstrap value
    with torch.no_grad():
        obs_t = torch.as_tensor(obs, dtype=torch.float32, device=device).unsqueeze(0)
        _, last_value = model.forward(obs_t)
    # 다음 호출에서 상태 이어가도록 보관
    env.last_obs = obs
    env.last_info = info
    return last_value.squeeze()


# ──────────────────────────────────────────────────────────────────────────
def ppo_update(model: ActorCritic, optimizer: optim.Optimizer, buf: RolloutBuffer,
               args: Args) -> dict:
    n = buf.n_steps
    indices = np.arange(n)
    adv = buf.advantages
    adv = (adv - adv.mean()) / (adv.std() + 1e-8)

    metrics = {"policy_loss": 0.0, "value_loss": 0.0, "entropy": 0.0, "approx_kl": 0.0, "clip_frac": 0.0}
    n_batches = 0
    for _epoch in range(args.epochs):
        np.random.shuffle(indices)
        for start in range(0, n, args.minibatch_size):
            mb = indices[start:start + args.minibatch_size]
            mb_t = torch.as_tensor(mb, dtype=torch.long, device=buf.obs.device)
            new_logp, ent, value = model.evaluate_actions(
                buf.obs[mb_t], buf.masks[mb_t], buf.actions[mb_t]
            )
            ratio = torch.exp(new_logp - buf.logprobs[mb_t])
            mb_adv = adv[mb_t]
            surr1 = ratio * mb_adv
            surr2 = torch.clamp(ratio, 1 - args.clip_eps, 1 + args.clip_eps) * mb_adv
            policy_loss = -torch.min(surr1, surr2).mean()
            value_loss = 0.5 * (value - buf.returns[mb_t]).pow(2).mean()
            entropy_loss = -ent.mean()
            loss = policy_loss + args.value_coef * value_loss + args.entropy_coef * entropy_loss

            optimizer.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), args.max_grad_norm)
            optimizer.step()

            with torch.no_grad():
                approx_kl = (buf.logprobs[mb_t] - new_logp).mean().item()
                clip_frac = ((ratio - 1).abs() > args.clip_eps).float().mean().item()
            metrics["policy_loss"] += policy_loss.item()
            metrics["value_loss"] += value_loss.item()
            metrics["entropy"] += (-entropy_loss.item())
            metrics["approx_kl"] += approx_kl
            metrics["clip_frac"] += clip_frac
            n_batches += 1
    for k in metrics:
        metrics[k] /= max(n_batches, 1)
    return metrics


# ──────────────────────────────────────────────────────────────────────────
def main():
    args = parse_args()
    print(f"args: {args}")
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    device = torch.device(args.device)
    opponent = _get_opponent(args.opponent)

    # 학습용 env (상대 자동 진행 wrapper)
    train_env = PPOOpponentEnv(
        env=SplendorEnv(max_steps=400, seed=args.seed),
        opponent_policy=opponent,
        ppo_player=0,
        seed=args.seed,
    )
    obs_dim = train_env.observation_size

    model = ActorCritic(obs_dim, N_ACTIONS).to(device)
    optimizer = optim.Adam(model.parameters(), lr=args.lr, eps=1e-5)
    buf = RolloutBuffer(args.n_steps, obs_dim, N_ACTIONS, device)

    # 초기 reset
    obs, info = train_env.reset(seed=args.seed)
    train_env.last_obs = obs
    train_env.last_info = info

    t0 = time.time()
    ep_returns: list[float] = []

    for update in range(1, args.updates + 1):
        last_value = collect_rollout(train_env, model, buf, device, ep_returns)
        buf.compute_gae(last_value, args.gamma, args.gae_lambda)
        metrics = ppo_update(model, optimizer, buf, args)

        if update % args.log_interval == 0:
            recent = ep_returns[-50:] if ep_returns else [0.0]
            elapsed = time.time() - t0
            print(
                f"[upd {update:4d}] return={np.mean(recent):+.3f} "
                f"(n={len(ep_returns)})  "
                f"pl={metrics['policy_loss']:+.3f} vl={metrics['value_loss']:.3f} "
                f"H={metrics['entropy']:.3f} kl={metrics['approx_kl']:+.4f} "
                f"clip={metrics['clip_frac']:.3f}  "
                f"elapsed={elapsed:.1f}s"
            )

        if update % args.eval_interval == 0:
            policy_fn = make_ppo_policy_fn(model, device, deterministic=False)
            for opp_name, opp_fn in (("random", random_policy), ("greedy", greedy_buy_policy)):
                report = evaluate_vs_policy(
                    learner_policy=policy_fn,
                    opponent_policy=opp_fn,
                    n_episodes=args.eval_episodes,
                    seed=10_000 + update,
                )
                print(
                    f"  eval vs {opp_name:<6s} "
                    f"win={report['win_rate']:.2%} draw={report['draw_rate']:.2%} "
                    f"score={report['mean_learner_score']:.2f}/{report['mean_opponent_score']:.2f} "
                    f"steps={report['mean_steps']:.1f} "
                    f"nobles={report['mean_nobles_gained']:.2f} "
                    f"T3buys={report['mean_tier_buys'][2]:.2f}"
                )

    print(f"\ntotal time: {time.time() - t0:.1f}s")


if __name__ == "__main__":
    main()
