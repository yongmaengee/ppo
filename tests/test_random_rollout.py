"""랜덤 에이전트 두 명으로 SplendorEnv 가 정상 종결되는지 확인."""
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from splendor import SplendorEnv  # noqa: E402


def random_policy(mask: np.ndarray, rng: np.random.Generator) -> int:
    legal = np.flatnonzero(mask)
    return int(rng.choice(legal))


def play_one(seed: int, max_steps: int = 400, reward_mode: str = "sparse"):
    env = SplendorEnv(reward_mode=reward_mode, max_steps=max_steps, seed=seed)
    obs, info = env.reset(seed=seed)
    rng = np.random.default_rng(seed + 9999)

    total_steps = 0
    rewards = [0.0, 0.0]
    while True:
        actor = info["current_player"]
        mask = info["action_mask"]
        assert mask.any(), "마스크가 비어있으면 안 됨 (PASS 가 보장돼야 함)"
        a = random_policy(mask, rng)
        obs, r, terminated, truncated, info = env.step(a)
        last_actor = info["last_actor"]
        rewards[last_actor] += r
        total_steps += 1
        if terminated or truncated:
            return {
                "steps": total_steps,
                "scores": info["scores"],
                "terminated": terminated,
                "truncated": truncated,
                "rewards": tuple(rewards),
                "final_rewards": info.get("final_rewards"),
            }


def main():
    n_games = 30
    terminated_count = 0
    truncated_count = 0
    total_steps = 0
    score_totals = [0, 0]
    for s in range(n_games):
        result = play_one(seed=s)
        total_steps += result["steps"]
        if result["terminated"]:
            terminated_count += 1
        if result["truncated"]:
            truncated_count += 1
        score_totals[0] += result["scores"][0]
        score_totals[1] += result["scores"][1]
        print(
            f"game {s:02d}: steps={result['steps']:3d} "
            f"scores={result['scores']} "
            f"final_rewards={result['final_rewards']} "
            f"terminated={result['terminated']} truncated={result['truncated']}"
        )

    print("─" * 60)
    print(f"평균 step 수      : {total_steps / n_games:.1f}")
    print(f"정상 종결         : {terminated_count}/{n_games}")
    print(f"truncated         : {truncated_count}/{n_games}")
    print(f"평균 점수 (P0/P1) : {score_totals[0] / n_games:.2f} / {score_totals[1] / n_games:.2f}")


if __name__ == "__main__":
    main()
