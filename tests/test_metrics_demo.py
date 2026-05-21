"""MetricsWrapper / evaluate_vs_policy 동작 데모.

랜덤 vs 랜덤, 그리디 vs 랜덤 두 시나리오로 지표가 합리적으로 나오는지 확인.
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from splendor import evaluate_vs_policy, greedy_buy_policy, random_policy  # noqa: E402


def _print_report(title: str, report: dict):
    print(f"\n=== {title} ===")
    print(f"n_episodes        : {report['n_episodes']}")
    print(
        f"win/draw/loss     : {report['win_rate']:.2%} / "
        f"{report['draw_rate']:.2%} / {report['loss_rate']:.2%}"
    )
    print(f"mean steps        : {report['mean_steps']:.1f}")
    print(
        f"mean score        : learner={report['mean_learner_score']:.2f}  "
        f"opp={report['mean_opponent_score']:.2f}"
    )
    print(f"mean nobles       : {report['mean_nobles_gained']:.2f}")
    print(f"mean gold used    : {report['mean_gold_used']:.2f}")
    print(f"illegal attempts  : {report['total_illegal_attempts']}")
    print("action fractions  :")
    for k, v in report["mean_action_fractions"].items():
        print(f"    {k:<14s} {v:.3f}")
    print(
        f"mean tier buys    : T1={report['mean_tier_buys'][0]:.2f}  "
        f"T2={report['mean_tier_buys'][1]:.2f}  T3={report['mean_tier_buys'][2]:.2f}"
    )


def main():
    rnd_vs_rnd = evaluate_vs_policy(
        learner_policy=random_policy,
        opponent_policy=random_policy,
        n_episodes=80,
        seed=42,
    )
    _print_report("random vs random (learner=random)", rnd_vs_rnd)

    greedy_vs_rnd = evaluate_vs_policy(
        learner_policy=greedy_buy_policy,
        opponent_policy=random_policy,
        n_episodes=80,
        seed=42,
    )
    _print_report("greedy-buy vs random (learner=greedy)", greedy_vs_rnd)


if __name__ == "__main__":
    main()
