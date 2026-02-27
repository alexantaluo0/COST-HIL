# Hypothesis 1 Ablation Configurations

Override `intervention_scheduler` in your training config for ablation experiments.

## Variants

| Variant | use_value_based_trigger | use_belief_uncertainty | use_auto_intervention | Description |
|---------|-------------------------|-------------------------|------------------------|-------------|
| **heuristic_baseline** | false | false | false | Original heuristic: high uncertainty + low recent_return |
| **theory_value_based** | true | false | false | Optimal stopping: benefit = σ·max(0,V_goal-V) > \|cost\| |
| **theory_full** | true | true | false | Value-based + belief state (EMA of uncertainty) |

## Usage

Merge the desired variant into your config. Example with `train_gym_hil_env.json`:

```bash
# Heuristic baseline (default)
python -m lerobot.rl.actor --config_path train_gym_hil_env.json

# For theory configs, edit train_gym_hil_env.json and set intervention_scheduler
# to one of the variants from configs/hypothesis1_ablation.json
```

Or use `jq` to merge:
```bash
jq -s '.[0] * .[1].theory_value_based' train_gym_hil_env.json configs/hypothesis1_ablation.json > train_theory.json
python -m lerobot.rl.actor --config_path train_theory.json
```

## Key Parameters

- **use_value_based_trigger**: Use Q(s,π(s)) for optimal stopping (benefit > cost)
- **use_belief_uncertainty**: EMA of uncertainty as Bayesian-style posterior
- **use_auto_intervention**: Auto-switch to teleop when suggested (requires env support)
- **intervention_cost**: Negative penalty per intervention step (e.g. -0.01)
