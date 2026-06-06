## 1. 项目介绍

# COST-HIL
Official implementation of COST-HIL: Cost-Aware Human-in-the-Loop Reinforcement Learning for Real-World Dexterous Manipulation
![Framework](./assets/framework.png)
![Training_Curve](./assets/train_curve.png)

[![Python Version](https://img.shields.io/badge/python-3.9%2B-blue)]
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-orange)]
[![Real Robot Learning](https://img.shields.io/badge/Field-Robot%20RL-green)]
[![MIT License](https://img.shields.io/badge/license-MIT-yellow)]

## Overview
Existing HIL-SERL relies on handcrafted fixed threshold rules for human intervention, uses equal-weight replay for all correction data, and ignores gripper frequent-switch jitter on physical robots, leading to excessive human cost, low sample efficiency and unstable hardware execution.

**COST-HIL treats human intervention as a cost-controllable schedulable resource instead of an external auxiliary signal.** Based on the HIL-SERL backbone (offline demonstration initialization + online SAC fine-tune), we propose three core modules to jointly optimize task success rate, human labor cost and robot execution stability for real-world dexterous grasping.

## Core Technical Improvements (Three Key Modules)
### 1. Cost-Aware Adaptive Intervention Scheduler
Formulate human intervention as a cost-aware optimal stopping problem under POMDP:
1. Calculate policy surprisal to construct instantaneous uncertainty, apply EMA smoothing to get stable belief uncertainty $b_t$;
2. Estimate no-intervention state value via multi-critic conservative evaluation, compute intervention benefit $B_t$;
3. Trigger human takeover only when intervention benefit > predefined per-step human cost, dynamically increase threshold along training to reduce redundant manual operation;
4. Embed human intervention penalty $\lambda_{int}<0$ into original reward: $r_t'=r_t+\lambda_{int}i_t+\lambda_{grip}c_t^{grip}$, endogenous cost optimization into RL objective.

**Effect**: Human intervention rate reduced by 80% compared with HIL-SERL baseline.

### 2. Informativeness & Reliability Joint Reweighted Replay Buffer
Abandon uniform sampling for autonomous and human-corrected samples:
1. **Informativeness**: Measured by absolute TD error, large TD means insufficiently learned high-value samples;
2. **Reliability**: Derived from sigmoid normalized critic Q value to filter low-quality noisy human corrections;
3. Combine two metrics to compute sample priority for prioritized experience replay;
4. Dynamically adjust online/offline sample mixing ratio according to moving average training loss of two data sources.

**Effect**: Improve utilization of high-quality human demonstration and suppress negative influence of invalid intervention data.

### 3. Gripper Cooling & Switching Penalty Regularization
Solve gripper frequent toggle jitter and hardware abrasion in physical grasping:
1. Cooling window constraint: forbid continuous gripper state switching within fixed timestep window K;
2. Add binary switching penalty item $c_t^{grip}=\mathbb{I}[g_t\neq g_{t-1}]$ to augmented reward to penalize meaningless frequent open/close.

**Effect**: Reduce single episode runtime by 46.3% without sacrificing task success rate.

## Experimental Results
Experiment platform: Zhiyuan Elf G2 dexterous robot arm with 2 wrist RGB cameras + 1 head RGB camera, control frequency=10Hz.
|Metric|HIL-SERL|COST-HIL|Relative Improvement|
|:---|:---:|:---:|:---:|
|Success Rate|80.7±2.1%|99.9±0.1%|+19.2%|
|Total Training Time|54 min|21 min|-61.1%|
|Intervention Rate|30%|6%|-80.0%|
|Single Cycle Time|4.23s|2.27s|-46.3%|

- Convergence speed: COST-HIL reaches 90% success rate within 7493 training steps, while HIL-SERL requires 21529 steps;
- After sufficient training, COST-HIL’s human intervention gradually drops near zero, baseline still keeps ~15% intervention ratio.

---

## 2.Environment Setup on Windows

### 2.1 Create Conda Environment

```bash
conda create -y -n lerobot python=3.10
conda activate lerobot
```

> **Note**：Special configuration is required if conda is disabled on your local device.


### 2.2 Clone Repository

```bash
git clone https://github.com/huggingface/lerobot.git
cd lerobot
```

### 2.3 Install Dependencies

```bash
pip install -e ".[hilserl]"
```

#### Fix for placo-0.9.18 Installation Failure on Windows

如果在 Windows 上遇到 `placo-0.9.18.tar.gz` 安装失败的问题，解决步骤如下：

1. **Install Visual Studio Build Tools in advance.**

2. **Install precompiled placo via conda:**：
```bash
conda install -y -c conda-forge placo=0.9.18
```

3. **Reinstall project dependencies:**：
```bash
pip install -e ".[hilserl]"
```

---

## 3. Configuration Modification

Default official configuration files need customized adjustments before running.

**Official reference config:**：[Official reference config](https://huggingface.co/datasets/lerobot/config_examples/resolve/main/rl/gym_hil/env_config.json)

### 3.1 `gym_hil_env.json` Parameters

- `control_time_s`: Max episode duration in seconds
- `repo_id`: Dataset repository ID on HuggingFace, set to null for local test
- `root`: Local storage path for collected datasets
- `num_episodes_to_record`: Total episodes to be recorded
- `push_to_hub`: Toggle automatic dataset upload to HuggingFace Hub
- `mode`: Available options: "record" for data collection, null for regular running


### 3.2 `train_gym_hil_env.json` Parameters

Customize dataset path and hyperparameters for training：

Training & Evaluation Pipeline

---

## 4. Collect Offline Demonstration Dataset

### 4.1 Collect Offline Demonstration Dataset

```bash
cd E:\HIL-SERL\lerobot
python -m lerobot.rl.gym_manipulator --config_path gym_hil_env.json
```

#### Operation Guide

The simulation window pops up after initialization：

1. Human intervention panel loads automatically
2. Press **Space** to switch into manual control mode

**Keyboard Mapping**：

| Key | Function |
|------|------|
| **Left Shift** | Move arm upward |
| **Right Shift** | Move arm downward |
| **Left Ctrl** | Open gripper |
| **Right Ctrl** | Close gripper |
| **↑ ↓ ← →** | Move end-effector on X-Y plane |
| **Enter** | current episode as success |
| **Backspace** | current episode as failure |


An episode automatically succeeds when lifting the block to target height.

---

### 4.2 Start Learner Process

```bash
cd E:\HIL-SERL\lerobot
python -m lerobot.rl.learner --config_path train_gym_hil_env.json
```

Launch the Actor process after dataset loading completes.

---

### 4.3 Start Actor Process (New Terminal)


```bash
cd E:\HIL-SERL\lerobot
python -m lerobot.rl.actor --config_path train_gym_hil_env.json
```
Random actions appear in early training stage; press Space to enable manual correction. Human intervention frequency decreases gradually as policy converges.

---

### 4.4 Evaluation

```bash
cd E:\HIL-SERL\lerobot
python -m lerobot.rl.actor --config_path eval_gym_hil_env.json
```

**Evaluation Result*：**99.5%** grasping success rate over **200** test episodes.

---

## 5. Common Troubleshooting

### 5.1 Missing Triton Compiler on Windows

Triton is unavailable under Windows, the framework automatically switches to Eager execution with reduced training speed. Use Linux environment for full Triton acceleration.

### 5.2 Original Code Defects

Multiple bugs in upstream source code have been fixed in this repository version. Use our modified code after environment setup.

---

## Config File List

- `gym_hil_env.json`：Data collection configuration
- `train_gym_hil_env.json`：Training hyperparameter configuration
- `eval_gym_hil_env.json`：Evaluation configuration

Adjust corresponding parameters according to your experimental requirements.

---

## Reference Links

- **Official HIL-SERL Doc**：[HIL-SERL Simulation Guide](https://huggingface.co/docs/lerobot/hilserl_sim)
- **Official Config Examples**：[Official Config Examples](https://huggingface.co/datasets/lerobot/config_examples)

---

## License
This project is released under Apache 2.0 License.
