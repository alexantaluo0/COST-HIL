# COST-HIL 论文复线代码

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

## 2. 环境搭建（Windows版）

### 2.1 创建 Conda 环境

```bash
conda create -y -n lerobot python=3.10
conda activate lerobot
```

> **注意**：如果本地电脑禁用了 conda，可能需要特殊处理才能创建环境。

### 2.2 克隆项目

```bash
git clone https://github.com/huggingface/lerobot.git
cd lerobot
```

### 2.3 安装依赖包

```bash
pip install -e ".[hilserl]"
```

#### Windows 常见问题：placo-0.9.18.tar.gz 安装失败

如果在 Windows 上遇到 `placo-0.9.18.tar.gz` 安装失败的问题，解决步骤如下：

1. **安装 Visual Studio Build Tools**（具体步骤可能因环境而异）

2. **使用 conda 安装预编译的 placo**：
```bash
conda install -y -c conda-forge placo=0.9.18
```

3. **重新安装依赖**：
```bash
pip install -e ".[hilserl]"
```

---

## 3. 修改配置文件

官方提供的配置文件在运行时会出现问题，需要进行修改。

**官方配置参考**：[官方配置文件](https://huggingface.co/datasets/lerobot/config_examples/resolve/main/rl/gym_hil/env_config.json)

### 3.1 `gym_hil_env.json` 配置说明

可能需要修改的参数：

- `control_time_s`：回合最大时长（秒）
- `repo_id`：HuggingFace 数据集仓库 ID（格式：`username/dataset_name`）。测试时这种方式不太可行，可以设为 `null`
- `root`：本地数据集保存路径
- `num_episodes_to_record`：要录制的回合数
- `push_to_hub`：是否自动上传到 HuggingFace Hub
- `mode`：模式设置，可选值：`"record"`（录制模式）、`null`（仅运行）

### 3.2 `train_gym_hil_env.json` 配置说明

可能需要修改的参数：

- 数据集路径
- 训练相关参数

---

## 4. 训练与评测流程

### 4.1 录制离线数据集

```bash
cd E:\HIL-SERL\lerobot
python -m lerobot.rl.gym_manipulator --config_path gym_hil_env.json
```

#### 操作说明

等待几秒后会出现仿真画面，操作步骤：

1. 会自动弹出干预窗口
2. 按 **空格键** 切换到人类操作模式

**键盘控制说明**：

| 按键 | 功能 |
|------|------|
| **左 Shift** | 机械臂向上 |
| **右 Shift** | 机械臂向下 |
| **左 Ctrl** | 夹爪打开 |
| **右 Ctrl** | 夹爪关闭 |
| **↑ ↓ ← →** | 控制末端执行器在 x、y 平面内移动 |
| **Enter** | 标记为成功 |
| **Backspace** | 标记为失败 |

抓住方块后，按右 Shift 抬起到一定高度会自动触发成功，结束一回合录制。

---

### 4.2 运行 Learner 进程

```bash
cd E:\HIL-SERL\lerobot
python -m lerobot.rl.learner --config_path train_gym_hil_env.json
```

等待离线数据集加载完毕后，再运行 Actor 进程。

---

### 4.3 运行 Actor 进程

在新的终端中运行：

```bash
cd E:\HIL-SERL\lerobot
python -m lerobot.rl.actor --config_path train_gym_hil_env.json
```

运行后会出现仿真画面，初期机械臂会随机动作。按 **空格键** 进行人类干预操作。

**训练特点**：前期人类干预较多，随着训练进行逐渐减少。

---

### 4.4 测试模型

```bash
cd E:\HIL-SERL\lerobot
python -m lerobot.rl.actor --config_path eval_gym_hil_env.json
```

**评测结果**：在 200 次抓取动作测试中，可达到 **99.5%** 的准确率。

---

## 5. 其他问题

### 5.1 Windows 缺少 Triton 编译器

在 Windows 运行时会出现缺少 Triton 编译器的问题，代码会降级为 Eager 模式，这会影响训练速度。

**建议**：如果在 Linux 上训练，可以继续使用 Triton 以获得更好的性能。

### 5.2 代码不完善问题

复现过程中发现代码有很多不完善的地方，已在复现过程中进行了修改。建议环境搭建好之后，使用修改后的代码版本。

---

## 配置文件示例

项目根目录下包含以下配置文件：

- `gym_hil_env.json`：数据录制配置
- `train_gym_hil_env.json`：训练配置
- `eval_gym_hil_env.json`：评测配置

请根据实际需求修改配置文件中的参数。

---

## 参考资源

- **官方文档**：[HIL-SERL Simulation Guide](https://huggingface.co/docs/lerobot/hilserl_sim)
- **配置示例**：[Official Config Examples](https://huggingface.co/datasets/lerobot/config_examples)
- **飞书文档**：[HIL-SERL 复现文档](https://longcheer.feishu.cn/wiki/IRUrwrrgriFTxvkATGqcv2GKnwh)

---

## 许可证

本项目基于 [Apache 2.0 License](./LICENSE) 开源。
