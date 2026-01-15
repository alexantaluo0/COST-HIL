# HIL-SERL 仿真环境复现指南 (Windows版)

## 1. 项目介绍

本项目复现了 LeRobot 集成的 HIL-SERL 仿真环境（抓方块任务）。与论文开源代码相比，主要区别如下：

1. **省去二分类奖励分类器训练步骤**：改为使用环境自带设置，当机械臂抓取方块并抬起到一定高度时自动判定成功，或手动按回车键标记为成功。

2. **框架改写**：使用 PyTorch 框架重新实现，原版使用 JAX 框架。

3. **自动图像裁剪**：省去手动裁剪腕部相机尺寸的步骤，改为自动裁剪成统一尺寸。

**参考文档**：[HIL-SERL Simulation Guide](https://huggingface.co/docs/lerobot/hilserl_sim)

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

1. 用鼠标点击一下仿真界面
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

### 4.4 人工干预标准操作规程（SOP）

在实际实验中，人为干预的时机对收敛性能有显著影响。为确保训练效果，建议遵循以下标准操作流程：

1. **初期全面干预**  
   在前 3-5 个阶段进行全面干预，每次都完成任务，提供稳定的初期指导。

2. **严重偏差后不干预探索**  
   一旦观察到严重偏差（例如机器人末端执行器朝无关方向探索），不要干预，让机器人探索直至该回合超时。

3. **反复严重偏差后应尽早干预**  
   如果在同一位置/状态下两次观察到严重偏差，则下次出现偏差时，应在偏差首次发生的最早时间点进行干预。

4. **连续失败后完全接管**  
   如果机器人连续五次未能完成任务，下一次操作由人工从头开始完全介入并完成任务，提供完整的人工演示。

5. **驯化抓手动作**  
   当夹爪反复关闭、抓住目标后又放下时，夹爪移动到目标位置后立刻人工接管，重点驯化抓手动作，使其一次性抓住目标。

> **注意**：虽然严格遵守 SOP 对操作人员来说不切实际，但它仍可作为高效在线训练的宝贵指导方针。

---

### 4.5 测试模型

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
