# LIBERO 单任务模仿学习实验报告

> 实验日期：2026-09-12  
> 实验任务：拾取小模具上的黑色碗，并将其放置到盘子上  
> LIBERO 任务名：`pick_up_the_black_bowl_on_the_ramekin_and_place_it_on_the_plate`

## 摘要

本实验基于 LIBERO-Spatial 的单个机械臂操作任务，依次实现并比较了 State BC、双相机 Vision BC、RNN BC、Transformer BC 和 ACT。实验重点包括 HDF5 数据的时序对齐、双相机视觉编码、Spatial Softmax、序列建模、动作分块、CVAE 潜变量以及 temporal ensemble。

为避免不同 rollout 配置造成不公平比较，本实验最终使用同一任务、同一组 10 个初始状态、相同的 10 步环境稳定过程和 220 步最大控制长度进行统一闭环评测。现有 checkpoint 的结果为：State BC 成功率 100%，RNN BC 成功率 90%，Transformer BC 成功率 0%，ACT 成功率 90%。ACT 在成功轨迹上的平均完成步数最低，为 92.7 步。实验同时表明，离线验证误差并不能可靠代表闭环控制成功率。

## 1. 实验目标

本实验的目标是完成一个从基础行为克隆到动作分块 Transformer 的单任务模仿学习流程，具体包括：

1. 解析 LIBERO HDF5 演示数据并建立正确的 observation-action 对齐关系。
2. 使用机器人本体状态建立 State BC 基线。
3. 加入主相机和腕相机，研究视觉输入和空间特征提取。
4. 使用 LSTM 和 Transformer 对历史观测序列建模。
5. 实现带 CVAE 风格编码器、动作 query 和 temporal ensemble 的 ACT。
6. 在相同初始状态上进行闭环评测，并保存可复查的视频证据。

## 2. 实验环境与数据

### 2.1 任务与数据划分

实验使用 LIBERO-Spatial 中的以下任务：

```text
pick up the black bowl on the ramekin and place it on the plate
```

数据集包含 50 条专家演示。固定随机种子为 0，以演示轨迹为单位随机划分：

| 集合 | 演示数量 | 比例 |
|---|---:|---:|
| 训练集 | 40 | 80% |
| 验证集 | 10 | 20% |

按完整演示划分能够避免同一轨迹的相邻帧同时进入训练集和验证集。

### 2.2 输入与输出

机器人状态为 15 维：

| 状态组成 | 维度 |
|---|---:|
| 末端位置 | 3 |
| 末端姿态（axis-angle） | 3 |
| 机械臂关节位置 | 7 |
| 夹爪位置 | 2 |
| 合计 | 15 |

视觉模型额外使用两路 128×128 RGB 图像：

- `agentview`：场景主相机；
- `eye_in_hand`：腕部相机。

动作是 7 维连续控制量：

```text
[dx, dy, dz, dRx, dRy, dRz, gripper]
```

其中夹爪专家标签为 -1 或 +1，闭环执行前重新二值化。

### 2.3 HDF5 时序对齐

数据生成过程先执行 `action[t]`，再保存执行后的 `obs[t]`。因此，HDF5 中的 `obs[t]` 近似对应执行完 `action[t]` 后的状态，而行为克隆需要学习当前状态到下一动作的映射：

```text
obs[t] -> action[t + 1]
```

所以单步数据使用：

```python
observations = observations[:-1]
actions = actions[1:]
```

RNN 和 Transformer 使用：

```python
observation_seq = observations[start:start + L]
action_seq = actions[start + 1:start + L + 1]
```

ACT 在时刻 `t` 的监督目标为：

```python
action_chunk = actions[t + 1:t + 1 + K]
```

轨迹末尾不足 `K` 步的部分用零填充，并通过 `is_pad` 在编码器与损失函数中屏蔽。

### 2.4 归一化

所有方法均使用训练集统计量对 15 维状态做标准化：

```text
normalized_state = (state - state_mean) / state_std
```

ACT 还对动作做标准化：

```text
normalized_action = (action - action_mean) / action_std
```

因此 ACT rollout 必须先执行反归一化，再进行 temporal ensemble、动作裁剪和夹爪二值化。

## 3. 方法

### 3.1 State BC

State BC 只输入 15 维机器人状态，使用两层 128 维 MLP，输出 7 维动作并经过 `Tanh`。该模型是最小基线，不直接观察物体图像。

```text
15 -> 128 -> 128 -> 7
```

训练配置：batch size 256、100 epochs、学习率 1e-3、MSE 损失。

### 3.2 Vision BC

Vision BC 使用两个独立 CNN 分别编码主相机和腕相机，同时使用 MLP 编码机器人状态。三路特征拼接后预测单步动作。

CNN 最后一层没有使用全局平均池化，而是使用 Spatial Softmax。对于每张特征图，Spatial Softmax 将二维响应转换为期望坐标 `(x, y)`，从而保留物体位置相关信息。128 张特征图最终产生 256 维空间坐标特征。

训练配置：batch size 64、50 epochs、学习率 3e-4、MSE 损失。

本次统一评测时工作区中缺少 `vision_bc_policy.pt`，因此该方法不进入最终量化比较。报告保留其方法设计和开发阶段的定性结论，但不虚构成功率。

### 3.3 RNN BC

RNN BC 对每个时间步提取双相机和状态特征，然后送入单层 LSTM：

```text
双相机 + 状态 -> 每帧 640 维特征 -> LSTM(hidden=128) -> 动作
```

序列长度为 10。训练时对序列中的每个时间步进行动作监督；rollout 时逐帧输入并携带隐藏状态，每 10 步重置一次隐藏状态以贴近训练长度。

训练配置：batch size 64、50 epochs、学习率 3e-4、MSE 损失、梯度裁剪 1.0。

### 3.4 Transformer BC

Transformer BC 将每个时间步的双相机和状态特征拼接为 640 维，再投影到 `d_model=128`。模型使用 4 个注意力头、1 层 Transformer Encoder、长度 10 的可学习位置编码以及因果注意力掩码。

每个时间步对应一个融合 token：

```text
[agent feature, wrist feature, state feature]
                    |
                  concat
                    |
              one temporal token
```

训练配置：batch size 64、50 epochs、学习率 3e-4、MSE 损失、梯度裁剪 1.0。

### 3.5 ACT

ACT 一次预测未来 `K=10` 个动作。模型分为三个主要部分：

1. 风格编码器：将 `[CLS]`、当前状态和专家 action chunk 编码为 CVAE 后验参数 `mu` 和 `logvar`；
2. 观察编码器：将主相机、腕相机、状态和潜变量组成 4 个 memory token；
3. 动作解码器：使用 10 个可学习 action query，并行解码未来 10 个动作。

主要配置如下：

| 参数 | 数值 |
|---|---:|
| `d_model` | 256 |
| 注意力头数 | 4 |
| 潜变量维度 | 32 |
| Style Encoder 层数 | 2 |
| Memory Encoder 层数 | 2 |
| Decoder 层数 | 2 |
| Chunk size | 10 |
| Batch size | 8 |
| 学习率 | 1e-4 |
| KL 权重 | 1.0 |

训练损失为带 padding mask 的 L1 重建损失与 KL 散度之和：

```text
loss = masked_L1 + KL_WEIGHT * KL
```

推理时无法获得未来专家动作，因此使用 `z=0`。每个控制时刻重新预测一个 action chunk，并对所有指向当前绝对时刻的重叠预测进行指数加权平均。

## 4. 统一闭环评测协议

为保证公平性，所有可用 checkpoint 使用同一个评测脚本：

| 配置项 | 数值 |
|---|---:|
| 初始状态 | `init_state[0:10]` |
| Episode 数 | 10 |
| 最大控制步数 | 220 |
| 环境稳定步数 | 10 |
| 随机种子 | 0 |
| 策略输入分辨率 | 每相机 128×128 |
| 视频录制分辨率 | 每相机 512×512 |
| ACT temporal ensemble 系数 | 0.01 |
| 成功判定 | LIBERO 环境返回 `done=True` |

每个 episode 都执行：

```python
obs = env.reset()
obs = env.set_init_state(init_states[episode])
```

这避免了只在一个初始状态上重复测试而得到虚高成功率。

## 5. 实验结果

![模型统一评测结果](figures/model_comparison.png)

### 5.1 汇总结果

| 方法 | 参数量 | Checkpoint 离线指标 | 成功次数 | 成功率 | 成功轨迹平均步数 |
|---|---:|---:|---:|---:|---:|
| State BC | 19,463 | MSE 0.043295 | 10/10 | **100%** | 96.6 |
| Vision BC | 406,023 | checkpoint 缺失 | 未评测 | 未评测 | 未评测 |
| RNN BC | 636,295 | MSE 0.038599 | 9/10 | **90%** | 102.2 |
| Transformer BC | 441,351 | MSE 0.041150 | 0/10 | **0%** | — |
| ACT | 4,054,983 | prior L1 0.437186 | 9/10 | **90%** | **92.7** |

注意：前三个 checkpoint 的离线指标是未标准化动作空间中的 MSE；ACT 指标是标准化动作空间中的 prior L1，二者不能直接横向比较。

### 5.2 各初始状态结果

| Init state | State BC | RNN BC | Transformer BC | ACT |
|---:|---:|---:|---:|---:|
| 0 | 成功，88 步 | 成功，98 步 | 失败 | 成功，90 步 |
| 1 | 成功，97 步 | 成功，111 步 | 失败 | 成功，94 步 |
| 2 | 成功，87 步 | 成功，102 步 | 失败 | 成功，93 步 |
| 3 | 成功，96 步 | 成功，102 步 | 失败 | 失败 |
| 4 | 成功，103 步 | 成功，103 步 | 失败 | 成功，92 步 |
| 5 | 成功，87 步 | 失败 | 失败 | 成功，95 步 |
| 6 | 成功，93 步 | 成功，100 步 | 失败 | 成功，92 步 |
| 7 | 成功，107 步 | 成功，102 步 | 失败 | 成功，95 步 |
| 8 | 成功，114 步 | 成功，103 步 | 失败 | 成功，94 步 |
| 9 | 成功，94 步 | 成功，99 步 | 失败 | 成功，89 步 |

原始机器可读结果见 [evaluation/results.json](evaluation/results.json)，全部 episode 视频见 [evaluation/videos](evaluation/videos/)。

### 5.3 不确定性

本次每种方法只有 10 个 episode。使用 95% Wilson 区间时：

| 观测结果 | 95% 区间（约） |
|---|---:|
| 10/10 | 72.2%–100% |
| 9/10 | 59.6%–98.2% |
| 0/10 | 0%–27.8% |

因此当前结果适合作为单任务功能验证和模型行为比较，但不足以支持非常精细的统计排名。

## 6. 结果分析

### 6.1 State BC 为什么表现最好

State BC 虽然不直接观察物体，但该单任务的场景变化较小，专家轨迹也具有较稳定的阶段结构。末端位姿、关节状态和夹爪状态能够强烈反映当前处于接近、抓取、抬起还是放置阶段。因此，小型状态模型可以形成有效的闭环控制器。

该结果不能推出视觉输入没有价值。它更可能说明当前单任务分布较窄，而状态与动作之间存在很强的相关性。换到物体位姿变化更大、遮挡更明显或多任务场景时，纯状态策略预计会明显受限。

### 6.2 RNN 的优势与失败

RNN 获得最低的传统验证 MSE，并在 10 个初始状态上成功 9 次。LSTM 隐藏状态能够整合近期运动历史，有利于判断任务阶段。但滚动执行时，模型接收到的是自己产生的状态分布，而不是训练集中的专家状态，误差仍会累积。初始状态 5 的失败说明序列记忆本身不能消除分布偏移。

### 6.3 Transformer 的离线与闭环差距

Transformer 的验证 MSE 为 0.041150，与 State BC 和 RNN 接近，但闭环成功率为 0%。这说明逐帧平均误差较小，并不保证关键阶段动作正确。

可能原因包括：

- 训练使用长度 10 的专家序列，rollout 中历史由模型行为产生；
- 初始时刻历史长度短于训练长度；
- 自注意力会同时利用多个历史 token，历史中的偏差可能被持续传播；
- 抓取和放置只允许很小的局部误差，普通 MSE 对关键动作阶段不够敏感。

### 6.4 ACT 的表现

ACT 成功率为 90%，与 RNN 相同，但成功轨迹平均只需 92.7 步，是所有成功方法中最低。动作分块为策略提供了短期计划，temporal ensemble 又能平滑不同查询时刻对同一动作的预测，因此轨迹执行更连贯。

ACT 的参数量约为 4.05M，显著高于其他模型。在只有 40 条训练演示的条件下，模型存在过拟合风险。初始状态 3 的失败说明更大的网络和动作分块并没有完全解决数据覆盖问题。

### 6.5 开发过程中的消融观察

开发过程中得到以下定性观察：

- 使用全局平均池化会丢失空间位置，替换为 Spatial Softmax 后视觉策略明显改善；
- 保持模型其他部分一致时，移除腕相机会降低表现，说明近距离抓取信息有价值；
- 将 RNN 隐藏维度从较大的设置降到 128 后，闭环成功率提高，表明小数据条件下更小的模型更容易优化和泛化；
- Transformer 上下文越长不一定越好，因为闭环历史包含模型自身误差；
- ACT 必须对预测动作反归一化，否则 temporal ensemble 和动作裁剪会在错误的数值空间中进行。

这些观察目前没有完整的多随机种子数值记录，因此只作为定性结论。

## 7. 局限性

1. 实验只覆盖一个 LIBERO-Spatial 任务，不能代表多任务泛化能力。
2. 每种方法只评测 10 个初始状态和一个随机种子。
3. Vision BC checkpoint 未归档，缺少最终统一评测数据。
4. 旧版 State/RNN/Transformer checkpoint 没有保存训练数据路径与 demo 划分；统一评测按当前单任务实验上下文加载。
5. 当前没有持久化每个 epoch 的训练曲线，无法完整分析收敛与过拟合过程。
6. 各模型的参数量、训练轮数和目标函数不同，本实验比较的是最终系统行为，而不是严格的等算力模型比较。

## 8. 结论

本实验完成了 LIBERO 单任务中从状态行为克隆、双相机视觉策略、时序策略到 ACT 的完整实现。统一评测显示：

- State BC 在当前窄分布任务上以最小参数量取得 100% 成功率；
- RNN 能有效利用历史信息，达到 90% 成功率；
- Transformer 的离线误差与其他模型接近，但闭环完全失败，说明离线指标不足以评价机器人控制策略；
- ACT 达到 90% 成功率，并具有最短的成功轨迹平均步数，体现出动作分块和 temporal ensemble 的价值。

最重要的工程结论是：机器人模仿学习必须以严格、可复现的闭环 rollout 为最终评价依据。数据时间对齐、环境重置、归一化一致性和相机输入正确性，都会比单纯增大网络更直接地影响成功率。

## 9. 复现实验

训练最终 ACT：

```bash
python my_bc/ACT/train_ACT_bc.py
```

统一评测并保存视频：

```bash
python my_bc/evaluate_single_task.py \
  --episodes 10 \
  --max-steps 220 \
  --record-resolution 512 \
  --save-videos \
  --output-dir reports/evaluation
```

主要产物：

- [统一评测 JSON](evaluation/results.json)
- [结果图](figures/model_comparison.png)
- [20 秒汇总视频](single_task_video_report.mp4)
- [同一初始状态四模型对比](comparison_episode_00.mp4)
- [全部 episode 视频](evaluation/videos/)
- [视频口播稿](video_report_script.md)

## 参考资料

1. LIBERO: Benchmarking Knowledge Transfer for Lifelong Robot Learning, arXiv:2306.03310.
2. Learning Fine-Grained Bimanual Manipulation with Low-Cost Hardware, Robotics: Science and Systems, 2023.
