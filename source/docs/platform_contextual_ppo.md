# 平台条件报价PPO

原平台actor的Tanh隐藏层饱和，不同日期的动作概率几乎相同。新流程按公开状态及报价属性对每项候选方案共享评分，使用训练集标准化、LayerNorm与SiLU；低折扣课程只在训练启用，评估时开放全部55项报价。

`platform_features.py`提供55×88特征，包含公开参考出清、弃电与输电利用率、原任务、报价让利及参考电价下的潜在空间。没有用户私有偏好、用户真实响应、候选实际利润或候选出清标签。`platform_policy.py`实现共享报价评分，`platform_learning.py`提供观测缓存与固定单位利润反馈。

用户新增显式私有参与条件：账单加自身延迟迁移损失不能高于原合同。用户PPO仍在多个可接受合同中学习选择，未硬编码最优方案。该条件保证原合同是实际可用的外部选项，不能将其计作学习得到的IR保证。

先固定三个已有用户模型训练平台，再恢复四主体联合训练：

```bash
python scripts/train_platform_contextual.py --seed 0 --users initial_users --steps 16384 --eval-interval 2048 --joint-cycles 8 --output results/platform-contextual
python scripts/run_coordination.py --days 32 --checkpoints results/platform-contextual/checkpoints --output results/platform-comparison
```

`initial_users`包含三个已有合同评分PPO文件`user_0.zip`、`user_1.zip`、`user_2.zip`。本轮对应上一版81920步用户权重；函数启用参与条件时保留权重与优化器。平台只按验证集平均自身利润选checkpoint，收益相同保留训练更充分的版本，然后进行固定8轮联合更新。

训练场景0–63，验证9000–9015，原测试10000–10031；设置固定后另外在20000–20031检查新场景。用户私有参与约束、平台利润目标与系统成本目标分别记录。主体利润提升可能伴随系统成本节约减少，不能把平台逐利直接等同于系统最优。

环境RNG在首次封装前显式初始化，避免每阶段重建VecEnv时丢失pending seed。回归测试覆盖相同种子下场景序列、四个策略参数和更新记录复现。
