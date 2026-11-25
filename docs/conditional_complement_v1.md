# 双向条件互补学习：固定表示的第一版

本文件描述待验证候选，不代表已有改进结果。代码分支为 `codex/conditional-complement-v1`，结果独立保存至 `outputs/conditional_complement_v1/`。

## 来源与边界

从代码 `2b6690ac8600732b98365e91f6fb7b464e33912f` 建立隔离工作树。来源锁固定为结果提交 `b6ca40cfb1abef54c669bdcf932ce95bdd272a66` 中的 `fixed_transition_final_test_v1/checkpoint_lock.json`，其规范化 SHA256 为 `41e039e20a6408eea0457e50fd81878bd94c763a6873a375c52472cb030894ca`。权重和验证预测继续从该锁绑定的 `5f3975aeecd41c8c1150b97f88813b7ea703e54d` 获取；不会因分支移动而更换来源。

| 主干 | 原C | Frozen Single | Joint Single | Ordered起点 |
| --- | ---: | ---: | ---: | ---: |
| 42 | 48 | 94 | 59 | 191 |
| 43 | 188 | 50 | 25 | 97 |
| 44 | 167 | 150 | 131 | 123 |

轮次用于交叉检查，权重哈希与 N0/C 来源绑定为权威。运行中仍调用原 `run_ordered_transition_readout.load_source`，并与归档恢复的 C 参数摘要比较。锁定阶段核对12份验证预测，不加载历史测试预测。开发只允许加载 train/val 图；原图加载器的测试隔离保留。

保留冻结图编码器、跨层表示、Flow查询池、Ordered读出、Flow/Packet投影和固定 `packet + residual`。仅训练原32维融合门控、分类头、新增修正头。源 Ordered 单独冻结保存。缓存是注入后、融合前的32维投影，包含样本ID、标签、69维原门控输入和源模型 log probabilities；绑定 split、清单、源权重、代码、环境、批量布局及载荷哈希。

此前 True/Shuffle 与 beta 门控未通过门槛的结论不变；本机制不是图上的双向消息传递。教师输出不解释为校准置信度，条件残差不解释为已证明的独有或因果信息。

## 公共准备与五组训练

两个32→8线性教师分别在原投影上以交叉熵训练；两个32→32仿射条件预测器在无参数 LayerNorm 表示上以双向 MSE 训练。四个模块共同训练固定20轮，batch512、Adam lr=.001、weight_decay=0，锁定最终轮，不按验证分数选探针。4,220次公共更新另计。训练后全部冻结，输出 train/val 教师八类指标、条件 MSE、预测训练均值的 MSE、分歧子组和逐样本诊断。没有可修复验证分歧时终止候选训练并保留证据。

| 内部组名 | 简称 | 修正结构 | 额外训练约束 |
| --- | --- | --- | --- |
| cc_t0 | T0 | 无 | 无 |
| cc_t1 | T1 | 64→40→32 MLP | 无 |
| cc_t2 | T2 | 双向条件残差、rank8乘性交互、72→32→32 | 无 |
| cc_t3 | T3 | 同T1、同初始化 | SD + keep |
| cc_t4 | T4 | 同T2、同初始化 | SD + keep |

两个修正头最后一层权重和偏置全零。MLP可训练参数3,912，交互头3,904；T2/T4另外使用2,112个冻结条件预测器参数。两个教师共528个参数，仅准备和监督使用，推理不需要。源门控与分类头共2,984个可训练参数。报告分别列出完整推理参数、可训练参数、准备成本及新增仿射 MAC 估算，不以虚设参数配平。

最终八分类 NLL；T3/T4另加 `0.1*SD + 0.1*keep`。SD温度2，仅一个教师正确时对正确教师进行KL蒸馏，除以整个batch样本数；keep仅对源 Ordered 已正确的训练样本保护真实类 log probability，同样除以整个batch。教师概率、正确掩码和源概率均detach。标签不进入模型forward。

主干seed42；新模块初始化seed42，由独立命名CPU RNG流初始化，不改变全局CPU/CUDA随机流。候选采样seed42；公共准备采样使用独立固定seed42020。每个独立主干均重新准备教师与预测器。五组200轮，每组每轮211次更新，总211,000次候选更新；gate/head lr=.0001、新模块lr=.001，Adam betas=(.9,.999)、eps=1e-8、weight_decay=0。不解冻投影，不搜索rank、损失系数或学习率。

所有原图和缓存验证均batch16，没有扩大验证batch。每轮全量验证，包含epoch0，Macro-F1最大、同分最早。30/100/150/200只是同一轨迹预算窗口，200轮为主结论。训练账本保存每项损失、每模块平均梯度范数和每轮首批各未加权损失的梯度范数。

## 恢复、验证与晋级

沿用已有 `PairedSampler`、RNG保存及原子 `torch.save`。每个batch的全部组别完成后才提交事务，保存所有模型、Adam、随机状态、采样顺序/游标、累计更新、损失计数、历史与最佳选择。半批失败后进程退出，重启从最后完整事务恢复；已经提交的更新不重复，未提交的部分重做，Adam不重置。`best/last/epochs.jsonl` 可从恢复状态重建；epoch0另存。恢复绑定不一致时直接报错，不覆盖失败目录。

CPU机制测试覆盖零输出、实际非零Ordered残差下的epoch0一致、冻结梯度与buffer、分歧掩码/空集、温度和归一化、逐步建立上游梯度、独立优化器、半批故障及恢复完全一致、原图合成回放、阶段门槛和上传失败处理。正式服务器还必须对全部13,446条验证流核对来源与epoch0，再进行真实特征梯度探测；最终最佳权重通过原图验证回放。

在T2/T4中按验证Macro-F1选候选，同分T2。晋级必须同时通过：候选−T0≥.10pp；T2−T1或T4−T3≥.05pp；高于源Ordered与Frozen Single；四个重点类别相对T0每类F1下降不超过.30pp；两个仅单分支正确子组的破坏率不高于T0。空组标N/A，仅检查非空组。使用未舍入值，门槛不是显著性检验。

未通过则上传并停止本版。若T3领先而结构路线未过门槛，保留普通MLP协同目标的证据，不将其称作条件交互成功；原计划没有给T3单独数值晋级规则，因此不自行发明规则自动扩展该路线。

通过后自动按锁定方案完成43、44：胜出T2运行T0/T1/T2；胜出T4运行完整五组。不因43负向而跳过44。复验要求候选−T0三个主干均正且均值≥.10pp，对应MLP平均≥.05pp且43/44均正，并检查类别和分歧破坏限制。报告三主干及新增43/44，配对差在主干内计算，样本标准差n−1。

复验通过后，启动器先发布9个最终候选/匹配对照的检查点锁，再原图回放验证，最后固定测试。测试阶段0次优化器更新，全部13,448条流；共享每批冻结图特征。旧 Ordered、Frozen Single、C 的同权重历史测试预测逐样本复核。测试有历史使用，必须披露；不据测试重选模型或调参。

## 服务器运行

保持现有 `wx` 环境，不升级依赖。第一次运行时，先把下面的 `xgnid_revision` 设为交付的完整代码提交。工作树已有且提交不同则停止，保留旧工作树。

```bash
conda activate wx
xgnid_repo='/home/nzxn/下载/WX/load/XG-NID'
xgnid_code="$(dirname "$xgnid_repo")/XG-NID-conditional-complement-v1"
cd "$xgnid_repo" || exit 1
git fetch --no-tags origin codex/conditional-complement-v1 || exit 1
xgnid_revision="$(git rev-parse origin/codex/conditional-complement-v1)"
if [ -e "$xgnid_code/.git" ]; then
  [ "$(git -C "$xgnid_code" rev-parse HEAD)" = "$xgnid_revision" ] || exit 1
  [ -z "$(git -C "$xgnid_code" status --porcelain --untracked-files=no)" ] || exit 1
else
  git worktree add --detach "$xgnid_code" "$xgnid_revision" || exit 1
fi
mkdir -p "$xgnid_repo/tmp" "$xgnid_repo/outputs/conditional_complement_v1"
nohup python -u -B "$xgnid_code/scripts/launch_conditional_complement.py" \
  --repo "$xgnid_repo" --stage screen --backbone-seed 42 --resume auto \
  --publish-on-exit \
  >> "$xgnid_repo/tmp/conditional_complement_v1.launcher.log" 2>&1 < /dev/null &
echo $! > "$xgnid_repo/outputs/conditional_complement_v1/launcher.pid"
tail -F "$xgnid_repo/outputs/conditional_complement_v1/training.log"
```

`--resume auto` 是同一个完整启动命令的真正续训。启动器和运行器均有独立文件锁，禁止同目录并发训练；已完成阶段校验后复用。不要删除恢复状态后重跑择高。

查看阶段/上传日志：

```bash
tail -F "$xgnid_repo/tmp/conditional_complement_v1.launcher.log"
cat "$xgnid_repo/outputs/conditional_complement_v1/status.json"
cat "$xgnid_repo/outputs/conditional_complement_v1/training_exit.json"
```

仅服务器来源/准备/梯度/epoch0探测（仍包含公共20轮准备，候选200轮尚不启动）可加 `--probe-only`；只锁定来源、不训练可加 `--lock-only`。默认完整队列自动按门槛继续，用户无须手动串行启动。

正常结束、可捕获异常、SIGINT/SIGTERM均触发发布；SIGKILL或断电时只能依靠上一事务，在重启后继续。每阶段发布后，从精确Git提交读取Git/LFS真实载荷，独立复算指标及配对计数；失败则停止后续阶段。上传进程的日志先写输出目录外，关闭后再发布日志/退出码收据，防止边写边打包。

现有 `sync_server_outputs.py --only` 接受任意合法的outputs直属目录，没有静态白名单；已验证该目录合法，不需要扩大发布范围。仅发布本轮输出至 `codex/server-outputs`，权重、恢复状态和大缓存沿用LFS。失败现场和旧结果保留。训练退出码与上传/队列错误分别记录。

独立重试发布（训练进程停止后）：

```bash
python -B "$xgnid_code/scripts/sync_server_outputs.py" publish \
  --repo "$xgnid_repo" --only conditional_complement_v1
```

精确结果提交的独立复算：

```bash
python -B "$xgnid_code/scripts/audit_conditional_complement_archive.py" \
  --repo "$xgnid_repo" --commit '<结果完整SHA>'
```

根目录包含来源锁/协议/配置，`seed42/43/44`包含公共准备、固定表示缓存、逐轮账本、恢复状态、最佳/末轮权重、回放与中文报告；`aggregate`为复验汇总，`fixed_test`仅在验证晋级后生成。`publication_history`保存上传历史与GitHub独立复算收据。
