# Seed42：真实连续配对 vs 流内第二项重新配对

本轮只运行 True 与 Shuffle 两组联合训练。代码基于 `4f2dd776fd3040a94dd4199d47866f1ed9968a03`，独立分支为 `codex/transition-pair-shuffle-s42`，独立输出目录为 `outputs/transition_pair_shuffle_s42_v1`。原工作区和历史结果保留。

当前状态：本地合成 CPU 验证通过；本机 Torch 2.3.1+cpu、PyG 2.5.3，无 CUDA。服务器原图 CUDA 探测和正式训练尚未执行，测试集未评估。验证日志和源文件摘要见 `docs/transition_pair_shuffle_s42_v1/cpu_validation.json`、`cpu_validation.log`。

## 依据与检验范围

历史结果来自 `ac00f1f319aec1b1079e15ca5609c2ec08c85ba9` 的 `outputs/joint_transition_fusion_s42_v1`。历史 J3 验证 Macro-F1 为 97.874050%，最佳第 84 轮；它仅作为外部参考，不能代替本轮重新训练的 True。

两组均沿用 `run_ordered_transition_readout.load_source(..., seed=42)` 的实际原 C 与来源校验；必须与历史 J3 的来源绑定完全一致。原 C 模型摘要为 `c29a6433880f6e5ad8a45d387abef07cf54f895277f9f89b313edecfa7feedc6`，实际 C 检查点 SHA256 为 `3070acd40203ee3a3944dc5805452eb3a2d73218aa24b31379bdfd5cbb523568`。

该消融检验新增读出内的显式连续配对作用。原图编码器已经包含的时序信息仍然保留。本轮不追加 seed43/44，不评估测试集。

## 两组的唯一干预

| 内部标识 | 简称 | 第一个转移 | 第二个转移 |
| --- | --- | --- | --- |
| pair_true | True | 保留真实片段第一项 | 原片段真实连续第二项 |
| pair_shuffle | Shuffle | 同样的第一项，顺序保持 | 置换同流真实片段的第二候选列表 |

真实片段继续由 `attach_continuity` 的共享中间报文标记确定。有效转移数 ≥2 不足以证明存在连续片段。对每个流，先取所有真实片段的左索引 `left` 和右索引 `left+1`；Shuffle 只重排右索引候选。两侧各自的集合、出现次数、片段数量和第一项顺序均保留，禁止跨流。

有至少两个片段时，生成无固定点置换，每一条片段的右索引都改变；一个片段保持原样；没有片段时沿用原单尺度回退。此定义允许自配对和逆序组合：例如两个片段时，唯一无固定点置换是交换两条右候选。它改变片段内部对应关系，并非只调整完整片段列表的排列。

固定独立配对种子为 **20261121**。算法版本 `sha256-local-random-rejection-derangement-v1`：以版本、种子、分区、sample_id 和真实片段数的规范 JSON 计算 SHA256，将摘要作为局部 `random.Random` 的种子，拒绝有固定点的候选置换。生成过程不读取标签，不触碰全局 Python/NumPy/Torch/CUDA RNG，也不访问训练采样器的 generator。训练、验证各自固定；同一样本在不同轮次、batch、缓存重载和续训中保持同一映射。

映射逐样本保存于 `pairing_train.jsonl.gz`、`pairing_val.jsonl.gz`，含第一项、真实第二项、置换后的第二项、置换索引、片段数量、索引改变数及实际特征改变数。逐类覆盖统计在映射完成后才读取标签。

实际特征改变按冻结的完整 70 维转移向量逐项精确比较；“特征改变比例”的分母为改变了索引的片段数。另报告至少一条片段特征改变的流占可打乱流的比例，避免把索引变更等同于信息变更。

## 固定训练协议

- 两组从同一原 C 的融合参数及相同读出初始化开始，读出输出投影全零；全量第 0 轮 log probability 和预测必须与原 C 一致。
- 每组 7,144 个融合参数和 3,728 个读出参数，共 10,872；参数存储和 Adam 状态相互独立。
- 两路图编码器、原跨层表示及原查询池保持冻结和 eval。训练只优化 Flow/Packet 投影、逐维门控、最终分类头及 Ordered 读出。
- Packet 均值查询、单转移输入、可用尺度平均、八分类最终 NLL 保持 J3 定义。缓存只保留冻结特征，训练前向重新计算融合层。
- 主干 seed42，读出 seed42；固定原 split、样本清单和八类映射，train/val/test 数量为 107621/13446/13448，只加载 train/val 图。
- 完整 200 轮，无早停；batch512、eval16；每轮 211 次更新，每组 42,200 次更新。同一个 batch 依次更新 True、Shuffle，样本顺序完全相同。
- Adam betas=(0.9,0.999)、eps=1e-8、weight_decay=0、amsgrad=False；读出 lr=0.001，融合 lr=0.0001。
- 每轮验证，以总体验证 Macro-F1 选最佳，包含第 0 轮，同分选最早。分组结果使用总体选出的检查点，不分别按分组选模。
- 汇总 30/100/150/200 轮预算窗口。True 相对历史 J3 的差异单独报告，以本轮同时实施的 True/Shuffle 配对为主。

## 验证与恢复

本地 26 项 CPU 检查覆盖无固定点、数量与集合保持、流间隔离、真实连续性、零/单片段回退、重复向量下的实际改变率、标签与全局 RNG 隔离、batch 重排、缓存往返、与 J3 相同的 True 前向、非零输出读出的实际变化、显式公式复核、梯度分阶段建立、冻结状态、两组 Adam 中途恢复、配置错配拒绝、最佳原图回放、分组输出与自动上传路由；同时回归原 18 项检查。

服务器入口先用原图训练样本进行梯度与非零输出结构探测，再验证原 C、两组全量第 0 轮一致，准备固定映射和覆盖统计。缺少可打乱样本或第二项特征完全未改变时，保存诊断并停止，避免把无效干预写成完成训练。

`recovery.pt` 继续在每个配对更新事务和每轮验证后原子保存模型、两组独立 Adam、轮次、更新数、随机状态、采样器排列/游标、损失累积、账本和最佳权重。若只更新了部分组就失败，该事务未提交，恢复从上一完整事务继续。恢复校验配置、代码、环境、原 C、冻结缓存摘要和两分区映射摘要，拒绝来源不符或改变置换种子的状态。

最终最佳检查点必须经过原图验证回放，重新由相同 sample_id 生成固定映射，并核对索引、预测摘要和全部分类指标。归档预测中的 `flow_weights` 等附加字段不会再参与整个字典比较；真正的样本、标签、预测、指标或摘要差异仍会失败并保留收据。

## 服务器启动

激活原 wx 环境，使用现有原始图、固定 split、实际原 C 和已完成的 J3 输出。服务器历史目录只读；配置已绑定所需历史文件的 SHA256。

```bash
conda activate wx
cd '/home/nzxn/下载/WX/load/XG-NID'
xgnid_repo="$PWD"
xgnid_code="$(dirname "$PWD")/XG-NID-transition-pair-shuffle"

git fetch --no-tags origin refs/heads/codex/transition-pair-shuffle-s42
xgnid_revision="$(git rev-parse FETCH_HEAD)"
if [ -e "$xgnid_code" ]; then
  test "$(git -C "$xgnid_code" rev-parse HEAD)" = "$xgnid_revision" || {
    echo '已有工作树版本不同，请保留现场并核对版本。'
    exit 1
  }
else
  git worktree add --detach "$xgnid_code" "$xgnid_revision"
fi

mkdir -p "$xgnid_repo/tmp" "$xgnid_repo/outputs/transition_pair_shuffle_s42_v1"
nohup python -u -B "$xgnid_code/scripts/launch_transition_pair_shuffle.py" \
  --repo "$xgnid_repo" \
  > "$xgnid_repo/tmp/transition_pair_shuffle_s42_v1.launcher.log" 2>&1 < /dev/null &
echo $! > "$xgnid_repo/outputs/transition_pair_shuffle_s42_v1/launcher.pid"

tail -F "$xgnid_repo/outputs/transition_pair_shuffle_s42_v1/training.log"
```

图根目录若与 split 中登记路径不同，在启动器增加 `--data-root '/实际原图根目录'`。如只先执行 CUDA、小批原图和全量零轮探测，可用：

```bash
python -u -B "$xgnid_code/scripts/launch_transition_pair_shuffle.py" --repo "$xgnid_repo" --probe-only
```

该探测准备验证集固定映射及覆盖；正式入口随后还会构建完整训练集缓存并校验训练集有效覆盖。中断后，待旧启动器退出，重新运行同一后台入口，自带 `--resume`。

```bash
# 查看启动/上传输出和退出状态
tail -F "$xgnid_repo/tmp/transition_pair_shuffle_s42_v1.launcher.log"
# 可捕获地停止：保留已提交恢复状态并触发现场上传
kill -TERM "$(cat "$xgnid_repo/outputs/transition_pair_shuffle_s42_v1/launcher.pid")"
```

## 输出、上传与判读

新目录保留配置/来源/环境/输入与代码摘要、原图探测、零轮收据、冻结特征、两分区固定映射及覆盖、每组账本/best/last/最佳预测、完整恢复状态、原图最佳回放收据、逐类 P/R/F1 与混淆矩阵、总体和可打乱/不可打乱配对变化、门控/残差/注意力、预算窗口、中文报告及训练/上传日志和退出码。

主要入口与模块为 `scripts/run_transition_pair_shuffle.py`、`transition_pair_cache.py`、`transition_pair_results.py`、`launch_transition_pair_shuffle.py`、`src/xgnid/transition_pair_shuffle.py`。共享训练器、诊断器及启动器增加可复用的两组支持，旧四组默认行为由回归测试覆盖。

统一启动器在正常结束或可捕获失败后，复用 `sync_server_outputs.py` 将**整个新目录**发布到 `codex/server-outputs`，大文件沿用 LFS。新目录发布不会替换旧实验目录。训练退出码保留，上传日志关闭后进入收据发布，避免源目录校验期间变更日志。

训练停止后，上传失败可独立重试：

```bash
python -B "$xgnid_code/scripts/sync_server_outputs.py" publish --repo "$xgnid_repo" --only transition_pair_shuffle_s42_v1
```

总体及可打乱/不可打乱分组结果在 `validation/all`、`validation/shuffleable`、`validation/unshuffleable`。分组 Macro-F1 始终按固定八类计算，零支持类保留为 0；空分组报告 NA。不可打乱流的输入结构相同，但两组训练后的公共参数可能不同，预测不要求一致。

事前参考为 **True−Shuffle ≥0.05 个百分点**，使用未四舍五入值判断，并结合逐类得失和可打乱覆盖率决定资源投入，不作统计显著性声明。若 True 没有优势，如实报告，不能继续把收益归因于真实相邻配对。支持后再安排 seed43、44；本入口不自动追加种子或测试评估。
