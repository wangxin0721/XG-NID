# N0／N1：E2 预训练起点下的融合改进

本轮替代此前 lr=.001、固定20轮的草案。最终设置为 lr=.0001、最多30轮、patience12，不启动冻结门控或额外基线重复实验。

2026-10-28 本机状态：已实现两个独立入口及配对队列，完成历史 E2 checkpoint、源码、split/manifest 哈希核验和本机6项检查。配置的 GPU-XG-NID 经跳板172.16.250.13连接，两次SSH超时；本机无torch/PyG，未执行真实张量等价检查、batch探测或正式训练。不能把静态检查通过表述为96.5994%已经实测复现。

| 项目 | N0 | N1 |
|---|---|---|
| 独立入口 | scripts/run_e2_n0_continue.py | scripts/run_e2_n1_concat.py |
| 融合 | H((zf+zp)/2) | Hcat([zf;zp]) |
| 分类头 | 8→32→8 | 16→32→8 |
| 总参数 | 846800 | 847056 |

两项保留原异构图编码器、边属性、Flow/Packet池化、两路64→8投影，全部更新编码器、投影和分类头。两路均读取完整异构图，不称为隔离模态。N1只扩展shared_head[0]并改为拼接前向；不使用历史拼接模型。新第一层权重为[W/2,W/2]，复制偏置，激活及后续层原样保留。历史gate和辅助头仍保留用于权重兼容，不进入损失。

两项严格加载同一个登记E2-B512 best.pt，SHA256为1709ed1a19b8a2e20b9b1ef3dac6cc6a5906fd93937e8d18a51cc64f85c44b08。只读取注册表的E2参考，不依赖E0/E3结果或额外诊断。此前seed42已有测试结果，本轮属于验证集探索。

## 固定规则

- 原冻结划分，seed42；训练batch512，评价batch16，workers2。
- 替换层后新建Adam，lr=.0001、weight_decay=0；两项均不继承旧动量，无调度器。
- 仅最终分类NLL，不调用原model.loss()，无辅助监督、类别权重或新增编码器。
- 最多30轮，patience12；严格验证Macro-F1改善才替换，epoch0纳入选模，同分保留较早轮次。早停从干净起点计数，不沿用历史stale。
- 每完整轮107621个样本、211次更新；最多6330次更新/项。各自早停，总更新数可能不同，报告实际轮数、更新数及曝光量。
- Python/CPU/CUDA训练起始RNG相同。预先生成30轮采样顺序SHA，每项逐轮核对，并在最终汇总核对两项公共轮次顺序。不声称不同结构训练逐位一致。
- 不评价测试集，不自动增加seed、调参、重试或延长预算。

## 训练前与训练后检查

1. 在全部13446个验证样本上，N0必须复现原E2验证矩阵及Macro-F1（报告四舍五入为96.5994%）。N1必须与N0预测类别零差异，log-probability允许allclose(atol=rtol=1e-5)，检查模型参数/缓冲区未改变。
2. 两项各做3个batch的可丢弃探测，确认损失与梯度有限，两个编码器、两个投影及分类头有梯度；第一层权重实际更新；旧gate和辅助头没有梯度。探测后严格恢复干净模型状态、清空梯度及缓存、恢复起始RNG，正式训练重新创建Adam。
3. 保存epoch0候选、每轮日志、last.pt及best.pt。加载最佳模型后再次复现验证矩阵，保存完整逐类P/R/F1及逐样本预测。只保存state_dict，重建N1时需通过本轮适配入口恢复结构。
4. 分别报告N0、N1相对原E2的Macro-F1百分点差及修复/退步计数；配对汇总报告N1−N0和按类别的修复/退步计数。预测按sample_id匹配，拒绝重复ID或标签不一致。保留last_validation，best_epoch=0表示没有验证提升。

任何训练前检查失败，立即停止，不进入正式训练。没有实现resume；已存在输出拒绝覆盖。status.json记录当前阶段，根summary.json只在两项均完成且配对检查通过后产生。

## 服务器启动

代码修改、核验、提交和推送由助手完成；用户仅在服务器执行以下拉取与启动命令，无需手动传包。通过git archive提取scripts/configs到独立目录，不替换服务器现有源码、数据或权重。启动器固定使用原wx环境，可通过XGNID_PYTHON指定已有解释器。

```bash
cd /home/nzxn/下载/WX/load/XG-NID
git fetch origin codex/r0-edge-preparation
rev=$(git rev-parse FETCH_HEAD)
bundle=$(mktemp -d /tmp/xgnid_e2_n01.XXXXXX)
git -c core.autocrlf=false archive "$rev" scripts configs | tar -x -C "$bundle"
bash "$bundle/scripts/launch_e2_concat_pair.sh" 0
```

正式交付命令使用助手核验已推送的固定提交号替代上面的"$rev"。注册表从交付配置同目录读取，允许/tmp下的独立源码包；结果只允许写入服务器仓库outputs。运行环境固定Python3.9.23 / torch2.3.1+cu118 / PyG2.5.3。

启动器后台依次执行初始检查、两项探测、N0正式训练、N1正式训练及配对分析。物理GPU0默认要求空闲；如明确允许共享，用 `bash "$bundle/scripts/launch_e2_concat_pair.sh" 0 --allow-shared-gpu`。共享耗时不能用于独占速度比较。

```bash
tail -n 30 -F outputs/e2_concat_pair_b512_s42_lr1e4_v1.log
cat outputs/e2_concat_pair_b512_s42_lr1e4_v1/status.json
```

Ctrl+C只退出tail。单独探测可用新输出目录：

```bash
python -B scripts/run_e2_concat_pair.py --mode probe --physical-gpu 0 --output outputs/e2_concat_probe_s42_v1
```

以下独立入口命令假设代码已同步到仓库scripts/configs；独立解压时请将脚本路径改为"$bundle/scripts/..."并添加 `--config "$bundle/configs/e2_concat_pair_20261028.json"`。两个独立入口也可分别运行，每个都先完成相同初始检查和可丢弃探测。以下命令是配对队列的替代方式，不要两种方式重复启动：

```bash
python -B scripts/run_e2_n0_continue.py --physical-gpu 0 --output outputs/e2_n0_s42_lr1e4_v1
python -B scripts/run_e2_n1_concat.py --physical-gpu 0 --output outputs/e2_n1_s42_lr1e4_v1
python -B scripts/run_e2_concat_pair.py --mode analyze --output outputs/e2_independent_pair_s42_v1 \
  --n0-output outputs/e2_n0_s42_lr1e4_v1/N0 --n1-output outputs/e2_n1_s42_lr1e4_v1/N1
```

## 决策边界

主要看N1−N0，同时看各自相对原E2的变化。N1优于N0和原E2后再冻结方案、补配对seed43/44；两项都提高且接近时，暂不能把收益归于拼接。N1没有优势则结束本候选，后续单独研究8维压缩，不自动混入其他模块。E0保留为最终强参考。

本机检查：`python -B scripts/test_e2_concat_pair.py`。覆盖Python3.9语法及无torch导入、真实参考哈希/配置拒绝、epoch0严格选模与patience边界、sample_id配对计数和错误身份拒绝，以及用NumPy张量替身执行实际转换函数的数值等价检查。NumPy检查不能替代真实checkpoint/GPU检查。
