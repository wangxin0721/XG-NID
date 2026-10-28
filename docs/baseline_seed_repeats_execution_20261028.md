# E0/E2 强基线重复实验（2026-10-28）

本轮只安排E0-B512 seed43/44和E2-B512 seed43/44共4次正式训练。
已有seed42直接复用；B-H512/1024收尾，不再补seed、重跑诊断或延长预算。
F0/F1/F2仅保留后续方案，不与本轮同时启动。
实现和本地检查已完成；服务器探测及4次正式训练尚未执行。

## 配置与版本

两模型均保留原数据划分、预处理、初始化批16、训练批512、验证批16、Adam lr0.001、weight decay0、最多100轮、patience12、严格验证Macro-F1选模、workers2、每轮107621样本/211次更新。
不引入辅助监督、校准、类别加权、困难样本或学习率搜索。
模型结构、参数、聚合方式及分类头分别沿用各自seed42，不要求E0/E2参数相同，也不将差异单独归因于编码器数量。

模型初始化seed和训练采样器seed都改为对应43或44；E0/E2在同一seed使用相同逐轮样本排列。
二者结构不同，初始化张量哈希不应相同。运行时分别记录实际初始化及RNG哈希，并验证探测与正式训练重建结果一致。

seed42旧入口版本：

| 模型 | 原入口SHA256 | 独立快照 |
|---|---|---|
| E0 | 33dfd3d6ff0acdae2dcdb31e986d3044200b32f11ed7e1b1b251be2115fb2b61 | scripts/baseline_seed42_frozen/e0_batch512.py |
| E2 | a290381515e74b2aeba3e40fcdc8e51ad798aa094d564c27f06ac0da3b78cabd | scripts/baseline_seed42_frozen/e2_batch512.py |

E2当前常用入口后来添加了1024支持，故从db232e4提取真正绑定seed42的旧版本；E0也按结果内绑定SHA提取。
两套入口及原配置快照字节完全匹配seed42记录，冻结模型/数据9个源文件另外核对哈希。没有编辑原训练文件。

独立AST适配器只做：把初始化和采样器seed参数化、记录实际seed、保留seed42复现校验、移除训练结束时的测试分区加载/评估/导出、增加批次进度和检查点状态元数据。
实际训练更新函数仍引用各自旧入口对象；单元测试确认整个epoch计算循环与旧版相同，仅剔除监控及新增状态记录后逐项比对。

旧快照的注释/配置中仍记载原先full后的测试评价，这是历史记录；新入口实际执行路径中仅有验证评价，并由图读取守卫禁止test分区。已有seed42曾有测试结果，这一历史曝光在分析中披露；新4次训练不调用测试集。

## 排程

默认物理GPU1，允许共享，顺序为E0_s43 → E2_s43 → E0_s44 → E2_s44。
每项先做原入口规定的1次预热+3次测量优化步的短探测，均为临时模型；正式训练使用新进程、干净初始化，探测更新不进入正式训练或指标。
各模型/seed启动前还用原seed42初始化做零优化步校验，确认原路径能重现绑定状态；这不是额外seed42训练。
新训练seed的初始化/随机状态也先记录，再在实际探测/训练构造时核对。没有额外候选、自动重试或自动增加seed。
任何任务失败立即停止队列，保留日志和best/last检查点，不覆盖旧结果。

输出根目录：`outputs/baseline_seed_repeats_b512_s43_s44_v1`。
根目录progress.json显示当前模型、seed、阶段、轮次和批次。
`updated_utc`来自子任务实际进度，`queue_poll_utc`仅表示队列检查时间；轮次/批次长期不动仍需排查。
每个正式训练目录为`E0_s43/E0`、`E2_s43/E2`等；探测目录单独以`_probe`结尾。

## 服务器启动

在普通终端提示符下逐行执行，不用heredoc结束标记：

```bash
cd /home/nzxn/下载/WX/load/XG-NID
git fetch origin codex/r0-edge-preparation
bundle=$(mktemp -d /tmp/xgnid_baseline_repeats.XXXXXX)
git -c core.autocrlf=false archive FETCH_HEAD scripts configs | tar -x -C "$bundle"
bash "$bundle/scripts/launch_baseline_seed_repeats.sh" 1
```

交付消息提供固定提交号，可替换FETCH_HEAD以复现本次版本。
启动器先执行标准库版本、证据、数据划分及AST入口检查。通过后后台提交4次训练；输出或日志已存在时拒绝重复启动。

实时进度：`watch -n 5 cat outputs/baseline_seed_repeats_b512_s43_s44_v1/progress.json`。
训练日志：`tail -n 5 -F outputs/baseline_seed_repeats_b512_s43_s44_v1.log`。

## 完成产物与判读

完成后自动核验6个模型/seed的配置、更新/曝光量、严格最佳与早停、验证混淆矩阵及8类完整P/R/F1、所有重叠轮次的配对数据顺序、检查点字节SHA。
seed42读取原验证指标，新4项检查test=None且test_evaluated=false。
分析不反序列化权重、不重跑模型，也不输出逐样本私有数据。

- `summary.json`：4次任务完成状态及分析摘要。
- `analysis.json`：每seed E2−E0 Macro-F1差值（百分点）、配对差值均值和样本标准差、各模型均值/样本标准差、全部8类P/R/F1及混淆矩阵、每项最佳/停止轮次、最佳及累计更新、样本曝光量、总参数量和描述性耗时。
- `report.md`：配对结果表、实际训练量、WebBased/Recon/BruteForce逐seed完整P/R/F1。

三seed只提供初步方向与波动证据，seed42已参与过探索；均值略高不等于稳定显著提升。
若E2多数seed更好且收益值得额外复杂度，再考虑冻结E2的F0/F1/F2；若接近/排序反复，结合性能及计算成本选主基线；若E0持续更好，优先围绕单编码器研究。
这里没有事先规定实用收益的数值阈值，因此分析器只报告方向、差值与代价，不自动认定实际意义或新增排程。
共享GPU的耗时仅为描述性记录，不作为独占性能基准；模型存在保留的非活动参数，total参数不等于有效容量。

## 本地验证

8项标准库测试通过：仅4次任务、初始化范围、epoch计算等价、只评估验证集、复用原更新对象、入口变化拒绝、配对样本标准差、缺失/非法seed结果拒绝。
真实本地seed42配置/入口/结果/权重字节/划分绑定检查通过；Bash语法检查通过。
另用seed42汇总复制为明确标记的模拟seed43/44，并使用非权重占位字节，验证完整报告流程及配对统计；这不是新训练结果。
本地未导入torch、未反序列化权重、未加载图或执行实际前后向。模型初始化复现与真实探测检查将在服务器执行。
机器记录见`baseline_seed_repeats_local_validation_20261028.json`。
