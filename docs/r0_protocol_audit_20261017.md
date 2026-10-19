# R0 Edge 复现准备实交付（2026-10-17）

本次执行代码核查、独立入口实现和合成小图测试，不修订 v3，不开展 M1，不改变冻结 E0～E3。R0 当前名称必须是 **官方 Edge 协议重实现**；没有原论文复现成功结论，也没有服务器冒烟通过结论。

## 1. E0/E1 实际证据

依据已提交 `experiment_plan_20261017.md` 的台账：E0 seed42 已完成，最佳31轮、43轮早停，completed/exit_code=0；验证 Accuracy/Macro-F1/Weighted-F1=96.7128/95.1084/96.7202%，测试=96.8248/95.2321/96.8235%，431704参数，约5:59:39。冻结 split ID=`a8ae9c934eec9fee970cc725e60051d0c709e0f16c7dda14749350ff2006975a`，样本数107621/13446/13448。

这是**既有回传的台账证据**。本地 `outputs/grouped_full_E0_s42_v1` 不存在，无法本轮重新核验服务器权重、日志、预测、配置、epochs、summary和status的完整交付。不能把旧 `outputs/GNN4ID*` 或本地pilot代替正式分组E0。完整文件仍保留在服务器，需用交接命令采集清单。

E1：台账仍是“命令已提供、运行/完成待回传”。本地没有正式E1的PID、日志、status或完整结果。发现已有SSH别名GPU-XG-NID，并用非交互只读连接检查；连接在跳板阶段超时（`Connection timed out during banner exchange`）。未访问到服务器进程，未启动/停止训练。E1运行状态**未知**，不能记为未启动、运行中或已完成。见 `r0_server_runbook.md` 的发现与取证命令。

## 2. 固定来源与可恢复性

- 论文：作者仓库链接的 [arXiv 2408.16021v2](https://arxiv.org/html/2408.16021v2)，2025-05-07；本地已有正式期刊第9页截图 `tmp/pdfs/xgnid_paper_page9.png`，页眉ESWA287(2025)128089，第3.5.1节/表4与v2对应内容一致。没有完整期刊PDF的本地可审计副本，不能据单页证明全文逐字一致。
- 仓库：[GNN4ID固定提交551d1f10189e5259f8f34e32f918d557a2b3a64e](https://github.com/Yasir-ali-farrukh/GNN4ID/tree/551d1f10189e5259f8f34e32f918d557a2b3a64e)，提交时间2026-02-23。已从官方API和固定raw URL实际下载核对。提交晚于发表，**未证明它就是论文训练时的提交**。
- 文件原始SHA256、换行归一化比较、Python AST比较见 `r0_source_lock.json`。当前vendored Model/Training/Functions与官方固定版归一化文本相同、AST相同，原始hash差别来自换行。R0运行时锁定Model正文；既有vendored文件不改写。
- 两个本地notebook不是官方字节副本：模型notebook改了路径、glob导入、测试CSV位置和保存方式，预处理notebook调整了import单元位置。核查依据为下载的官方版，未覆盖本地notebook。

“已确认”表示有本次可读代码/原文证据；“可恢复”表示给定输入与清单可执行恢复，不表示已经运行服务器；“缺少证据”表示可能存在但未取得；“无法恢复”仅限目前公开文件/现有产物，未来取得原始清单可重新审定。

## 3. 协议差异表

官方依据分别是固定版 `Utility/Model.py`、`Training.py`、`Functions.py`、`GNN4ID_Model.ipynb`、`Data_preprocessing_CIC-IoT2023.ipynb`。下表中的官方训练设置是**公开示例协议**，不能直接认定为生成论文分数的完整运行记录。

| 项目 | 官方可见实现 | 当前项目/冻结分组 | R0处理与证据状态 |
|---|---|---|---|
| Edge架构 | 两层HeteroConv，关系GATConv((-1,-1),64,edge_dim=-1,add_self_loops=False)，默认跨关系sum | E0/E1直接包装同一个官方类；自定义edge/双分支编码器使用mean | **已确认**；复用ReferenceEdge并校验官方源码正文 |
| 节点变换 | 每层各节点类型BN(eps=1)+LeakyReLU，无额外自环/残差 | E0一致；自定义编码器有独立实现 | **已确认**；不换成自定义edge类 |
| 图级分类 | 分别mean pool Flow/Packet64，按flow/packet插入顺序拼接128；Linear128→64→16→8，无层间激活，log_softmax/NLL | E0一致；自定义baseline头含额外激活 | **已确认**；R0检查节点插入顺序 |
| Flow特征 | README称82维；Functions按CSV剩余列的原顺序取float32，去Label和14个报文列；不做标准化 | 服务器旧图/冻结组Flow82；本地旧pilot构图不同 | **维度与规则已确认**；具体82列顺序须对应CSV表头和图生成来源，现有.pt不能单独证明 |
| Packet特征 | 默认仅payload，字节0..255，截断/补零至1500，不启用8flags | 服务器旧图1500；当前通用graph_building曾输出1513 | **已确认**；R0只接受1500，不截断1513来冒充官方 |
| Contain属性 | direction、ip_size、transport_size、payload_size，float32四维；反向store复制同一属性 | 冻结旧图4/4维 | **已确认**；校验反向索引和属性；不把transport_size解释成头长 |
| Link属性 | delta_time[1:]先转int再float32，原始shape[E] | 旧图支持[E]或[E,1] | **已确认**；读取时仅clone并unsqueeze，不重写文件、不新增数值；时间单位须追溯提取器/原始数据 |
| 图连接 | 一Flow连全部Packet，相邻Packet链；随后ToUndirected增加reverse contain并双向化Link | 冻结旧图采用此结构；早期本地pilot有单向Link | **已确认**；R0拒绝不符合官方双向链的图，不自动修图 |
| 类别顺序 | Benign/WebBased/Spoofing/Recon/Mirai/Dos/DDos/BruteForce=0..7 | 冻结组同映射 | **已确认**；配置与清单一致校验 |
| 数据来源 | 发布处理后数据链接，notebook从源train/test读取官方编号图 | 冻结组把已有图按输入指纹重新分组；历史paper-align是另一次按数量构造的split | **缺少作者身份清单**；R0保留官方文件名编码的现有源成员，或要求显式路径/offset/partition。不得用旧随机split或分组split充当原划分 |
| 数据规模 | 表4列出了每类train20000；test六类4000、WebBased1090、BruteForce467 | 分组107621/13446/13448；旧paper-align训练144000、验证16000、测试25557 | **数量可核对**；R0正式门槛train160000/test25557且逐类一致，数量相同仍不证明身份一致 |
| 原始采样 | split_csv部分sample固定42；训练聚合下采样未设random_state；补样用Python random.randint无seed | 冻结组seed42按指纹分组，不再采样 | **规则可恢复，具体随机成员无法由当前公开代码恢复**；不重抽测试成员、不按分数挑划分 |
| MAC过滤 | 论文描述攻击保留攻击者MAC/Benign排除攻击者MAC；官方Benign分支只计算df[mask]，没有赋回df，攻击分支有赋回 | 现有图已无MAC身份列，未证明执行的是哪版过滤 | **已确认代码/文字偏差；当前数据是否受影响缺少证据**；不改历史数据、不宣称已完成论文过滤 |
| 亚类比例 | 论文描述保留比例；官方按亚类先切分/限样，再合并主类采样 | 当前图仅8类标签，缺亚类身份 | **缺少证据**；无法从8类计数恢复亚类比例 |
| Adam/lr | Edge训练函数Adam(lr=args['lr'])，notebook lr0.01 | 冻结Adam0.001 | **已确认**；R0=0.01 |
| weight_decay | notebook字段1e-5，但Adam构造未传入；实际默认0 | 冻结0 | **已确认**；R0=0，不照抄未生效字段 |
| batch/预算 | notebook train64、test1、30epoch | 冻结train16、上限100epoch | **已确认**；R0恢复64/1/30，冒烟仅32训练图/2轮 |
| scheduler | Edge函数scheduler构造/step都被注释；非Edge的plateau不能移植过来 | 冻结固定lr | **已确认**；R0不启用scheduler |
| 选模 | Edge函数没有验证集、best变量未使用；测试在训练后，保存最终模型 | 冻结验证Macro-F1、patience12、best后测试 | **已确认示例行为**；R0最终30轮last.pt，无best/早停；论文具体选模运行证据缺失 |
| 训练随机种子 | notebook无torch/numpy/Python训练seed；局部sample42不等于训练seed42 | 冻结明确42 | **缺少证据，原始RNG轨迹无法恢复**；R0新增42并披露，含显式lazy初始化及字典序输入次序 |
| 测试评价 | classification_report(label,prediction)与最终归一化CM真值行/预测列；Training.calculate_metrics中的一次原始CM参数顺序反了 | 冻结输出macro/weighted/per-class等 | **已确认**；R0显式输出正确方向CM、Accuracy、Macro/Weighted-F1与support；不复刻打印方向错误 |
| 论文分数 | v2表5/6 Proposed Precision0.95、Recall0.99、F1约0.97；2026-10-19补查用户期刊PDF第11页图5，纵轴明确Macro F1-Score，CIC-IoT2023为97% | E0测试Macro-F1约0.952321属另一协议；R0真实Macro-F1为0.9282109574 | **更正：图5的97%明确是Macro-F1**；精确预测和对应Edge检查点仍缺少证据。R0与该报告值约差4.18个百分点，不能拿Weighted-F1缩小差距，也不据此调测试集 |
| 环境 | 官方requirements torch2.3.1+cu118/PyG2.5.3/sklearn1.3.2 | 本地torch2.8.0+cpu/PyG2.6.1；服务器当前环境待实测 | **公开声明已确认、运行环境待确认**；不替换服务器现有CUDA包，记录实际版本和差异 |

## 4. R0可执行实现和数据方案

- `configs/r0_edge.json`：单独冻结官方示例Edge设置与重实现声明。R0不使用train_grouped的早停/验证split。
- `scripts/r0_data.py`：只读官方编号图或显式locations JSONL，输出新目录的samples.jsonl、manifest.json、membership_evidence.txt和status。每行保存位置/偏移、来源分区、标签、Edge/NoEdge哈希及唯一行ID。过采样副本保留为不同位置，不去重，不改变评价权重。报告分区内重复、跨分区共有输入与影响测试样本、冲突标签。
- 文件名模式仅接受`data_N.pt`/`data_test_N.pt`，**只证明当前文件命名声称的分区**。映射来源另存evidence；没有原作者row/capture身份仍保持historical_author_sample_identity_verified=false。其他命名/多图容器必须显式locations，不猜测缺失位置，也不直接套用历史随机split索引。
- `scripts/run_r0_edge.py`：R0独立smoke/full流程、Adam/NLL、每轮训练集评价、last.pt与优化器/RNG、epochs、config、最终测试metrics/predictions、summary/status/error日志。full只在最后一轮选定后测试一次。CPU smoke不读测试图；manifest扫描会读取测试图检查输入，不计算测试预测。
- 正式运行必须满足逐类表4数量，并持有同manifest/config/代码/torch/PyG/CUDA/GPU设备类型的真实CUDA smoke报告和未变动smoke checkpoint。输出目录存在即拒绝；没有自动resume。数据和GPU尚不具备本地执行条件，命令见runbook。
- `scripts/collect_experiment_status.py`：只读采集E0/E1文件存在性、JSON、epoch尾部及可选PID/日志；不能将文件存在直接解释为进程仍运行。

既有文件不修改，尤其`src/xgnid/model.py`、`edge_experiments.py`、`scripts/train_grouped.py`、grouped loader/split、参考Model及E0～E3所有数据/历史结果。实际加载会核对原始图哈希，clone仅用于标量Link视图兼容。不能对源图修补后仍称同一输入。

## 5. 本地已完成检查与未完成项

在`C:/Users/wx/anaconda3/envs/nids/python.exe`运行`python scripts/test_r0_edge.py`，4项全部通过，最近完整结果：`Ran 4 tests in 1.122s ... OK`（不含解释器启动/包导入时间）。合成图24张（train16/test8），Flow82/Packet1500，4包双向链；不是论文数据实验。四个新增Python脚本py_compile通过，冻结源码相对HEAD差异检查退出0。

1. 2轮CPU训练/反向/Adam更新、输出[B,8]、六个GAT真实收到4/4/1边属性、非零边投影梯度、逐关系干预与重复前向抖动、last checkpoint重载。Contain单入边允许零敏感度，不要求所有边投影都非零。
2. 跟踪实际图读取，仅train被smoke加载；禁止覆盖；逐类非表4数量不能进入full；篡改counts_match标志仍不能绕过真实数量门槛；图特征变化/清单变化被拒绝；未审定配置被拒绝。
3. E3双分支shape与门控/置信度混合/归一化/shared_head实际计算一致；关闭use_gate等于0.5表示融合。详见E3报告。
4. 用明确标注的人工授权记录单测服务器门槛：CPU、不同manifest、不同代码、未完成两轮、checkpoint哈希变化均拒绝；这验证放行逻辑，不构成真实GPU冒烟证据。

未完成：服务器清单扫描、真实源图完整性/身份审查、CUDA属性路径及优化步骤、真实R0两轮smoke、全量30轮训练、真实指标与成本；E1真实状态；E0全套原始产物逐文件复核。当前只能验收协议核查、可恢复性说明、独立入口、轻量测试及命令、E3代码核查这五类本地交付；**E1项未完成，阶段六项不能整体通过**。五类本地交付也不等于R0服务器就绪或复现成功。

## 6. Git交付边界

仅提交本次新增的configs/r0_edge.json、r0_data/run_r0_edge/test_r0_edge/collect_experiment_status四个脚本、r0_source_lock.json、三份R0/E3文档。现有工作区大量无关修改/删除、数据、权重、预测与原notebook修改不纳入提交。提交信息：`feat: add isolated official Edge protocol reimplementation and audit`。最终SHA/推送状态以本次实际Git操作和聊天回报为准；本文件不预填未产生的SHA。
