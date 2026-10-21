# E3门控改进执行计划与服务器交付（2026-10-21）

用户已要求制定计划并推进。本次落实独立诊断、E3a和等权初始化对照的配置与入口，不修改冻结模型/训练文件或已有结果，不将候选机制写成已证明的根因。沿用本地审查开发→GitHub交付→服务器真实实验的流程。

用户随后转发了另一环境的`gate_improvement_plan_20261021.md`和临时目录截图。本文吸收其三个控制要求：E3a仍保留代理输入、单路必须通过H、分支监督必须配等权对照；并补充WebBased验证诊断。其`aedf3e8`不在本机Git中，也未收到对应代码，不能声称合并了该提交或通过了其人工CPU模型测试。本交付的实际文件均在`D:\XG-NID`，不是另一ChatGPT临时目录；本机检查只使用标准库。用户已明确授权将本批计划、配置、工具和检查记录公开推送，不包含数据集、权重或预测文件。

后续真实状态更新（2026-10-21）：阶段B已回传completed/exit0，两组原验证结果复现、模型状态不变、测试图0；详见[诊断结果与补充缓存分析](e3_gate_diagnostic_results_20261021.md)。它显示E3内部动态融合有作用，去掉直接混合的推断Macro略降。先用已有缓存分解WebBased失败，再进入C/D两项最小训练；不预设代理混合是主因、不强行压平权重。下表保留首次交付时点，最新执行证据以该结果记录为准。

补充缓存分析现已回传completed：WebBased错误81个，其中偏正确路仍错49、偏错误路8、两路均错24；49个中Packet-only48、Flow-only1。详见[缓存结果与E3a命令](e3_gate_cache_results_20261021.md)。阶段C进入真实probe/筛选待执行，不新增其他训练机制。

最新执行修复：用户随后回传E3a旧probe成功、screen在初始化字典精确比较处失败；正式screen未进入训练。当前请改用[初始化核对修复说明](e3a_initialization_check_fix_20261021.md)和新版本probe_v2/screen_v2，下面首次交付的旧screen命令仅作历史记录。诊断、缓存、共享模块和模型训练定义不变。

## 当前问题与验收目标

同batch512、seed42：E2最佳验证Macro-F1为96.5994%，E3为96.3669%；测试分别96.6583%和96.1653%。这不能单独定位问题，但足以要求先检验门控是否利用互补性。当前E3的熵/margin来自没有独立类别监督的8维中间表示，还直接参与最终权重混合。E3-B512权重较极端只是诊断线索，不据此强制所有样本接近等权。

本轮目标：复现原检查点验证结果；分离学习门控、直接代理混合、初始偏向这三个因素；以验证集判断候选是否值得后续配对重复。原测试此前已被查看，本研究登记为探索性追加。本轮筛选不创建测试图加载器，也不产生新的测试分数。

## 执行顺序

| 阶段 | 工作 | 执行门槛和交付 | 当前状态 |
|---|---|---|---|
| A 元数据与产物 | 核对E2/E3-B512的摘要、配置、9份冻结源码、4个best/last文件和样本清单身份 | 配对输入、源码、初始化及训练控制一致；模型加载前完成哈希检查 | 本地标准库检查通过；服务器仍重新核对 |
| B 验证诊断 | 原检查点逐样本比较单路、等权、固定alpha、纯门控推断干预和原动态输出 | 先严格复现原验证混淆矩阵；状态不变；输出缓存和report.json，测试图加载0 | 独立工具已实现，服务器执行待回传 |
| C E3a | 从头训练，只去掉最终权重中代理混合及其归一化 | B成功报告匹配；新probe通过；train512/eval16，seed42、lr0.001、max100、patience12不变 | 独立probe/screen入口已实现；尚未执行 |
| D E3a_equal_init | 在C的纯门控基础上，只将最后一层Linear权重/偏置初始化为0 | C完整验证筛选结束且预算允许；独立新probe；其他初始张量与随机状态保留 | 入口已实现并设前置约束；尚未执行 |
| E 决策 | 比较验证总体/逐类收益、权重选择、互补性与成本 | 确定是否进入分支监督研究，或保留等权主对照；必要方法再补43/44 | 等待真实B/C/D结果，不自动扩大搜索 |

先执行B，回传结果后再安排C。D是与C的单因素对照，不与C并行启动，也不从C的已训练权重继续。两项筛选均不自动做最终测试；正式测试和多种子确认另按冻结选择登记执行。

## 配置与实现边界

- [配置](../configs/e3_gate_study.json)：固定当前两组B512参考、split、控制、21点alpha规则、两个候选和预算。
- [共用标准库模块](../scripts/e3_gate_study.py)：源文件/检查点字节哈希、配置守卫、GPU UUID查询、禁止测试图的加载守卫、纯门控AST变换与alpha并列规则。
- [验证诊断入口](../scripts/diagnose_e3_gate_server.py)：`check`不导入torch；`diagnose`仅在Linux服务器加载已核验权重，eval/no_grad，不创建优化器或反向计算。
- [候选训练入口](../scripts/run_e3_gate_ablation_server.py)：`check / probe / screen`。只提供训练/验证筛选，无full/test模式；不改旧代码、不覆盖旧结果，无自动重试、降batch、恢复或连跑。
- [纯标准库检查](../scripts/test_e3_gate_study.py)：覆盖真实参考文件字节、真实验证矩阵、AST改动边界、alpha并列及诊断/配置拒绝条件，不运行模型或合成模型实验。

E3a通过解析原`_graph_logits`函数，**只删除两条精确匹配的后门控赋值语句**；其余前向AST保持原样。原门控输入中的熵、margin、余弦及两路z全部保留。因此E3a只隔离直接代理混合，不是“全部移除置信度输入”。变换后的AST哈希写入检查点配置。

两个候选先按原32张训练候选/16图初始化顺序完成lazy初始化，并核对原初态SHA `0270349d613f11b453b92e4e928a3c2e7b67d062b5e3ba73c8ba4fb58dbd521a`。E3a保留全部张量；等权版随后只把`gate.net.2.weight/bias`置零，不重新抽样初始化，不调用新的随机数。新初态SHA及所有变化张量单独记录。真实probe会验证最终权重等于原始学习门控；等权版另验第一次实际eval权重恰为0.5/0.5。

原B512入口共享的初始化、TrainBatches、图指纹验证、DataLoader/评价batch、普通NLL、Adam参数、严格改善早停和best/last保存继续使用。训练2个worker显式使用Linux fork，并在每个进程的实际图读取前拒绝测试partition。诊断使用0个worker以精确记录实际加载计数，评价batch仍为16；模型初始化需要读取32张训练候选，之后仅遍历验证图，这与“无新训练、测试图0”分别记录。

## 诊断内容与正确解释

每个原检查点输出：

1. `H(z_flow)`、`H(z_packet)`、`H(0.5*z_flow+0.5*z_packet)`、原动态输出，以及E3的`H(g_flow*z_flow+g_packet*z_packet)`纯门控推断干预。这些是同一冻结模型干预，不是独立训练的E3a或单模态基线。E2旁路未训练的门控，raw_gate记null，不报告其纯学习门控成绩。
2. alpha为0、0.05、…、1，共21项；按验证Macro-F1选择，完全并列先最接近0.5再较小alpha。固定alpha只缓存/计算分类头，不训练21个模型，不读取测试图。
3. 两路均对、仅Flow对、仅Packet对、均错四格；各格动态/等权正确数，以及最终较大权重是否指向正确分支。权重方向与最终混合分类并不等价，分别保留。
4. 两路表示h64/z8、共享头单路类别概率、原始g、最终w、熵补量、margin、余弦；按sample_id/输入组/标签对齐导出`validation_cache.jsonl.gz`，便于后续离线统计而无需重复读取图。
5. 表示线性CKA、最终权重分位数与极端占比；依据`H(z)`对错统计原z代理；对真正分类概率计算多类Brier、NLL、10等宽箱ECE。它们是诊断统计，不在本轮拟合温度或校准模型。
6. 单列WebBased的验证Precision/Recall/F1、完整去向/来源混淆，重点看Recon/Benign/Spoofing；输出各类别的分支正确四格和raw/final门控选择。已知测试WebBased差约2.72个百分点，对八类测试Macro差的算术贡献约69%，仅作为描述和预先固定的检查重点，不据它选择新损失或配置。

诊断**不采集参数梯度**，报告明确标记eval/no_grad；训练probe单列门控各参数梯度与实际更新。不能将推断成功等同于训练梯度已验收，也不把非零梯度等同于可靠选择。

如果原验证混淆矩阵与保存值不一致、指标误差大于1e-10、模型状态改变或检查点文件改变，诊断直接失败，保留日志，不出具成功报告。筛选入口要求完整成功report、配对来源、缓存哈希与共用模块版本一致，不能用其他split/旧版本/测试集诊断替代。

## 服务器首先执行：下载并只做诊断

在服务器wx环境、仓库根目录运行。仅fetch并提取新增文件，不checkout/pull覆盖训练工作树。首次使用的交付提交见本次回复；下面将FETCH_HEAD解析为固定SHA，目录名记录实际版本，已有目录直接复用，不覆盖其中正在使用的工具。

```bash
. /home/nzxn/miniconda3/etc/profile.d/conda.sh && conda activate wx
cd /home/nzxn/下载/WX/load/XG-NID
git fetch origin codex/r0-edge-preparation
study_rev=$(git rev-parse FETCH_HEAD)
study_dir="/tmp/xgnid_e3_gate_$study_rev"
if [ ! -e "$study_dir" ]; then
    mkdir "$study_dir" &&
    git show "$study_rev:scripts/e3_gate_study.py" > "$study_dir/e3_gate_study.py" &&
    git show "$study_rev:scripts/diagnose_e3_gate_server.py" > "$study_dir/diagnose_e3_gate_server.py" &&
    git show "$study_rev:scripts/run_e3_gate_ablation_server.py" > "$study_dir/run_e3_gate_ablation_server.py" &&
    git show "$study_rev:configs/e3_gate_study.json" > "$study_dir/e3_gate_study.json"
fi
python -B -u "$study_dir/diagnose_e3_gate_server.py" --repo . --config "$study_dir/e3_gate_study.json" --mode check
```

先确认`metadata_checks_passed`。下面按既有共享GPU意愿显式选择物理GPU0及`--allow-shared-gpu`，保留其他任务；若选择另一张卡，显式修改索引。不声称该卡当前空闲或有资源预留，发生OOM/其他错误保留现场，无自动重试。

```bash
if [ -e outputs/e3_gate_diagnostic_b512_s42_v1 ] || [ -e outputs/e3_gate_diagnostic_b512_s42_v1.log ]; then
    echo "诊断目录或日志已存在，先核对状态，不重复启动。"
elif python -B -u "$study_dir/diagnose_e3_gate_server.py" --repo . --config "$study_dir/e3_gate_study.json" --mode check; then
    nohup python -B -u "$study_dir/diagnose_e3_gate_server.py" \
        --repo . --config "$study_dir/e3_gate_study.json" --mode diagnose \
        --physical-gpu 0 --allow-shared-gpu \
        --output outputs/e3_gate_diagnostic_b512_s42_v1 \
        > outputs/e3_gate_diagnostic_b512_s42_v1.log 2>&1 < /dev/null &
    echo "$!" > outputs/e3_gate_diagnostic_b512_s42_v1.pid
    echo "已提交诊断，请检查日志和最终状态；没有提交新训练。"
fi
tail -n 30 -F outputs/e3_gate_diagnostic_b512_s42_v1.log
```

Ctrl+C仅退出tail。结束后读取以下文件；完整report和两个压缩缓存应传回本地，不能只凭提交提示认定完成。

```bash
cat outputs/e3_gate_diagnostic_b512_s42_v1/status.json
cat outputs/e3_gate_diagnostic_b512_s42_v1/report.json
```

## 诊断通过后的E3a命令：当前不自动执行

保存同一个`study_dir`版本与配置，先确认诊断真实成功。probe只用前三个seed42训练batch，首批额外预热1步，然后三个实测更新；它不训练完整epoch或接触验证/测试图。probe通过只能证明抽样执行/容量，不能保证全数据永不OOM。

```bash
python -B -u "$study_dir/run_e3_gate_ablation_server.py" \
    --repo . --config "$study_dir/e3_gate_study.json" --mode probe --variant E3a \
    --diagnostic-report outputs/e3_gate_diagnostic_b512_s42_v1/report.json \
    --physical-gpu 0 --allow-shared-gpu --output outputs/e3_gate_probe_E3a_b512_s42_v1
```

成功应为completed/exit0、warmup_updates1、measured_updates3、optimizer_update_verified/pure_gate_verified=true、test=null。匹配同GPU UUID、共享策略、入口及诊断绑定后，才独立提交screen：

```bash
if [ -e outputs/e3_gate_screen_E3a_b512_s42_v1 ] || [ -e outputs/e3_gate_screen_E3a_b512_s42_v1.log ]; then
    echo "E3a产物已存在，检查状态，不重复启动。"
else
    nohup python -B -u "$study_dir/run_e3_gate_ablation_server.py" \
        --repo . --config "$study_dir/e3_gate_study.json" --mode screen --variant E3a \
        --diagnostic-report outputs/e3_gate_diagnostic_b512_s42_v1/report.json \
        --probe-report outputs/e3_gate_probe_E3a_b512_s42_v1/summary.json \
        --physical-gpu 0 --allow-shared-gpu --output outputs/e3_gate_screen_E3a_b512_s42_v1 \
        > outputs/e3_gate_screen_E3a_b512_s42_v1.log 2>&1 < /dev/null &
    echo "$!" > outputs/e3_gate_screen_E3a_b512_s42_v1.pid
fi
tail -n 20 -F outputs/e3_gate_screen_E3a_b512_s42_v1.log
```

根status、summary和`E3a/metrics.json`均应完成，metrics的test始终为null。按验证选模保留best/last，不报告测试分数。等权初始化版将variant改为`E3a_equal_init`，使用全新probe/screen目录，并且两个阶段都必须加：

```text
--parent-screen-report outputs/e3_gate_screen_E3a_b512_s42_v1/summary.json
```

它只把原E3a完成报告当作顺序/预算证据，不加载E3a训练权重。等权版仍从原配对初态开始，新probe不能复用E3a的probe。待C结果和预算核查后再给出该项完整实际启动命令，不自动排队。

## 预算、选择与后续方向

诊断暂限8 GPU小时；筛选一轮最多2配置、暂限8 GPU小时，沿用原计划框架。第2～3轮记录预计100轮总成本、父运行耗时和超预算提示；它是排程警示，不自动杀当前任务或把截断结果当完整比较。若E3a已用完本轮8小时，入口拒绝追加等权版。确认实验仍需先核对总预算，不凭现有单次共享耗时承诺具体完成时间或独占加速。

优先看验证Macro、逐类F1和分歧子集；如果E3a或等权初始化只有偶然单seed小幅改善，不直接声称门控贡献。最终候选与必要对照固定split补训练seed43/44，不改变样本成员，不挑最好seed。不继续根据测试分数搜索batch、学习率、门控宽度、辅助损失或阈值。

- 有互补且门控选择差：研究有明确分类监督的分支预测与可靠性输入，先做相同监督下的固定/动态配对；仍需文献差异、校准和机制证据。
- 缺少互补：优先一个分支差异化编码改动，不先增大门控网络。
- 现有简单等权仍更好：保留E2-B512作为候选主对照，负结果如实记录，不因预设论文题目强行保留门控。

本轮未实现分支监督M1、同batch单编码器容量对照、正式测试或三种子完整矩阵。E3a和等权初始化是诊断性消融，本身不自动构成新算法贡献。

## 本地验收与待服务器验收

本地已通过8项纯标准库检查，包含真实E2/E3-B512参考源码/配置/4个检查点文件哈希、两组保存验证矩阵的指标复算和WebBased验证指标。Python3.9 AST检查通过。原冻结文件无本次改动；没有导入torch、反序列化checkpoint、加载图或执行模型/训练。

机器记录：[e3_gate_study_static_checks_20261021.json](e3_gate_study_static_checks_20261021.json)。服务器元数据复核、CUDA诊断、实际模型状态/验证复现、probe梯度与更新、正式筛选均须真实回传，当前不标记通过。
