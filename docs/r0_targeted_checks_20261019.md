# R0 补充核查：数值来源与训练单步

用户要求在已有检查基础上补齐两类证据，未要求重新制定实验计划或启动完整重训练。新增独立工具`scripts/audit_r0_targeted_server.py`；冻结R0/E0～E3、已有数据/权重/结果均保持不变。工具只在Linux服务器运行，本地仅解析源码/AST和检查交付文件，未运行模型、真实数据检查或新增脚本。

## 1. 训练单步：执行实际源码而非另写一套近似循环

- 从R0 launch记录定位已通过训练的worktree，核验冻结代码、配置、官方Model/Training和附加特征函数身份、原last.pt SHA。只从训练清单选每类8个固定内部位置，共64张训练图，不依测试错误选样本。
- 使用新建的官方Edge模型和ReferenceEdge，不使用最终训练权重进行优化。两者载入完全相同的初始化参数和BN buffers；记录共享初始state摘要、样本身份摘要，并将Python/NumPy/torch CPU随机性固定为42。
- 官方路径直接编译锁定`train_with_edge_Att`函数，仅给一个batch和一个epoch。其末尾训练准确率计算被替换为不推断的占位返回，日志不输出为实验准确率；不影响已经完成的optimizer step。原官方模型loss方法保留，额外记录其真实标量。
- R0路径直接抽取原训练入口的Adam构造、model.train、计数初始化和batch循环AST，省去数据准备、记录、检查点保存及完整epoch遍历。不是另外重写一个相似的NLL循环；实际执行片段保存于新诊断目录。
- 每侧只做一次zero_grad→forward→loss→backward→Adam step。对比训练态输出、真实损失、每参数None/零/非零梯度、更新后参数及buffers、Adam步数/动量、参数组、后续torch RNG；四个BN计数应均为1。用rtol1e-6/atol1e-7比较张量，保留最大差值和更新参数数量。
- 只改变临时模型内存，不保存任何诊断权重、不恢复/覆盖正式R0 optimizer、不启动完整训练。该检查不估计多种子波动，也不证明历史作者初始化、完整采样顺序或CUDA执行与本次CPU相同。

## 2. 特征数值：最终训练CSV可核对的部分

- 只读训练候选CSV，先核对其冻结SHA；分块遍历160000行，记录82列有限/非有限值、零值、min/max，不填补、不归一化、不拟合变换。读取约1GB CSV，一次扫描用于新增数值证据，不重复完整图扫描。
- 用四个基础包长统计列独立重算packet_size_variation（pandas std，ddof=1），检查独热0/1及互斥约束、附加特征非负性、同组累计双向包数不小于累计正向包数。默认350窗口的布尔/unique计数上界只作为假设警示；0个协议/Expiration标志可能是原Categorical范围之外值，不自动判错。
- 对本次64个训练样本，从候选CSV独立解码payload字节、前1500字节截断/右侧补零、float32 Flow直接转换、Contain属性和按Link端点索引对齐delta_time；不调用官方构图辅助函数来生成比较值。零payload允许官方一零字节哨兵。比较只读原图，不生成或覆盖图文件。
- 这是最终CSV数值约束和候选行映射的核对，不是原PCAP字节或全部上游值已恢复。发现候选数字CSV行与图不匹配时也应先核对来源映射，不能直接判定图错。

## 3. 原始滚动上下文：不能由最终平衡CSV补造

用户目前不确定是否保留上游CSV。默认工具在仓库data目录中做有界、仅表头搜索，跳过实际图分片目录，不读PCAP，不查shell历史、不搜索整个home。最多300个CSV表头、深度6、10000目录；发现保留src_ip/dst_ip/时间的文件只列为候选，不自动采用。字段齐全仍不证明完整、未切分或未采样。

如之后确认一个**完整逐PCAP、采样前、含原始protocol/expiration/IP/时间列**的CSV，可显式使用`--upstream-csv /真实文件.csv`。工具按官方函数默认350执行完整文件的排序/分组/rolling/独热代码，唯一源码变化是将最后的data.to_csv替换成return data，保持原文件只读。在独立内存结果中，以不含29附加列的基础数值和packet列匹配固定训练样本，对照29个附加值；匹配冲突/多种上下文/未覆盖均单列，不能只挑一个碰巧匹配的结果。

默认单个上游文件内存保护为256MiB；更大文件需先评估服务器内存并显式调整`--max-upstream-mib`。原始函数需要完整上下文，不能只截取已经采样或删除IP/时间后的数据重算350。暂缺完整上游文件时，该项标为未执行，不能称“特征数值来源全部验证通过”。实际source文件的完整性和历史窗口仍需来源记录证明，工具不搜索窗口或按测试结果挑窗口。

## 4. 服务器命令

在服务器原仓库wx环境、新终端执行。只fetch并提取新增工具到/tmp，不pull或切换旧worktree。

```bash
. /home/nzxn/miniconda3/etc/profile.d/conda.sh && conda activate wx
cd /home/nzxn/下载/WX/load/XG-NID
git fetch origin codex/r0-edge-preparation
git show origin/codex/r0-edge-preparation:scripts/audit_r0_targeted_server.py > /tmp/xgnid_r0_targeted.py
python -B -u /tmp/xgnid_r0_targeted.py --repo .
```

默认单CPU线程、nice10、隐藏全部GPU；无需等待GPU空闲，但CSV读取仍会占用一些CPU/磁盘。若只希望先做单步，可在末行追加`--step-only`，它不会做CSV扫描/候选搜索。没有原始上下文时无需自行猜路径；默认命令会列候选并保持上游重算未完成。

输出集中到一个新`outputs/r0_targeted_audit_<UTC>/`，包含report/status、实际执行训练源码片段、数值明细和上游候选清单；只有显式重算上游时才增加upstream_replay.json。终端显示简要检查结果及真实输出路径。回传终端末尾JSON；有候选时再按记录核对是否确为采样前完整文件，不重复运行旧gap audit。

## 5. 验收与范围

本地只做Python3.9语法、原训练语句抽取/编译（不执行）和源码引用静态核查。服务器实际结果已于2026-10-19回传，详见下一节。`training_run_started=false`表示没有启动正式训练，工具**确实会对两份临时模型各做一次诊断反向及优化更新**。`test_graphs_loaded=0`、`test_predictions_read=false`，不产生新的测试集预测。

R0没有独立验证集；不能把从原train/test合并后得到的分组val直接当成R0未见的独立验证集。Recon/WebBased后续方法选择应在合适的训练/验证协议中独立登记，不能因已见测试混淆而反复选参。两项检查即使全部通过，也只减少特定实现错误的可能，不能排除全部预处理或完整训练轨迹差异。

## 6. 服务器实际结果（2026-10-19）

实际目录为`outputs/r0_targeted_audit_20261019T015739852831Z`，工具SHA256为`904219733907a02a6cbb1e363646f6fe6dd4c959121c41e26d07fb5952d154db`。用户回传的终端JSON已归档至[r0_targeted_server_evidence_20261019.json](r0_targeted_server_evidence_20261019.json)；服务器完整明细文件尚未回传，不将终端摘要冒充完整报告文件。

| 检查 | 实际结果 | 结论范围 |
| --- | --- | --- |
| 官方/R0训练单步 | 64张训练图覆盖8类；输出、loss、梯度、参数及buffers、Adam状态最大差值全部0；参数组和更新后随机状态一致 | 相同初始状态、同一批输入下，两条CPU训练路径一致 |
| 更新与BN | 46个参数张量更新；46个非零梯度、10个零梯度；4个BN计数均为1 | 诊断完成真实反向和一次更新；零梯度本身不构成错误证据 |
| 训练特征数值 | 全160000行、82列；所检查的非有限值、公式、独热、非负性和计数约束异常均为0；无全零列 | 已检查的最终CSV数值约束通过，不等于完整历史特征来源已恢复 |
| 独立图数值抽样 | 本次64张训练图的payload/Flow/边属性比较，数值及payload长度不一致均为0 | 新增训练抽样核对通过，不外推到全部图 |
| 上游CSV | 项目data目录有界搜索候选0；未执行完整rolling重算 | 不能据此称整个服务器没有上游文件，也不能证明历史窗口与上下文一致 |
| 正式产物保护 | 原checkpoint未变；未用GPU，未加载测试图或读取测试预测 | 没有重训、修改正式结果或增加测试集推断 |

诊断loss为`2.1091785430908203`，来自新初始化模型的单步，不是R0最终模型损失。总体状态`targeted_checks_complete_history_unresolved`，failed_checks和findings均为空，root_cause_established仍为false。

本轮针对性检查已完成，不重复执行。未解决项仍是完整采样前逐PCAP上下文、作者样本身份及历史处理记录；已确认的训练基础顺序与初始化时机差异尚未建立对4.18个百分点差距的因果关系。不能据此排除全部代码问题，也不据测试结果调参或自动重训。
