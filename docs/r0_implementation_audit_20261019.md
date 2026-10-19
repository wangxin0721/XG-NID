# R0 差距核查：静态结果与服务器取证

范围：仅核查R0分数差距；不修改冻结R0/E0～E3代码、数据、划分、权重或训练配置，不启动重训练，不开发M1。2026-10-19本地核查为代码/文档读取、官方缓存哈希核验和Python语法解析，未在本地运行新脚本、加载模型或真实图。SSH通过完整路径OpenSSH进行了只读连接尝试，仍在banner exchange超时；服务器动态项目仍待真实执行。

## 一、已完成的静态核对

固定官方提交仍为 `551d1f10189e5259f8f34e32f918d557a2b3a64e`。本次核验先前实际下载缓存的原始SHA256，Model、Training、Functions和两个notebook与`r0_source_lock.json`的上游SHA全部一致；没有将当前可变main当成该固定版。在线重新读取raw/API失败，未获得新的有效提交历史，不能将旧histories.json中的null字段当作历史证据。

| 项目 | 本次核对结果 | 对差距的解释边界 |
|---|---|---|
| Edge模型 | R0 `ReferenceEdge`直接实例化官方`HeteroGNN_Edge`；两层关系GAT、BN eps1、LeakyReLU、分别mean pool、128→64→16→8无头间激活 | 没有发现R0换成用户双分支或门控的偏差 |
| 边属性 | 包装器传入三种边属性，标量Link仅clone后变视图；六GAT输入及实际作用已有服务器冒烟 | 实际新检查点的独立等价性核对留给服务器 |
| 目标与优化 | 原官方loss为NLL，R0显式NLL相同；Adam0.01、weight_decay0、30轮、64/1 batch、无Edge scheduler、最后一轮 | 与固定版Edge示例路径一致；无法证明就是论文最终运行设置 |
| 标签与评价 | 0～7类别顺序一致；argmax；真值行/预测列CM；最后一轮测试一次 | 回传指标与CM算术已核对；需进一步以保存预测文件重建CM |
| **基础样本顺序** | **确认有差异**：官方Functions.get(idx)直接读`data_{idx}.pt`；R0.prepare按sorted(rglob)建立字典序清单 | 基础索引排列和随机批次轨迹会不同；尚无实验能将4.18个百分点归因于此。不能说成员因此换了；两种顺序覆盖的是同一现有源成员 |
| lazy初始化时机 | R0在创建训练DataLoader前用16张图eval前向；官方示例没有这一步，lazy层在首次训练前向才完成初始化 | 会改变随机数消费与打乱的相对时序；作者seed未知，不能据此断言分差大小 |
| 官方notebook默认路径 | GNN4ID_Model.ipynb的活动代码是`HeteroGNN`+`train`+`test_cm`；Edge类/训练/测试调用被注释 | 论文描述GAT，R0运行Edge符合本任务。公开notebook默认运行不能被直接当成生成论文97%的记录，也不应为追分擅自切SAGE |
| aggr参数 | 类构造器接收aggr="mean"但未将其传给HeteroConv，实际默认跨关系sum；R0复用同类 | 未发现R0与官方该项不同；不要仅看参数文字就认定是mean |
| MAC过滤 | 官方split_csv的Benign分支只有df[mask]表达式，没有赋回df；攻击分支赋回 | 是公开代码与论文过滤描述的偏差；现有图无MAC，尚不能确认当年CSV是否受影响 |
| temporal窗口 | 官方Additional_Features默认350，流水线调用未显式覆盖 | 这只是示例默认，尚不能证明服务器旧CSV实际采用350、或逐特征值等同作者数据 |

模型源码位置：`third_party/GNN4ID/Utility/Model.py`；训练：`Utility/Training.py`的`train_with_edge_Att`；数字索引：`Utility/Functions.py`的`get`；字典序：`scripts/r0_data.py`的`prepare`；R0预热：`scripts/run_r0_edge.py`的模型构建段。官方notebook核对使用`tmp/r0_upstream/GNN4ID_Model.ipynb`，原始SHA=`aadffb666ee9caa633fa592609f934931f372546b62735cffcc1e16fc9fbe00b`。

论文Macro-F1口径已经由期刊PDF第11页图5核实，见`r0_paper_gap_20261019.md`。当前92.8211%与论文约97%差约4.18个百分点；这次核查尚未定位唯一原因，不把执行成功当性能复现成功。

## 二、服务器实际核查工具

新增`scripts/audit_r0_gap_server.py`。通过R0 launch JSON自动定位原worktree、清单和结果，避免手工拼接异常旧路径。唯一新增产物是一个`outputs/r0_gap_audit_<UTC>/`诊断目录，包含状态、report和抽样对照，原实验文件全部只读。

1. 标准库阶段：核对最终status/summary/监督退出码、30轮日志、逐轮样本数和lr、代码/配置/清单绑定、last.pt SHA；逐行核对25557条保存预测与冻结测试清单的sample_id、label及顺序，从保存预测重建CM、Accuracy、Macro/Weighted-F1和每类指标。不重新预测测试集。
2. CPU阶段：核对候选源CSV原SHA，分块读取两个CSV确认类别数量和完整82列顺序；训练/测试各类固定取清单内第0、1/3、2/3、最后位置，共64张图，尝试`data_N`对应CSV第N行。仅对这64行用原作者辅助方法在内存中重建预期张量，逐项比较Flow、Packet、三类边索引/属性及标签；不生成.pt或改图。
3. 检查实际checkpoint内部epoch/config/binding、有限权重、Adam lr/weight_decay、75000次优化步骤及四个BN计数；CPU上加载同一权重，用**训练图**核对包装模型与原官方模型forward/NLL一致性，以及batch16与单图eval差异。没有backward或optimizer.step，不对测试图做新推断。

**证据限制：** 数字文件名与CSV行号只是假设，失败先表示“候选对应关系未成立”，不能直接判定生成代码错误；64张均匹配也不能证明185557张全量映射，更不能证明作者历史样本身份相同。MAC过滤、实际窗口和亚类比例通常需要原始提取记录，不能凭82维恢复。CPU与历史GPU存在数值实现差异，forward检查使用明确容差。

## 三、服务器命令

在原仓库wx环境、新终端执行。不会切换现有分支，不pull、不改原worktree、不安装环境，不启动E2/E3或R0。

```bash
. /home/nzxn/miniconda3/etc/profile.d/conda.sh && conda activate wx
cd /home/nzxn/下载/WX/load/XG-NID
git fetch origin codex/r0-edge-preparation
git show origin/codex/r0-edge-preparation:scripts/audit_r0_gap_server.py > /tmp/xgnid_audit_r0_gap.py
python -B -u /tmp/xgnid_audit_r0_gap.py --repo .
```

工具限单CPU线程、降低优先级、隐藏全部GPU。读取约1.2GB候选CSV并抽样加载64张图，需要等待实际完成；CSV分块阶段显示行数进度。没有完整训练集前向扫描。若E2已经在跑，CPU/磁盘读取仍可能造成争用，可只做标准库核对：在末行追加`--metadata-only`，之后另安排完整诊断。

完成后回传终端JSON，重点看`failed_checks`、`base_order_is_official_numeric`、CSV抽样matching_samples、checkpoint/BN计数、官方forward差异；完整字段在工具打印的report.json。候选映射有差异还需回传同目录csv_graph_samples.json中对应差异项。`checks_passed_with_unresolved_protocol_differences`仅表示本次所选核对通过，不表示已达到97%或已排除所有问题；`root_cause_established`保持false。

首次本地交付仅Python3.9语法解析与静态审查；后续已经收到服务器真实通过回传，范围见下节。没有在本地运行模型或诊断脚本。任何失败都保留原结果并报告，不自动修补数据、换seed或重训练。

## 四、真实服务器回传：三项检查全部通过

来源为用户完整粘贴的终端JSON；工具来自提交`1e00494`。原数值保存于`r0_gap_server_evidence_20261019.json`，完整服务器report及CSV抽样详情文件仍在服务器，并未整文件取回。诊断目录为`outputs/r0_gap_audit_20261019T011605512759Z`。

| 检查 | 真实回传 |
|---|---|
| 正式运行产物与清单 | passed；30轮、状态/配置/源码/清单绑定及最终权重SHA通过 |
| 保存预测与指标 | passed；25557条身份、标签、顺序正确，重建Macro-F1=0.9282109573752343 |
| CSV/图、checkpoint、CPU前向 | passed；训练32/32、测试32/32抽样张量匹配，候选对应差异0 |
| 优化与BN计数 | Adam state step均75000；四个BN num_batches_tracked均75000，与30×2500 batch一致 |
| 包装模型与官方模型 | 使用同一最终权重和所检训练batch，最大logprob差0.0 |
| batch与单图eval | 最大logprob差4.57763671875e-05，在工具设定rtol=atol=1e-5的联合容差内通过；不是所有测试图的数值等价证明 |
| 实际诊断环境 | CPU，Python3.9.23、torch2.3.1+cu118、PyG2.5.3、pandas2.0.3、numpy1.23.5、sklearn1.3.2 |

最终权重SHA=`7a86dbc2b7345295be9b564cb581eb9dff20227018449aaf8e1a47aeba6ff581`；预测SHA=`8c1e13b25066f4f5e9b0499f52fc6f39c6329c50730abd395f0c0e8122a6b543`。没有启动训练、没有使用GPU、没有新增测试推断。

源train/test均确认是字典序而非官方数字基础顺序。**必须区分训练与测试：** 训练基础顺序可能影响随机批次/优化轨迹；因果影响仍未测定。测试以同一权重、eval、batch1逐图推断，预测与标签已逐条对齐，仅改变样本遍历顺序不会改变这套样本上的汇总F1，不能拿测试排序解释4.18个百分点。

现阶段没有发现抽样输入被错构、包装器改变官方输出或指标算错的证据；不能扩展为所有代码/所有输入均无问题。64个匹配只证明当前候选CSV与这64张图的一致性，不能证明作者历史样本相同。时间窗口、历史过滤与亚类比例仍缺来源记录；root_cause_established=false。

无需重跑本次审查。下一步可以直接从现有`csv_graph_samples.json`提取82列顺序和temporal列清单，核实可见时间特征，再判断需要哪份原始生成记录。不要因检查通过就反复换seed或修改测试集。

## 五、用户确认：CSV由原始PCAP自行提取

用户在来源问题中明确回答“自己从原始PCAP提取生成”。这是用户报告的历史来源，不能单凭该回答推出当前文件与作者公开处理后CSV内容不同；此前按来源描述作出的这项推断已由第九节的完整下载SHA证据更正。此前64张匹配证明的是本地处理链在抽样范围内一致，不是作者论文当年样本身份已对齐，也不说明自行提取必然错误。

接下来的核查需恢复实际提取脚本/notebook及执行参数。固定官方代码可确认的对照项为：NFStream6.5.3（requirements声明），accounting_mode=1、idle_timeout=120、statistical_analysis=True、n_dissections=0、My_Custom(limit=20)；additional_features按输入CSV逐文件调用，按bidirectional_first_seen_ms排序，默认每分组350条流记录rolling。原示例先提取、增加temporal特征，再过滤/划分/采样。若实际流程改变了文件合并边界、先采样再算rolling、协议处理或流切分，最终特征可能不同，需要真实命令/源文件证据确认；这些目前都不是已确认错误。

当前安装的NFStream版本只能作为环境快照，不能替代历史提取环境。现有报告可直接读出82列顺序、temporal列和MAC列是否留存；无需重跑CPU前向、重新扫描CSV或重新训练。用户实际提取命令/脚本名称仍待提供。

## 六、用户补充及官方实际提取调用

用户进一步说明“从作者给出的代码里执行提取”，并展示服务器的`Data_preprocessing_CIC-IoT2023.ipynb`。这确认用户报告使用作者处理链，但截图仅展示当前打开的预处理说明，无法恢复实际提取版本、环境、命令、输入PCAP列表和成功日志。不得将“自行从PCAP提取”直接写成不符合官方处理方法。

本次再次读取已核验SHA的官方固定版`GNN4ID.ipynb`全部代码单元：

- cell11指定`feature_Extractor='Utility/Feature_extractor_flow_packet_combined.py'`，对每个PCAP调用`subprocess.run(['python', feature_Extractor, single_pcap_file, Out_path], capture_output=True)`。脚本CLI有两个位置参数：pcap_files和Destination_path，后者在代码中与输出文件名直接拼接，应保留目录尾部分隔符。
- cell13对每个提取CSV执行`additional_features(file)`，使用默认window_size=350；实际rolling对象是分组内的流记录条数。
- 随后按README建议执行`Data_preprocessing_CIC-IoT2023.ipynb`中的`split_csv(files)`、`Combining_classes(...)`、列删除与训练/测试CSV聚合；最后回到构图调用`NIDSDataset(...single_file=True)`。
- NFStream参数及声明版本见第五节，均直接在官方脚本/requirements读取，不把当前安装元数据等同历史运行元数据。

新增代码级发现：官方cell11在subprocess.run之后无条件调用`os.remove(single_pcap_file)`，没有先核对returncode，且capture_output会收集输出。这可能让提取失败不易从notebook表面发现，并删除原PCAP；**尚无证据表明用户实际发生过这种失败**，不能直接用它解释F1。不要为核查重新执行官方整本notebook，以免改变原数据。

另一个待追溯的平台细节：官方name_mapping将BrowserHijacking映射为`Webbased-BrwserHijack`（小写b），而后续主类使用`WebBased`；文件路径还包含Windows反斜线拆分和glob。Linux移植时需要核对实际修订及输入文件覆盖范围。当前CSV数量匹配表4、64张样本匹配并不能恢复所有亚类来源；这仍是待证实项，不是已定位根因。

下一步只读取已有报告中的82列及temporal列，并结合用户保留的原提取记录确认执行顺序；不重复当前已通过的产物/前向核查，不自动重训练。

## 七、特征列真实回传：附加列名称和顺序核对通过

用户执行只读命令，从上述服务器诊断目录的`csv_graph_samples.json`提取train的完整flow_feature_order、temporal_columns和mac_columns_present。回传共82个Flow列，分解为46个基础统计列、1个packet_size_variation、28个rolling/分组附加列、7个独热列。工具的temporal_columns按列名前缀筛选，也包含packet_size_variation；不能将其称为29个全是时间窗口特征。

本地仅对字符串及官方源码AST核对：回传29个附加列的**名称和顺序**与锁定官方Additional_Features.py一致。缓存与vendored该文件字节相同，SHA256为`d40fd8b3bb834cb6aefe0641bc41e1f6a72b70cad24dff83e4e6620f9f42c07f`。末尾Exp_0、Exp_-1、proto_1、proto_2、proto_6、proto_17、proto_58与公开默认类别顺序一致。先前服务器CPU核查已要求train/test的完整82列顺序相同并通过，无需再重复读取测试列。

mac_columns_present=[]符合官方预处理先过滤、再删除src_mac/dst_mac等标识列的输出结构，不能据此推断未做MAC过滤，也不能反向证明过滤确实按论文执行。29个附加列齐全不支持“完全漏掉时间特征列”的解释；列名存在及64张抽样值匹配仍不能证明历史window_size、逐文件计算范围或所有行的值与作者实验完全一致。

这项列结构核对已完成，无需再要求用户重复回传表头或重跑完整审查。当前已检查范围内未发现可直接修复的输入/封装/评价错误；论文分数差距的根因仍未确定。保持原始结果和冻结训练不变，后续需要作者样本/历史生成记录，或另行登记的独立受控实验来检验已确认的训练顺序与初始化时机差异，不根据测试反馈反复调参。

## 八、服务器补充核查：训练单步及数值约束通过

2026-10-19用户回传`outputs/r0_targeted_audit_20261019T015739852831Z`终端JSON，已归档为[r0_targeted_server_evidence_20261019.json](r0_targeted_server_evidence_20261019.json)。实际结果和限制见[r0_targeted_checks_20261019.md](r0_targeted_checks_20261019.md)第六节。

官方与R0实际训练源码在共享新初始权重、同一批64张训练图下，训练输出、损失、梯度、更新后参数/BN和Adam状态最大差值全部0。全160000行训练CSV的82列约束检查无异常，新增64张训练图的独立数值比较无不一致。原正式checkpoint未改变，没有GPU训练或测试集推断。

这进一步降低了模型封装、该受控单步训练及已检查特征数值错误的可能性；并不证明完整训练轨迹或历史预处理与作者相同。有界data目录搜索未找到上游候选，完整rolling重算未执行；不能推断全服务器文件不存在。当前没有发现需要修复的确定实现错误，根因仍未定位，不自动改变协议或重训。

## 九、作者公开CSV完整下载：内容身份已获得更强证据

用户要求检查本地压缩包并寻找满足作者需求的数据。实际ZIP仅包含47列统计CSV，缺少逐包payload，不能用于官方图。随后成功取得固定README链接的公开Google Drive目录，并完整下载作者df_class_8_train.csv和df_class_8_test.csv至独立本地目录。完整来源和下载证据见[author_data_rebuild_20261019.md](author_data_rebuild_20261019.md)及[xgnid_author_data_download_20261019.json](xgnid_author_data_download_20261019.json)。

实际完整SHA分别为train `1b546e6c69faad482b98b4c95219ef9b13aba0c13afcfe0b4abe69c1e7f38b02`、test `08b256997d0368dde6c85a3b43ef3d56b4ce336939fcdce41596725bbdee42ed`，均与此前服务器回传SHA完全一致。两份现有文件与作者现公开文件内容相同，不再仅标为列名/大小相似。此结论没有恢复历史生成/下载行为，未本轮直接访问服务器重算当前文件，也不等于绑定作者论文当年检查点。

R0-A已由用户停止（训练PID369197退出，监督退出码143），保留原文件，不作为完成30轮结果。用户已授权以作者公开CSV在新目录全量重新生图；工具已准备，服务器实际生成和逐张新旧图比较待执行，不自动恢复R0-A或开始训练。
