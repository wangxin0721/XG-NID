# 本地提交与服务器核查

目标是先核实服务器实际数据能否支撑正式实验。本轮不登录服务器、不自动提交或推送GitHub、不运行本地训练、不改写历史数据/划分。服务器可能有本地缺失的图、原始数据和身份清单，不能用本地结论替代服务器检查。

## 1. 本地交付范围

`scripts/package_edge_delivery.py` 中的 FILES 是明确的18文件清单，包含受控模型、构图、核查/试运行工具及本轮文档。`src/xgnid/__init__.py` 和参考 `third_party/GNN4ID/Utility/Model.py` 是运行依赖，也打入源码包；它们没有本轮实质修改。没有打包数据、权重、历史结果、报告Word、图纸或其他既有改动。首次交付zip包含当时的14个文件；后续增加了图内容审查、分组划分及其测试脚本，重新打包时使用新输出目录。

当前工作区有与本轮无关的删除和修改，且部分 outputs 文件以前已被Git跟踪；`.gitignore` 不会阻止已跟踪文件被提交。不要使用 `git add .` 或 `git add -A`。

PowerShell，在本地仓库根目录执行；输出目录每次使用新名字：

```powershell
python scripts/package_edge_delivery.py --output outputs/edge_delivery_20261016
git diff -- src/xgnid/model.py src/xgnid/graph_building.py
git diff --cached --name-status
git add --pathspec-from-file=outputs/edge_delivery_20261016/git-files.txt
git diff --cached --name-status
git diff --cached --stat
git diff --cached --check
```

提交前确认暂存区只有清单内文件；如果原先已经暂存了其他工作，先保留并单独处理它们，不要混入本次提交。本轮只准备命令，没有操作暂存区。

确认本轮暂存内容后，按原有流程提交/推送当前开发分支：

```powershell
git commit -m "Add controlled Edge experiments and server data audit"
git push
git rev-parse HEAD
```

记录最后的提交SHA，供服务器核对。打包目录中的manifest记录的是打包时HEAD和各文件实际SHA256，并不声称HEAD已经包含未提交代码；真正上传后以新提交SHA为准。源码zip是独立备份/审阅材料，无需提交GitHub。

## 2. 服务器拉取

以下为Linux Bash，在服务器真实仓库根目录执行。先激活你一直使用的训练环境。不要照搬本地Windows的Python路径，也不要直接执行旧bootstrap或安装/替换CUDA依赖。

```bash
git status --short
git branch --show-current
git pull --ff-only
git rev-parse HEAD
python --version
```

拉取前先确认服务器工作区没有需要保留的未提交代码；如有冲突先处理自己的服务器改动，不使用reset --hard。确认HEAD与本地刚推送的提交一致，之后再核查数据。这里不假定分支名，也不改变服务器现有分支。

## 3. 先盘点，后选择待查数据

```bash
RUN="outputs/server_edge_check_$(date -u +%Y%m%dT%H%M%SZ)"
mkdir "$RUN"
python scripts/check_edge_server.py \
  --data-root data \
  --check-runtime \
  --output "$RUN/inventory.json" > "$RUN/inventory.log" 2>&1
RC=$?
printf '%s\n' "$RC" > "$RUN/inventory.exitcode"
cat "$RUN/inventory.log"
```

如果数据在仓库外，用真实路径替换 `data`；也可以重复 `--data-root` 检查多个已知数据目录。不要把根目录 `/` 作为搜索范围。

`inventory.json`包含解释器/包版本、CUDA可用性和GPU名称、CSV实际表头、图文件数量与大小、若干图路径、原始pcap和身份manifest候选文件。默认不反序列化任何.pt；不检查全部图是否可用。inventory退出0只表示所请求检查已执行，不表示正式训练数据已验收。缺失目录会明确记录exists=false，必须人工判断是否漏填路径。

可选：针对以前真正训练使用的split检查本机路径和索引交叉：

```bash
python scripts/check_edge_server.py \
  --data-root /实际数据目录 \
  --split /实际旧实验目录/split.json \
  --output "$RUN/split_check.json" > "$RUN/split_check.log" 2>&1
RC=$?
printf '%s\n' "$RC" > "$RUN/split_check.exitcode"
```

只检查显式指定的split，不自动采用最新或任意split。索引无交叉不代表流无交叉，也不能证明索引仍对应当前图顺序。旧绝对路径不存在时只报告，不擅自替换路径。

可选：从盘点结果中选服务器自己生成、确认可信的单个图文件检查。此步骤会加载该整个文件，只限制检查其中前8张图；大型合并文件需留意内存，优先选单图/小分片。不会把同目录权重当图扫描加载。

```bash
python scripts/check_edge_server.py \
  --data-root /实际数据目录 \
  --trusted-graph-file /实际图目录/某个图文件.pt \
  --graph-limit 8 --check-runtime \
  --output "$RUN/graph_check.json" > "$RUN/graph_check.log" 2>&1
RC=$?
printf '%s\n' "$RC" > "$RUN/graph_check.exitcode"
```

当前合同检查对应本轮关系命名和4/4/1维属性。旧图关系名、维度或容器不同将如实报告，不自动转换；需要根据原构图代码确认含义后再适配。该工具不进行模型前向或训练。

## 4. 对服务器实际CSV重新全量审查

选定服务器真正用于训练/测试的两个CSV，将训练文件放在第一个参数。不要直接用猜测的文件名。此命令只读源文件，输出到新报告路径，不依赖torch/pandas：

```bash
python scripts/audit_edge_data.py \
  /实际数据目录/训练.csv \
  /实际数据目录/测试.csv \
  --output "$RUN/source_audit.json" > "$RUN/source_audit.log" 2>&1
RC=$?
printf '%s\n' "$RC" > "$RUN/source_audit.exitcode"
```

完整扫描耗时和内存取决于数据量；保留日志和退出状态。字段缺失会报错，不填造边属性。额外非数值身份字段会在数值检查中报告，需要明确区分身份元数据与模型特征后再构图，不能盲目丢弃原始身份列。

重点查看：

- `sources[].sha256/rows/label_counts/columns`：与实际图来源是否一致，而不是只比较文件名。
- `sources[].invalid_rows`及examples：报文列表长度、数值和payload是否有效。
- `sources[].within_file_duplicate_rows`：文件自身重复数量。
- `cross_file_duplicate_rows`：后输入文件中与先输入文件精确重复的行数量，包含标签。
- `cross_file_duplicate_features`：排除Label后，后输入文件中与先输入文件重复的特征行数量；`conflicting_label_rows`为与首见相同特征存在标签冲突的后续记录数。

哈希基于解析后的CSV字符串单元格，不做数值格式归一化，故1与1.0可能被视为不同。这是精确重复诊断，不能证明无同流或近重复泄漏。`errors.duplicate_rows`为兼容旧报告保留，包含先前文件重复与本文件重复，不能直接等同文件内重复。

本地旧审查记录为train160000、test25557、跨文件完整重复3346。它只对应文档记录的两个本地SHA256；服务器文件哈希不同就不能沿用3346结论。即使服务器没有精确重复，仍需身份/分组证据。

## 5. 是否可以进入正式实验

以下问题均有可核验答案后才冻结正式实验：

1. 使用哪一份CSV/图数据，来源、标签映射、构图脚本和SHA256是什么？
2. 有没有capture/flow身份、时间和原始划分清单？同一流及其报文是否跨集？身份列存在不等于分组划分已正确执行。
3. train/val/test的精确重复、排除标签后的重复、冲突标签各有多少？既有平衡/过采样是否发生在划分之前？
4. 旧split路径、图顺序、样本ID与实际数据能否一一对应？
5. 三类边的索引和4/4/1维属性是否齐全，边属性方向是否有明确来源？
6. 服务器环境是否能导入torch/PyG并执行后续真实图前向？

若身份或重复问题不能排除，先交付诊断和重建分组划分方案，不启动正式性能对比。保留旧实验，用独立目录建立新基准，不宣称严格复现旧结果。

如果只缺.pt但CSV字段完整且身份问题可处理，再生成服务器新图并冻结manifest。当前 `run_edge_chain.py prepare/train` 是256图级别的CPU试运行入口，**不是服务器正式GPU全量训练入口**；不要简单把epochs改成100就当正式实验。下一步需根据服务器实际图容器和数据身份，适配固定split和device入口，再按E0→E1→E2→E3顺序执行冒烟及正式训练。

## 6. 取回哪些文件继续审查

取回本次RUN目录中的inventory/source_audit、可选split_check/graph_check报告，以及对应log和exitcode。无需传回大型CSV/.pt或权重。提供服务器使用的数据根目录、训练环境名和提交SHA即可结合报告决定数据是否可用。本地生成的核查结果只能用于验证脚本本身，不能充当服务器结果。

本轮本地轻量验证：两个回归测试通过（精确重复/特征重复/冲突标签计数、拒绝覆盖、旧split重叠及无效索引），标准库盘点成功，已有小图文件前2图检查和环境导入成功；未启动新训练或重跑完整CSV扫描。服务器检查尚未执行。

## 7. 对已找到的历史split检查实际图内容重复

服务器后续回传确认：三个基础基线均记录185557图，train/val/test为144000/16000/25557，索引合法且不相交；GNN4ID_paper_align与GNN4ID的split SHA256相同；rerun文件SHA不同，尚不能据此判断索引是否相同。三个数据路径都在服务器存在，均未保存样本身份。

新增 `audit_graph_split.py` 按仓库当前loader规则（递归字典序文件、容器内图顺序）逐文件读取，写SQLite身份清单，不把全部图同时加载到内存；单个合并容器仍须整体加载，当前服务器主要是单图小分片。工具只反序列化显式指定的可信目录里的图，最多解析3层目录内路径引用，拒绝读取目录外引用；不修改图、标签、划分或模型，不训练，不使用GPU。

它分别比较“节点+拓扑+边属性”和“节点+拓扑”两种精确张量指纹，覆盖Edge和No Edge可观察到的输入。标签不参与指纹，冲突标签另报。输出每对数据集共有的输入组数及两侧受影响样本数。空Link形状保持原样，仅统计，不修复。边属性结构异常另列graph_issues。成功退出只表示内容审查执行成功，不表示数据没有重复。

即使重建的图数量、类别分布与记录一致，也无法单凭这些证明历史图顺序完全未变；报告明确标记historical_order_verified=false。结论均附带“按当前loader重建的索引映射”，不能冒称已经精确还原当时实验。精确重复检测也不等于原始流分组检查。

在服务器拉取本次更新后，先检查三个split索引是否完全相同（文件内容SHA不同可能只是路径字段不同）：

```bash
python -c 'import json; from pathlib import Path; paths=["outputs/baselines/GNN4ID_paper_align/split.json","outputs/baseline_rerun_paper_align_v2/split.json","outputs/GNN4ID/split.json"]; records=[json.loads(Path(p).read_text()) for p in paths]; print({p:records[i]["split_indices"]==records[0]["split_indices"] for i,p in enumerate(paths)})'
```

然后只扫描一个代表划分，避免重复读取约12GB图。若三个索引不同，首次报告只适用于指定split：

```bash
python -u scripts/audit_graph_split.py \
  --trusted-data data/processed/CICIoT2023_processed/processed \
  --split outputs/baselines/GNN4ID_paper_align/split.json \
  --output outputs/server_graph_split_audit_v1 \
  > outputs/server_graph_split_audit_v1.log 2>&1
RC=$?
printf '%s\n' "$RC" > outputs/server_graph_split_audit_v1.exitcode
tail -n 60 outputs/server_graph_split_audit_v1.log
```

这是完整磁盘读取，可能持续较长时间；每2000个文件记录进度。请等待命令结束；如需另开终端观察，可用 `tail -f outputs/server_graph_split_audit_v1.log`。保持原始图目录和split在扫描期间不变。取回 `outputs/server_graph_split_audit_v1/report.json`；失败时取回日志及该目录下status.json。SQLite是服务器本地身份清单，无需上传GitHub或发送整个文件。

## 8. 首次全图审查结论与标量Link兼容更正

服务器返回185557图，全部Flow82维、Packet1500维float32；数量、索引范围、完整分区和各类数均与记录匹配。pre_filter.pt/pre_transform.pt是元数据字符串，按旧loader规则跳过，不是缺失图。

按当前loader映射：Edge输入有4775个测试样本与训练重复（18.68%）、5564个验证样本与训练重复（34.78%）；No Edge对应4783、5570。测试还与验证有重复，不能直接将两项受影响样本数相加。Edge分区内重复行train40587、val1708、test1851。这些计数只描述当前重建划分，不将historical_order_verified=false改写为已复现历史。

首次工具将非空Link属性149124图计为invalid，另36433图为空Link `(0,)`。该分类过于严格：vendored Functions.py 的 `_get_link_edge_features` 返回一维delta_time数组，而服务器PyG2.5.3的GATConv.edge_update明确支持一维属性并内部view(-1,1)。因此不能仅凭该分类认定图损坏。已修正检查器和受控模型：接受有限的 `[E]` 标量Link（包括E=0），保留其他关系4维要求；自定义encoder仅在forward创建 `[E,1]` 视图，不写回图，不改变数值或连接。长度错误、非有限值仍拒绝。原报告保留，不能在未重查的情况下将149124个旧错误全部声称为通过。

API依据：[PyG2.5.3 GATConv源码](https://pytorch-geometric.readthedocs.io/en/2.5.3/_modules/torch_geometric/nn/conv/gat_conv.html)、[ToUndirected源码](https://pytorch-geometric.readthedocs.io/en/2.5.3/_modules/torch_geometric/transforms/to_undirected.html)。本地PyG2.6.1的轻量测试验证原始参考模型和自定义分支模型的一维/二维Link预测逐项一致，且原图属性维度保持不变；尚未替代服务器PyG2.5.3实际模型冒烟。

此外，旧构图调用ToUndirected()，会使同类型Packet Link产生反向边；原本仅有向链上的GAT单入边分析不能直接推广至此旧图。先轻量抽查真实旧图的标量属性、反向边和入度；不必为了初步定位再次完整扫描14分钟。

```bash
python scripts/check_edge_server.py \
  --data-root data/processed/CICIoT2023_processed/train \
  --trusted-graph-directory data/processed/CICIoT2023_processed/processed \
  --graph-limit 32 --check-runtime \
  --output outputs/server_graph_compat_check_v2.json
```

上例用已知train/子目录做轻量CSV表头盘点，同时从显式指定可信processed目录按字典序抽查前32张图（不把pre_filter/pre_transform当图）。不代表全数据验收；结果含link_topology的边数、具反向的唯一边数、最大入度。取回该JSON。目录抽样不跟随路径引用；若遇到不支持容器如实报错，不隐式改写或转换。

下一正式基准应以本次SQLite身份清单为起点，在冻结当前图版本和路径/顺序清单后，以No Edge输入指纹分组、检查组标签冲突并统计每类独立组数，再决定可行的分层train/val/test数量。如此可使两种输入模式的精确重复均不跨集；验证/测试中重复输入如何计权或去重也应事先固定。不能先保留旧Table4测试数量再为凑数制造重复，更不能声称输入去重解决了缺失原始流身份的问题。旧图和旧split保持不变，新划分独立存放。取得原始流/capture身份仍是更强的分组依据。

## 9. 冻结按输入分组的新基准

服务器32图抽查全部passed，包含单报文、双报文和多报文；非空Link确有反向边，部分节点最大入度2。该结论是格式抽查，不是全图或模型前向验收。

服务器SQLite统计如下，括号中是Edge独立输入数：Benign23890（23890）、WebBased5447（5447）、Spoofing17392（17392）、Recon22780（22780）、Mirai24000（24000）、Dos14752（14752）、DDos23866（23918）、BruteForce2336（2336）。合计134463个No Edge组、134515个Edge输入；其余51042行为完全相同Edge输入的副本。目录内存在原始过采样痕迹，不能再把20000张图等同于20000个独立输入。

`build_grouped_split.py`仅使用标准库，以只读模式打开SQLite，验证其SHA256与完成的审查报告一致，并拒绝同No Edge组标签冲突。每个Edge输入选最小audit_index作代表；保留52个边属性变体，但它们与对应No Edge组必须同集。按类别将No Edge组排序、以seed42+label打乱，验证/测试各取floor(10%)组，剩余训练。各类有足够组，得到约80/10/10的分组划分；比例按组计算，样本比例因边属性变体略有差别。所有模型共用相同样本清单，不做训练重采样或类别加权。

输出独立目录：samples.jsonl（稳定sample_id、group_id、partition、相对图路径、容器内offset、原audit索引、两种输入哈希）、split.json（协议、来源、split ID、清单哈希、类别分布和不重叠断言）、status.json。不复制或重建.pt，不覆盖源CSV/图/旧split，也不读取所有图。split ID由冻结协议、样本清单和源SQLite哈希生成。

```bash
git pull --ff-only
python scripts/build_grouped_split.py \
  --manifest outputs/server_graph_split_audit_v1/graph_manifest.sqlite \
  --audit-report outputs/server_graph_split_audit_v1/report.json \
  --seed 42 --val-ratio 0.1 --test-ratio 0.1 \
  --output data/processed/ciciot2023_grouped_input_v1
```

预期保留134515个Edge输入，组成134463个No Edge组；实际分区和每类数量以新split.json为准。运行后先回传终端摘要或该split.json，不需要回传较大的samples.jsonl或SQLite。该命令仅建立元数据划分，不进行训练、图加载或模型优化。

这是将现有图集合重新建立的新基准，合并使用已有源train/test图，不能称为旧Table4划分或原实验严格复现。原始流/capture独立性仍未知。新协议按Edge独立输入等权评价，同一No Edge组中的不同Edge变体可贡献多个样本，但不得跨集；该计权方式需在论文中写明。

新split不提供旧split_indices，避免被旧CLI误当成185557图字典序索引使用；samples.jsonl使用显式路径+容器offset，不依赖旧排序的可信性。接下来的正式服务器入口必须读取此清单、核验取出的图输入哈希和标签，再执行GPU冒烟与训练。不要直接套用旧训练命令或CPU pilot入口。
