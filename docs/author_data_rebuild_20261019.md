# 作者公开数据核查及独立全量生图

> 服务器在线v1已回传Google Drive连接超时，失败在CSV获取阶段。现在优先使用末节的`--offline`命令：重新核验现有服务器CSV与作者完整下载SHA，再复制到独立v2目录生成；不重跑在线v1、不清理失败证据。

## 本地压缩包实际检查

`D:/XG-NID/data/CICIoT2023.zip`大小500559747字节，SHA256为`793b212d1dc40973068d06cec4d0676eaf7f17436a859dfa522303e199c7ea5a`。只读取ZIP目录、三份CSV表头和首条标签，没有解压、全表处理或本地生图。完整清单见[ciciot2023_archive_inventory_20261019.json](ciciot2023_archive_inventory_20261019.json)。

压缩包只包含`CICIOT23/train/train.csv`、`CICIOT23/test/test.csv`、`CICIOT23/validation/validation.csv`，解压内容共2319327087字节。每份47列，包含小写label及46个统计特征，如flow_duration、Header_Length、Protocol Type、Tot size、IAT等；没有PCAP、udps.payload_data、逐包delta_time/方向/长度，也没有原IP/时间上下文。

结论：不能确认此ZIP的原始下载来源或具体抽样版本；它不符合XG-NID官方Flow+Packet构图输入。不能凭CICIoT2023命名认定就是作者训练输入，也不能从这些统计量恢复payload，更不能补零伪造Packet节点。原压缩包保留，不处理为R0图。

## 找到并核实的作者公开入口

- 固定源码：作者[GNN4ID仓库](https://github.com/Yasir-ali-farrukh/GNN4ID/tree/551d1f10189e5259f8f34e32f918d557a2b3a64e)，commit `551d1f10189e5259f8f34e32f918d557a2b3a64e`。
- 2026-10-19用HTTP读取固定README成功；README明确链接[CIC_IoT2023_Processed_Data](https://drive.google.com/drive/folders/1FiZh87vvCZF3gX1Fnj9iTB4j74u-nuR6)。这次实际取得公开目录HTML，不是仅推测链接可用。
- 目录公开文件为train的`1IClFYOup0U58WKj2R0p1IJ_ZMc1Os2FD`（1027673808字节）和test的`1Yxpd-ruxC8B1zz3e8oLOcZcfrahIfhJc`（175063419字节），名称为df_class_8_train.csv、df_class_8_test.csv。
- 已实际用HTTP Range读取公开test CSV开头，返回206和Content-Range总长175063419，确认NFStream基础统计及udps.payload_data、delta_time、方向/包长等列。两份全量下载及其SHA以download.json为真实依据，不以目录大小相同代替内容相同。
- 原论文[数据处理描述](https://arxiv.org/html/2408.16021v2#S3.SS5.SSS1)和[CIC官方页面](https://www.unb.ca/cic/datasets/iotdataset-2023.html)区分原始PCAP和统计CSV。作者图需要其Flow+Packet提取结果；通用统计CSV不能替代。

作者公开数据已经完成其预处理，下载后不再重复采样、重新划分或再次计算rolling。此路线是“作者公开处理后数据→官方图”，不是“原始PCAP提取全流程重做”；原作者论文当年的精确成员绑定仍不能单凭公开文件名认定。

## 工具与输出隔离

`scripts/rebuild_author_graphs_server.py`有两种明确分离的操作：

1. `--download-only`：允许本地，仅下载到新`data/raw/xgnid_author_published_v1/`，记录来源、字节数、SHA256和表头。不导入torch、不运行模型或构图。用户已明确授权寻找并下载合适数据。原ZIP/旧CSV/旧图保留。
2. 默认服务器模式：从公开链接重新下载到`data/processed/xgnid_author_rebuild_v1/raw/`，强制与此次完整下载所固定的SHA相同，生成独立`processed/`下全部185557张单HeteroData图；作业状态/源码/逐图清单/日志/PID/exitcode集中在`outputs/r0_graph_rebuild_author_v1/`。不把大型数据加入Git，不替换冻结R0/E0～E3，也不自动训练。

服务器通过原R0 launch定位冻结worktree，校验官方Functions及原始R0源码/清单/权重和环境。从锁定Functions中直接提取NIDSDataset类，原process及六个特征/边辅助方法不改。为限制内存，只将pd.read_csv返回值替换为连续1024行块；该构图阶段完全逐行、不做分组/rolling/采样，因此保持行序和逐行输入。跳过Dataset初始化及末尾全目录glob计数，仅替换保存函数以观测新输出；所有flags=False/payload=True/single_file=True、20包既有输入、ToUndirected、float32/标量Link行为均沿用作者代码。实际类源码保存于作业目录。

每张图检查82/1500维、1～20包、标签、边方向/属性等；保存到新文件后重新加载核验张量指纹/标签，保存文件SHA。按CSV行号与旧冻结编号清单比较Edge/NoEdge输入及标签，记录全部差异而不筛选。下载文件SHA同时与旧候选CSV SHA比较；只有真实下载内容相同才能记为字节相同。相同则不应再把两者文件内容差异当成F1原因；不相同则保留新旧数据供分析。

正式完成条件：train160000、test25557、逐类符合表4、全部序列化检查通过、185557张新图和清单齐全、输入文件和原权重未改变。与旧图存在差异会完整报告，不静默修补；即使新旧全部相同，也只是构图一致性证据，不保证97%。

工具仅CPU、1线程，隐藏GPU；下载阶段显示字节数，生成每1000张显示速度和剩余时间估计。启动前检查约40GB保守余量及inode；这是防止生成途中空间不足的预算，不是实际大小预估。HTTP下载以16MiB范围分段，校验Content-Range；网络故障最多连续4次续传尝试。生成失败保留部分新文件及error，不自动重试/续跑生成、不清理旧数据。本地下载的显式`--resume-download`仅续传本工具download-only目录，不能用来恢复训练或构图任务。

## 服务器启动命令

R0-A已收到用户停止回传：PID369197退出，监督退出码143。不能将其列为完整30轮结果；已保存文件保留。新任务不恢复或重启它。

```bash
. /home/nzxn/miniconda3/etc/profile.d/conda.sh && conda activate wx
cd /home/nzxn/下载/WX/load/XG-NID
git fetch origin codex/r0-edge-preparation
git show origin/codex/r0-edge-preparation:scripts/rebuild_author_graphs_server.py > /tmp/xgnid_rebuild_author.py
python -B -u /tmp/xgnid_rebuild_author.py --repo .
```

默认命令提交后台监督和CPU worker；返回submission仅为提交，不是已生成。工具复制入口到新作业目录固定本次版本，终端关闭不影响已提交进程。拒绝已有作业/新数据目录，避免重复启动和覆盖。

```bash
tail -n 30 -f outputs/r0_graph_rebuild_author_v1/run.log
```

Ctrl+C仅停止查看。状态和完成结果：

```bash
python -B /tmp/xgnid_rebuild_author.py --repo . --status
cat outputs/r0_graph_rebuild_author_v1/exitcode
cat outputs/r0_graph_rebuild_author_v1/summary.json
```

exitcode和summary在结束后才出现。回传状态/最终summary，重点核对download_sources的两个SHA与旧候选是否相同、全部旧图输入/标签是否相同，以及差异计数。完成后再决定新数据的独立清单与训练接入，不将冻结实验自动改为新图。

## 当前验收范围

本地已做ZIP身份/表头检查、公开来源HTTP核验、工具Python3.9 AST/编译及锁定官方类结构静态核查。本地数据处理/构图/训练未执行；服务器全量下载、生图、逐图比较均待真实回传。下载专用模式的实际结果另以本地download.json和聊天交付记录为准。

## 本地真实下载完成及对旧CSV身份的更新

2026-10-19已实际完成两份全量HTTP下载，保存于`D:/XG-NID/data/raw/xgnid_author_published_v1/`，来源/响应元数据/完整表头及SHA记录在该目录的download.json。首次流式下载因网络超时中断，随后通过校验Range的分段续传完成；没有仅凭文件存在就记录成功。证据摘要已归档为[xgnid_author_data_download_20261019.json](xgnid_author_data_download_20261019.json)。

| 文件 | 实际字节数 | 实际完整SHA256 |
| --- | --- | --- |
| df_class_8_train.csv | 1027673808 | 1b546e6c69faad482b98b4c95219ef9b13aba0c13afcfe0b4abe69c1e7f38b02 |
| df_class_8_test.csv | 175063419 | 08b256997d0368dde6c85a3b43ef3d56b4ce336939fcdce41596725bbdee42ed |

两份实际文件均97列，含82个Flow输入、14个逐包字段和Label；列顺序一致。这里仅核对下载字节和表头，没有在本地全表预处理、全量数值检查、生图或训练。

**两份下载SHA分别与原服务器回传候选CSV SHA完全相同。** 这比此前“名称/大小相同”更强：新公开下载内容与此前服务器记录的文件逐字节一致。此前用户报告其CSV自行从PCAP提取；现在应区分“用户报告的历史来源”和“现有文件可核实的内容身份”，不再将与作者现公开CSV内容不同作为既有证据支持的原因。不能据此反推实际历史下载/提取行为，也不能证明服务器文件此刻未被后续修改；服务器工具会重新核验。

这项证据更新降低了CSV版本内容差异的可能，但未证明全部旧图与CSV逐行相同，也未绑定论文当年97%的具体输入和训练记录。全量重生图会补齐现公开CSV到185557张图的可追踪链路，不会自动改变既有F1结论。

## 服务器网络超时：使用严格哈希验证的离线v2

用户回传v1的run.log尾部：worker在download调用urllib连接Google Drive时出现`socket.timeout`/`URLError: timed out`，发生于两份CSV获取循环、进入任何构图之前。完整status/exitcode未回传，不填造监督状态或完成报告。失败v1的日志及部分文件原样保留。

已在本地完整下载作者文件并固定两个SHA，因此服务器不必再联网：`--offline`沿原R0 source_evidence定位现有服务器CSV，逐文件重新计算完整SHA；必须等于本次作者下载的固定SHA且字节数正确，才以独占创建方式复制到全新目录。复制时核对源大小/mtime稳定，复制后重新核验目标完整SHA。不匹配立即失败，不补值、不切换来源、不自动联网降级。

离线v2使用`outputs/r0_graph_rebuild_author_offline_v2/`和`data/processed/xgnid_author_rebuild_offline_v2/`，与失败v1及旧图完全分离。除输入获取改为严格验证复制外，官方构图、全量文件检查、新旧逐图比较及不自动训练规则保持相同。不是在失败目录中自动续跑。

若终端仍在tail旧日志，先按Ctrl+C退出查看，再在wx环境原仓库执行：

```bash
cd /home/nzxn/下载/WX/load/XG-NID
git fetch origin codex/r0-edge-preparation
git show origin/codex/r0-edge-preparation:scripts/rebuild_author_graphs_server.py > /tmp/xgnid_rebuild_author.py
python -B -u /tmp/xgnid_rebuild_author.py --repo . --offline
```

```bash
tail -n 30 -f outputs/r0_graph_rebuild_author_offline_v2/run.log
```

```bash
python -B /tmp/xgnid_rebuild_author.py --repo . --offline --status
cat outputs/r0_graph_rebuild_author_offline_v2/summary.json
```

本地已完成Python3.9语法/编译，以及只用微小临时文本的标准库I/O验证：正确SHA复制成功、错误SHA在写目标前被拒绝、既有目标拒绝覆盖、v1/v2路径隔离。没有处理真实CSV、加载图/模型或运行构图。服务器离线输入校验和全量生图仍待真实回传。
