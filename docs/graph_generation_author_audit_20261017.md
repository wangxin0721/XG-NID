# 图生成代码与原作者实现逐项核查（2026-10-17）

本次仅静态读取、逐文件比较和调用链审查，没有构图、导入/运行模型、训练、重切数据或改写已有源码。服务器正在使用的图和R0/E0～E3代码不变。

## 结论

仓库有两套不同构图路径，不能笼统回答“整个项目的构图代码都是原作者的”。

1. `third_party/GNN4ID/Utility/Functions.py`中的`NIDSDataset`是原作者公开实现：本次从官方固定提交重新下载，统一CRLF/LF换行后全文相同。
2. `src/xgnid/graph_building.py`是项目自建实现，输出特征和连接与官方默认方案有实质差异，不应称为原作者原版。它由`build_innov1_graphs.py`和旧CPU pilot的prepare调用。
3. 当前R0和分组训练读取服务器已有图，不调用上述自建构图器重新生成。服务器回传确认Flow82/Packet1500、双向Link，与官方默认输出规格相符；不符合当前自建构图器1513维/单向Link的直接输出。但缺历史生成命令及逐行来源映射，不能反向证明旧图必然由现存某份源码生成。

## 对照来源和文件证据

原作者仓库：https://github.com/Yasir-ali-farrukh/GNN4ID 。固定提交：`551d1f10189e5259f8f34e32f918d557a2b3a64e`。本次使用raw.githubusercontent.com的该SHA链接重新下载5个文件；没有用会变化的main内容替代固定版。原始SHA256、归一化比较和notebook代码单元比较见`graph_generation_author_comparison_20261017.json`。

| 当前文件 | 对照结果 | 解释 |
|---|---|---|
| Utility/Functions.py | 换行统一后全文相同 | 包含NIDSDataset的CSV→图完整逻辑 |
| Utility/Feature_extractor_flow_packet_combined.py | 换行统一后全文相同 | NFStream提取，20包限额、idle_timeout120、报文元数据和payload |
| Utility/Additional_Features.py | 原始字节相同 | temporal特征模块，默认window_size350；不能证明历史生成时实际采用该值 |
| GNN4ID.ipynb | 原始字节相同，10个代码单元依序相同 | 官方流水线示例及NIDSDataset调用 |
| Data_preprocessing_CIC-IoT2023.ipynb | notebook全文不同，8个代码单元依序相同 | 差异为说明/单元排列等非代码内容；不是已确认采样算法被更改 |

Functions官方原始SHA256=`3cdc144ce6252d38ecfd751c6e1bd7b6a0616c8262d7c89a2b395677ae383b04`；本地原始SHA不同但换行统一后内容相同，不能因Git显示M或字节SHA不同就认定算法被修改。

固定仓库提交时间晚于论文发表，原作者具体实验运行与此提交的绑定仍缺证据。本次进一步查询文件提交历史遇到GitHub API限流，未取得可用历史链；不填造发表时版本。确认的是“原作者固定公开版相同”，不是“作者当年训练的所有输入、运行参数和私有改动完全恢复”。

## 官方默认构图与自建构图的差异

| 项目 | 官方Functions/NIDSDataset默认路径 | 自建graph_building.py |
|---|---|---|
| Packet节点 | include_packetflag=False、include_packetpayload=True；仅1500 payload字节特征 | 13个元信息+1500 payload，共1513维；13项含方向、3种大小、delta_time、8flags |
| Flow节点 | single_file=True时去Label及14个报文列，保留其余CSV列和原顺序，转float32；维度依赖输入表头 | 去相同列，但_safe_float遇非数值填0；本身不强制82维 |
| Contain连接 | Flow0→每包，再由ToUndirected产生reverse contain | 直接写Flow→Packet与reverse contain |
| Packet Link连接 | 初始i→i+1，随后ToUndirected产生双向Link | 只写i→i+1，未调用ToUndirected |
| Contain属性 | direction/ip_size/transport_size/payload_size；先dtype=int再float32 | 同四字段语义，按float转换；越界/非法数值可填0 |
| Link属性 | delta_time[1:]，先dtype=int再float32，原始形状[E] | 保留float，形状[E,1]；非整数输入处理与官方不同，不能只说形状适配 |
| payload异常处理 | bytes.fromhex，非法hex通常抛错 | 非法hex被转为全零向量，可能掩盖源数据问题 |
| 保存容器 | torch.save(data)，单个HeteroData | torch.save([graph])，单图list |
| 标签 | 官方single_file从Label取标量int64 | 自建取Label后生成长度1张量；当前loader可接受两者，但不是逐字原版 |
| max20限制 | 在上游NFStream My_Custom(limit=20)执行；图函数本身不重新截断 | 自建图函数也没有统一强制20包截断；创新入口另有筛包规则 |

官方include_packetflag=True时还可附加8flags，但这仍不等于自建13元信息方案。不能把自建1513维图简单截到1500后当官方payload图，因为元信息在payload之前，截断会保留元信息并丢掉部分payload。

## 实际调用链

- 官方：GNN4ID.ipynb调用`NIDSDataset(..., skip_processing=False, test=False, single_file=True)`；flags默认False/payload默认True。测试图生成需对应test CSV并设置test=True。历史实际参数仍须看生成命令。
- 自建创新：`build_innov1_packet_selection.py`产生筛包后的innov1 CSV，再由`build_innov1_graphs.py`调用`build_graphs_from_csv→build_graph_from_row`，输出processed_innov1_graphs下图文件；属于改变报文选择和构图输入的路径。
- 旧本地pilot：run_edge_chain.py的prepare也调用自建build_graph_from_row；不能拿它的图规格替代服务器旧图结论。
- 分组E0/E1/E2/E3：grouped_graphs.FrozenGraphs按samples清单加载原图并检查输入哈希；不调用graph_building。
- R0：r0_data先冻结已有data_N/data_test_N位置和哈希；R0Graphs只读加载、核验后clone，必要时将标量Link从[E]视为[E,1]供batch使用，不写回.pt，也不重新构图。

## 当前旧图来源能确认到哪里

已确认：服务器已有185557图通过R0清单检查，Flow82/Packet1500、contain/reverse4维、Link1维、1～20包及官方双向链；文件名分区及逐类数与表4一致。两份候选CSV哈希也已固定。

仍缺：当年具体生成入口/提交、参数、CSV→graph index的逐行映射、原始capture/flow身份、完整特征提取与采样日志。因此只可判定“现有输入格式与官方默认实现兼容”，不能凭格式一致认定作者原始样本或历史生成命令已恢复。当前R0复用旧图的事实不改变。

本次没有修正官方现存Benign MAC过滤未赋值问题，也没有修正自建1513维/单向Link等差异；审查不擅自改变正在运行实验的输入定义。下一步若补历史来源证据，应在服务器做只读来源取证或独立重建少量映射并对比指纹，不能覆盖现有图来“验证一致”。
