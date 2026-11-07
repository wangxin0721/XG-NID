# Flow查询与Packet第二层的联合优化：seed42四组200轮

本轮按最新安排优先筛选联合训练结构；seed43、44复现暂缓。
固定原Mean/Flow查询公式，不叠加跨层投影、残差、关系选择或乘法融合。

| 模型 | 查询来源 | 起点 | 新执行轮次 | 可训练参数 |
|---|---|---|---|---:|
| Frozen-Mean | Packet均值 | 原Mean-Query的150轮last.pt | 151～200 | 8168 |
| Frozen-Flow | Flow64 | 原Flow-Query的150轮last.pt | 151～200 | 8168 |
| Joint-Mean | Packet均值 | 原D32-Gate第62轮 | 1～200 | 25384 |
| Joint-Flow | Flow64 | 原D32-Gate第62轮 | 1～200 | 25384 |

四组共用D32成熟起点、seed42的查询/键初始权重和每轮样本顺序。
冻结组第151轮实际承接各自末轮权重，两者此前已学得不同参数。
查询与键均为无偏置64→8，新增1024参数；Q零初始化、K沿用原seed42初始化。
仍接原64→32投影、32维动态门控及八分类头。

## 训练范围与设置

联合组仅解冻Packet编码器第二层指向Packet的Flow→Packet和Packet→Packet两个GAT模块，
分别8704和8512参数，合计17216。Flow分支、Packet第一层、第二层Packet→Flow及全部BN参数/统计冻结。
编码器保持eval模式不影响解冻GAT参数的梯度；Packet第二层、BN(eval)、激活和查询池化在autograd内。

Packet第二层LR预先固定为1e-6，沿用此前局部解冻实验的量级；本轮不搜索学习率。
查询/键LR1e-4、原D32融合头LR1e-5，Adam、WD0、普通最终输出NLL，train512/eval16、workers2。
联合组新建Adam；冻结组完整继承原Adam。两组联合学习率和解冻范围一致。
联合组每批从原图重新计算报文表示，不能复用冻结节点缓存训练。

四组累计200轮，无早停；第0轮参与最终选模，同分保留最早轮次。
训练后最佳单独从1～200轮选择；分别报告0～150、0～200最佳、训练后最佳和末轮及其轮次。
每模型累计42200次更新（200×211）。本轮实际新增105500次更新（2×50×211＋2×200×211）。

## 冻结两组的完整续训

原150轮summary须与已归档JSON语义一致，四份检查点及预测文件核对原SHA256。
重算完整150轮账本的选模、矩阵、更新数和采样顺序，并核对采样器终止状态。
严格载入各自last.pt模型、Adam和所有参数的31650步计数，
恢复Python、NumPy、Torch CPU/CUDA RNG及采样器生成器。
Adam从CPU载入，moment按参数迁移，普通step计数保持CPU语义。

原账本按字节复制再追加151～200轮；继承第0轮、训练后最佳和原最佳，
不能从第151轮重置选模或Adam。正式更新前再次恢复checkpoint RNG。
旧最佳若仍胜出，保留全部训练状态，仅更换本轮元数据封套，并记录原配置和SHA256。
保存epoch150.pt与best150.pt，计算额外50轮分别带来的收益。
原`outputs/flow_query_pooling_s42_v1`只读，新结果写`outputs/flow_query_joint_packet2_s42_v1`。

## 训练前与完成后验收

检查来源、缓存、完整样本ID、标签和固定split/manifest；历史实现保持原字节。
冻结模型的原第0轮验证复现D32，联合模型再用全13446张验证原图检查初始预测与混淆矩阵。
四组原始D32＋查询初始state摘要相同；冻结续训两组承接各自的已学习状态。

预检每类取2张真实训练图，共16张。冻结组在副本上承接原Adam更新3步，
更新前后比较原图与缓存的分类log-probability、32维融合门控、全部Packet注意力，
atol/rtol均为1e-5，预测必须一致。联合组初始同样核对旧缓存，
更新后只用**当前已更新编码器重新提取**的临时特征快照核对读出路径；
该快照不用于正式训练。检查两类GAT、融合头和查询/键均有梯度及参数更新，
分类输出、门控和报文权重均变化，冻结参数与BN不变。
Q零初始化时第一步K梯度为零，后续Q/K均须更新。
探测仅改副本，finally恢复全部RNG。

本地CPU回归执行连续训练与保存恢复的下一批、下一次更新和后续全程比较，
检查模型、Adam、RNG、采样器、账本及继承选模完全一致；
另用真实GAT结构的小图验证仅允许的Packet第二层参数更新，BN/其他参数不变。
这些合成数据CPU检查不能替代服务器CUDA与真实图验收。

训练结束严格重载每个选中检查点，回放全部13446张验证原图，
预测与选模时一致，混淆矩阵/指标一致，冻结部分摘要不变。
导出检查点/账本/预测哈希、八类精确率/召回率/F1、每样本门控与Packet权重、四对修复/退步。

## 主要比较与结论边界

1. Joint-Flow−Joint-Mean：联合训练时Flow查询的独立贡献。
2. Joint-Flow−Frozen-Flow、Joint-Mean−Frozen-Mean：两种查询各自的解冻收益。
3. (Joint-Flow−Joint-Mean)−(Frozen-Flow−Frozen-Mean)：查询优势是否随解冻扩大。

原seed42的150轮+0.047929个百分点独立保留，不覆盖为200轮结果。
继续关注WebBased、Spoofing、Recon修复/退步，尤其Spoofing召回。
JCL-Multi的97.524609%仅作历史成绩参考。当前只筛选这一轮四组，
不评估测试图、不声明统计显著、不自动启动seed43/44或后续消融。

## 服务器启动

先上传本次交付的离线tar.gz到项目根目录，核对交付SHA256，解压到独立临时目录。
在项目根目录执行`bash "$bundle/scripts/launch_flow_query_joint_packet2.sh" 1`，
启动器先在CPU运行8项回归，再后台启动GPU1的四组任务，并打印独立日志与PID。
也可手动执行`python -B scripts/test_flow_query_joint_packet2.py`，再执行
`python -B -u scripts/run_flow_query_joint_packet2.py --repo "$PWD" --train`。
runner完成全部四组预检后才开始正式训练；不加`--train`仅预检，写独立preflight目录。
原150轮输出、D32检查点、Packet节点缓存和原图必须保留。输出目录存在时拒绝覆盖。
