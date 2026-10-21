# E3验证缓存分析回传与E3a推进（2026-10-21）

用户回传`xgnid.e3-gate-cache-analysis.v1`完整报告：completed、exit_code0、val13446，未加载模型、未推断测试。服务器源目录为`outputs/e3_gate_cache_analysis_b512_s42_v1`。真实回传及本地核对见[机器证据](e3_gate_cache_results_20261021.json)。

## 结果与证据界限

分析入口SHA与交付`28f14d2`一致；原诊断报告SHA与其原服务器JSON序列化字节相符，两个缓存SHA与已接收原诊断的记录一致。各类别样本数、成功/失败总数、E2/E3配对正确数与原验证矩阵一致，尺度/概率汇总有限且均值位于记录范围内。

本机仍未收到原两个压缩缓存及`validation_fusion_failures.csv`，因此没有在本地独立重跑逐行分类、重算缓存/CSV文件哈希或复算每个均值；接受的是匹配交付源码的服务器实际回传，并完成汇总一致性核对。没有新增模型实验或测试分数。

## WebBased：确认两类错误的准确数量

544个验证WebBased，动态融合正确463、错误81。错误分解：

| 类型 | 数量 | 占81个错误 |
|---|---:|---:|
| 偏向正确单路，融合仍错 | 49 | 60.49% |
| 偏向错误单路，融合也错 | 8 | 9.88% |
| 两路H(z)单独均错，融合仍错 | 24 | 29.63% |

没有等权平局错误或“两路单独均正确但融合错误”的回传类别。正确结果中有12个由“两路单独均错”恢复，不能把单路正确并集当作融合上界。

49个第一类错误中，**48个的正确分支是Packet，1个是Flow**。具体由原分组和本次逐行分类合并推导：Flow-only正确246例，原规则全部选Flow而融合错1例，故另48例属于Packet；这落在此前集合界限[42,56]中，现在不再只报告下界。

196个Packet-only正确的WebBased中：182个偏Packet，134个融合正确、48个融合错；另14个偏Flow，6个融合正确、8个融合错。总计140正确、56错误。可见分支偏好和最终分类之间存在明显区别，不能把56个都解释为门控选错。

## 尺度和概率给出的线索

以下为49个“偏正确路仍错”样本的**整组均值**，包含48个Packet-only及1个Flow-only样本，不冒充48个Packet子集的独立统计。

| 统计 | Flow | Packet |
|---|---:|---:|
| 最终权重 | 0.3803 | 0.6197 |
| z表示范数 | 5.9577 | 7.5262 |
| 加权z范数 | 2.4038 | 4.6167 |
| H(z)给WebBased的概率 | 0.0353 | 0.7669 |

融合给真类的平均概率为0.2396，单路Packet为0.7669；原始Flow门控均值0.3817、最终0.3803。该组Packet的表示及加权范数已经较大，当前不支持简单断言“Flow范数更大，把Packet压没”。较大权重/范数也不能保证共享非线性分类头给出正确类别。

这些是失败子集的描述性统计，不证明因果，也不证明将两个z归一化或加辅助监督必然改善。原始/最终权重均值接近不能排除个别样本受代理混合影响；需要时再用原CSV检查分布，不从均值推断每条都相同。

## E2/E3同样本配对结果

| 验证类别 | E2对、E3错 | E3对、E2错 | E3净正确变化 |
|---|---:|---:|---:|
| Benign | 26 | 30 | +4 |
| WebBased | 39 | 14 | −25 |
| Spoofing | 48 | 20 | −28 |
| Recon | 20 | 43 | +23 |
| Mirai | 0 | 0 | 0 |
| Dos | 2 | 0 | −2 |
| DDos | 0 | 1 | +1 |
| BruteForce | 0 | 2 | +2 |
| 合计 | 135 | 110 | −25 |

WebBased是重要瓶颈，Spoofing也不能忽略；E3在Recon等类有补偿性改善。上述是两个独立训练模型的样本配对，不能直接归因为单一门控变量。

全类别351个E3验证错误中：偏正确路仍错75、偏错误路仍错86、两路H(z)均错且融合错190。不能将WebBased的错误类型占比推广为所有类别的主因。

## 下一步：保持两个最小训练，不追加机制

阶段B及补充缓存分析已完成真实回传，进入[原计划](e3_gate_improvement_plan_20261021.md)的阶段C：先做E3a真实训练批探测，再独立提交训练/验证筛选。仍使用`fda02b1`交付的原工具/配置，保持与已通过诊断共用模块字节一致，不修改冻结代码、图、划分或已完成检查点。

E3a只移除最终代理混合；不加入归一化、类别权重、Focal、分支监督或额外校准。它仍保留门控输入中的熵/margin。推断干预略降不代表从头训练一定下降，但新训练是验证假设，不承诺提分。E3a完整筛选结束并核算预算后，才做E3a_equal_init（E3a-Z），不同时排队或复用probe。

### 第一段：E3a真实probe

以下复用已在服务器诊断时下载的`/tmp/xgnid_e3_gate_fda02b1`，显式共享物理GPU0，不停止其他任务。若服务器重启清空/tmp，先按原计划重新提取`fda02b1`四个文件，不用新版本覆盖旧绑定。

```bash
cd /home/nzxn/下载/WX/load/XG-NID
/home/nzxn/miniconda3/envs/wx/bin/python -B -u \
    /tmp/xgnid_e3_gate_fda02b1/run_e3_gate_ablation_server.py \
    --repo . --config /tmp/xgnid_e3_gate_fda02b1/e3_gate_study.json \
    --mode probe --variant E3a \
    --diagnostic-report outputs/e3_gate_diagnostic_b512_s42_v1/report.json \
    --physical-gpu 0 --allow-shared-gpu \
    --output outputs/e3_gate_probe_E3a_b512_s42_v1
```

probe只读取真实训练图，预热1次、实测3次优化更新，无完整训练轮次，不读取验证/测试图。应有completed/exit0、optimizer_update_verified/pure_gate_verified=true、test=null。目录已存在则查看其status/summary，不删除重跑，不将失败当通过。

### 第二段：probe真实成功后再独立启动筛选

```bash
cd /home/nzxn/下载/WX/load/XG-NID
if [ -e outputs/e3_gate_screen_E3a_b512_s42_v1 ] || [ -e outputs/e3_gate_screen_E3a_b512_s42_v1.log ]; then
    echo "已有E3a筛选产物，核查状态，不重复启动。"
else
    nohup /home/nzxn/miniconda3/envs/wx/bin/python -B -u \
        /tmp/xgnid_e3_gate_fda02b1/run_e3_gate_ablation_server.py \
        --repo . --config /tmp/xgnid_e3_gate_fda02b1/e3_gate_study.json \
        --mode screen --variant E3a \
        --diagnostic-report outputs/e3_gate_diagnostic_b512_s42_v1/report.json \
        --probe-report outputs/e3_gate_probe_E3a_b512_s42_v1/summary.json \
        --physical-gpu 0 --allow-shared-gpu \
        --output outputs/e3_gate_screen_E3a_b512_s42_v1 \
        > outputs/e3_gate_screen_E3a_b512_s42_v1.log 2>&1 < /dev/null &
    echo "$!" > outputs/e3_gate_screen_E3a_b512_s42_v1.pid
fi
tail -n 20 -F outputs/e3_gate_screen_E3a_b512_s42_v1.log
```

入口仍会自行核对实际成功probe的版本/诊断/初始化/GPU UUID/共享策略，不因后台提交提示就放行。保持batch512、lr0.001、seed42、评价batch16、最多100轮和patience12；屏蔽测试加载，最终metrics的test=null。

当前实际状态是“原检查点验证诊断完成、缓存分析完成、E3a probe及筛选尚未执行”。保留E2-B512为候选主对照；先得到两项最小训练的验证结果，再决定是否有必要进行一次尺度或分支监督配对研究及seed43/44确认。
