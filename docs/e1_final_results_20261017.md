# E1 分组无边属性实验最终回传（2026-10-17）

来源为用户服务器status.json及E1/metrics.json完整回传。根状态completed、exit_code=0；mode=full，最佳轮35、停止轮47，与patience12一致。新终端pgrep未返回train_grouped.py或run_r0_edge.py进程，这是回传时刻的查询结果，不保证GPU此后一直空闲。没有在本地运行模型或实验。

## 真实结果

| 指标 | 最佳验证（13446图） | 最终测试（13448图） |
|---|---:|---:|
| Accuracy | 96.7723% | 96.6761% |
| Macro-F1 | 94.9898% | 94.8672% |
| Weighted-F1 | 96.7260% | 96.6316% |

测试集各类F1：Benign95.8654%、WebBased79.3307%、Spoofing93.6033%、Recon95.0182%、Mirai100%、Dos99.7621%、DDos99.8743%、BruteForce95.4839%。WebBased的Precision85.3814%、Recall74.0809%，其中113/544被预测为Recon。仅据本次完整混淆矩阵作描述，不据测试反馈改冻结E1。

参数431704；initial_state_sha256=`906e125925a8a2cc53a526602db0f047f76b1e41b8087c893f5cf1af41f64a6b`；config_sha256=`86f68c51ec8c315befc94036c2424790fd5a1c04c5d0f648c41872a373b876b7`。运行墙钟15078.1516秒，约4小时11分18秒；单次耗时不代表稳定加速结论。尚未回传检查点/预测文件的独立完整性清单，不能把读取指标等同于所有原始产物逐文件验收完成。

与既有E0台账（Accuracy96.8248%、Macro-F1 95.2321%、Weighted-F1 96.8235%）比较，E0分别高约0.1487、0.3649、0.1919个百分点。E0数值依据先前台账，未在本轮重新取回E0完整metrics；这是seed42单次结果，不宣称统计显著或多种子稳定优势。E0/E1为分组受控实验；R0为另一套含现有源重复的官方示例协议重实现，不能跨协议直接计算模型提升。

## 下一步交付

R0数据清单与真实图服务器CPU两轮smoke已回传完成。GPU smoke仍未运行。新增`run_r0_cuda_smoke_server.py`仅作为服务器启动包装器：从CPU报告manifest ID匹配已有准备目录，读取真实code_root，核对worktree提交，检查残留E1/R0训练进程后启动**仅CUDA smoke**。不用旧终端变量，不重拼含字面量特殊字符的目录，不切换E1工作树，也不串联full。该新增包装器本地仅静态审查，运行验证留给服务器。

在服务器原仓库wx终端、确认cuda:0仍可供本任务使用后执行（没有Shell变量或命令替换）：

```bash
git fetch origin codex/r0-edge-preparation
git show origin/codex/r0-edge-preparation:scripts/run_r0_cuda_smoke_server.py > /tmp/xgnid_r0_cuda_smoke.py
python /tmp/xgnid_r0_cuda_smoke.py --repo .
```

会实时输出日志、实际清单路径、新输出目录和CUDA smoke PID。发现残留训练、CPU报告不完整、找不到唯一对应清单或worktree提交变化时拒绝启动；CUDA不可用或具体检查失败则保留失败日志，不回退CPU冒充通过。包装器的进程检查仅覆盖已知E1/R0入口，不能证明其他用户/其他入口完全没有占用GPU。完整smoke summary/status/exitcode回传后才判断是否放行30轮正式训练；当前仍未放行。
