# 残差入口的归档依赖加载修正

服务器运行d817569时，CPU检查和旧CL折叠核验通过，但正式进程在load_classes()中加载
residual_readout.py时出现 `ModuleNotFoundError: No module named 'xgnid.relation_selection'`。
该调用发生在模型构建、缓存提取、梯度探测和训练之前，因而此次失败没有正式更新。

根因是train_grouped先从服务器检出目录导入了xgnid。临时归档中有新的relation_selection.py，
但已导入包的搜索路径没有包含它。原入口按路径加载了最外层residual_readout，却依赖包搜索路径寻找其新依赖。

修正为先按本次归档的绝对路径加载xgnid.relation_selection，再加载residual_readout。
没有修改模型数学表达式、参数、初始化、学习率、训练预算和折叠检查器。
新增独立子进程测试先导入xgnid，再将包搜索路径指向没有新模块的目录，清理新模块缓存，
随后调用正式load_classes并以真实GAT图验证输出。此测试在旧实现复现同一错误，在修正后通过。
当前6项标准检查及7项真实张量检查均通过；CPU环境为独立PyTorch2.3.1/PyG2.5.3。

## 服务器重跑

保留失败的outputs/residual_pair_s42_v1及其日志。启动器新增XGNID_RESIDUAL_OUTPUT_NAME变量，
默认输出路径仍为原v1；本次重跑显式选择新的residual_pair_s42_retry_v1。
不恢复失败进程内存或优化器，两组均重新从原D32开始，固定150轮。
已通过的outputs/cross_layer_fold_check_v1.json会按原规则重新校验并复用，无需重复或删除。

取最新修正提交，归档scripts/configs/src/xgnid及docs/packet_pool_results_20261102.json后运行：

```bash
XGNID_RESIDUAL_OUTPUT_NAME=residual_pair_s42_retry_v1 \
  bash "$bundle/scripts/launch_residual_pair.sh" 1
tail -n 50 -F outputs/residual_pair_s42_retry_v1.log
```

GPU预检、原图探测与正式150轮训练仍需服务器实际执行。torch-spline-conv的可选导入警告与本次模块路径错误分开记录。
