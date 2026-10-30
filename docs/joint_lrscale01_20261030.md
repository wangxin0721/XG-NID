# J0/J2下一轮：仅缩小联合阶段步长

用户提供的第一轮J2日志共12轮、每轮211次更新。预热V2为97.093667251942%；训练后最佳第3轮96.98297541433616%，低0.110691837606个百分点。第1轮96.97401376346687%，末轮96.97366991325892%。所有训练后候选未超过预热起点，最终第0轮入选。不能用J2最终高于J0的分数声称联合适应取得收益。

这组日志不能单独区分学习率、BatchNorm运行统计变化或其他优化因素。下一轮只将三个参数组学习率统一缩小为1/10：编码器1e-6、投影/H为1e-5、门控1e-5。仅一组固定配置，J0/J2各一次seed42，不扩展seed或LR网格。若仍无新增收益，再单独研究模式/高维融合，不在本轮混入其他因素。

新配置以joint_gate_recovery_v2_20261030.json为不可变基准，运行前逐字段核对：除学习率比例及来源记录外，所有字段相同。依然Adam重置、weight_decay=0、batch512、eval16、最多30轮、patience12、严格验证Macro-F1选模及第0轮候选；保持原train/eval与BN行为，不改dropout、损失或结构。

J0仍从原N0-s42出发，不从旧J0最佳权重继续。J2仍加载恢复队列V2第13轮最佳权重，固定SHA256=ebc151b5bd60f3d9fc6808d6f1b551ef5559da439ea109487ecded31929d06a7，不从旧J2的last.pt继续。只复用、不重跑预热。额外核对原联合结果的完成状态、N0身份、预热权重与基准配置哈希，确保新旧起点相同。

服务器执行前的真实图复现、两项梯度/实际更新探测、恢复干净起点、公共轮次数据顺序、最佳权重结构重建校验全部保留。本机无torch，仅完成静态/配置/统计测试，不声称真实张量探测已通过。

汇总补充selected_epoch_zero、完整last_validation和训练后最佳候选的轮次/分数/相对起点变化，以及末轮变化；逐轮日志新增train_loss、best_epoch与stale_epochs。不修改旧结果。summary还报告相同起点下的新旧J0/J2差值。旧联合入口也兼容这些附加报告字段，但已有输出不覆盖。

## 启动

助手推送后用固定提交号替换rev，默认GPU1并允许共享；正式训练只在服务器执行，不停止既有任务。

```bash
cd /home/nzxn/下载/WX/load/XG-NID
git fetch origin codex/r0-edge-preparation
rev=$(git rev-parse FETCH_HEAD)
bundle=$(mktemp -d /tmp/xgnid_joint_lr01.XXXXXX)
git -c core.autocrlf=false archive "$rev" scripts configs | tar -x -C "$bundle"
bash "$bundle/scripts/launch_joint_gate_lrscale01.sh" 1
```

输出outputs/joint_gate_lrscale01_s42_v1，拒绝覆盖已有输出/日志/PID。查看同名.log；完成后summary.json、comparison.json、report.md用于判断是否超过本轮各自起点、J2是否超过J0及旧联合结果。仍不读取测试集、不自动追加配置。
