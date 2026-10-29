# 五项实验中断恢复：转到GPU0

用户提供日志显示E0-FT-s42第18轮触发早停，随后E0-FT-s43第7轮后进程消失。第7轮1477次累计更新，best=4、stale=3。日志没有退出堆栈，不能断定退出原因；torch-spline-conv禁用警告不等于已导致退出。

恢复入口只支持本次已定位的E0-FT中断，后续V1/V2必须尚未开始或已完整结束；若发现半途训练的向量门控，立即拒绝，不从零重跑。

原outputs/fair_vector_five_runs_v1保持只读。恢复先校验原实验协议、来源权重、已完成result/metrics/checkpoint/预测哈希和/proc中原队列是否仍活跃，复制已有任务到独立outputs/fair_vector_five_runs_recovery_v1。已完成任务直接跳过；当前E0任务从last.pt恢复。新目录已有日志/PID时拒绝重复提交。

服务器严格核对last/best/config及逐轮历史；重放已完成轮次的采样索引、逐轮比对顺序哈希和保存的采样器RNG，再载入模型、Adam动量/step、Python/CPU/CUDA全局RNG、累计更新、最佳指标、最佳轮次及stale计数。从saved_epoch+1继续，原30轮上限和patience12保持。若已经触发早停或达到30轮，只完成最佳权重验证及报告，不再追加更新。

检查点比日志最多领先一轮时，根据检查点与采样器重放补该行，未知train_loss置null并明确标注；不捏造训练损失。仅允许末尾破损JSON行，拒绝中间损坏、大于一轮的缺口或best/last不一致。原始文件在旧目录保留，新副本记录修复前哈希。

旧worker RNG没有保存。恢复采用独立DataLoader播种生成器，避免新worker消耗已恢复的模型RNG；图输入无随机变换，但不声称与不中断轨迹逐位一致。GPU0的型号和共享占用写入receipt，不停掉其他进程。共享情况下耗时不能作为独占速度结论。

恢复后继续E0-FT44、V1、V2，均按原协议新开未启动任务。最终调用原公平对照分析，保留V2−N0≥.10、V2−V1≥.05个百分点门槛，不改变类别损失人工审核要求，不自动追加训练或测试。

本机只进行标准库语法、历史计数/早停、日志缺口及任务规划检查；未加载真实服务器张量。实际张量、Adam和采样器校验通过后才写recovery_receipt.json并开始恢复训练。

服务器使用助手推送的固定提交号，通过git archive提取scripts/configs到/tmp，执行launch_resume_fair_vector.sh 0。新日志在outputs/fair_vector_five_runs_recovery_v1.log，最终状态、summary.json及report.md均在新恢复目录。旧入口和原始目录无需覆盖或删除。
