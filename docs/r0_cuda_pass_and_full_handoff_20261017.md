# R0 CUDA冒烟真实通过与正式训练交付

依据用户真实服务器回传；本轮本地只编写和静态审查启动/状态脚本，没有运行模型或实验。CUDA冒烟启动器返回0，R0 summary为completed/exit_code0/mode smoke/epochs_completed2，test=null。

## 通过证据

- GPU Tesla T4，Python3.9.23、torch2.3.1+cu118、PyG2.5.3、CUDA11.8，device_type=cuda。
- manifest ID=`22d027fe3d0b0ad9079da9a13614597b0e40bb883a34b2d4d825452d6129d383`；config ID=`7463253af0c7d165dc7c3b681895e6d918e8c83f80994ac896ef5045f84e414b`。
- 两轮均32训练样本，lr0.01，loss有限；优化一步验证true，参数431704。
- 六个GAT属性输入均观察到：Contain/Reverse为125×4，Link为218×1，两层覆盖。
- Link两层边投影非零梯度约1.2396e-5/4.0542e-6，Reverse约0.0022825/9.2474e-6；Contain为0。
- 隔离扰动差Link0.03276539、Reverse0.90169907，超过容差1.90735e-5；Contain差1.90735e-6与重复前向抖动相当，符合单入边机制，不宣称Contain存在可辨识属性增益。
- last checkpoint SHA256=`1e1e54111edb108c9877faf1ac1d933e21fe16a1dac5e4c8ad469e5094db1f15`。
- torch-spline-conv禁用和NVML警告仍出现，但没有阻止这次实际CUDA图计算与优化验证；不据此重装驱动或依赖。

用户粘贴文本中的实际目录字符串被终端/富文本转义混入换行；后续从服务器JSON及Path实际读取，不根据粘贴路径手工重建。原始已通过的worktree/model/data源码不改变，full入口再次校验GPU smoke的代码/配置/清单/runtime和checkpoint SHA。

## 正式训练启动定义

独立R0，claim固定protocol_reimplementation；160000训练、25557测试；原始Edge GAT、Adam lr0.01、batch64/test1、seed42、30epoch、weight_decay0、无调度器/验证早停，选最终轮权重，仅结束后测试一次。保留现有源重叠，明确13.09%测试Edge输入与训练共享，不修改测试成员来贴分数。作者原始身份与CSV→图逐行生成证据仍缺失，不能称严格复现已成功。

新增`run_r0_full_server.py`：

- 自动匹配已完成CPU/CUDA smoke与源准备目录，要求唯一一致记录，检查配置、代码、worktree提交和CUDA checkpoint哈希；不按分数挑报告。
- 核验E1完成、检查已知训练进程后，仅启动一个后台监督进程。内层正式入口再核验运行环境与原GPU smoke一致；不回退CPU。
- 原worktree不切换或更新。启动记录`outputs/r0_full_launch.json`以exclusive模式创建，存在即拒绝第二次运行，防止重复粘贴和自动重跑。
- 监督器独立于当前终端，记录训练PID、监督PID、log及真实退出码；实际训练入口写status/config/epochs/last/metrics/predictions/summary。只有监督器提交不等于已开始或完成训练，需要状态回传。
- 不支持自动resume，不删除旧目录或失败记录，不替换冒烟checkpoint，不串联E2/E3。

在服务器原仓库wx终端，确认cuda:0仍由本任务使用后执行。无需旧Shell变量，不用用户复制含特殊字符的旧目录：

```bash
git fetch origin codex/r0-edge-preparation
git show origin/codex/r0-edge-preparation:scripts/run_r0_full_server.py > /tmp/xgnid_r0_full.py
python /tmp/xgnid_r0_full.py --repo .
```

启动后查看实际PID、status、每100batch日志及最近epoch；此命令只读，不启动第二次训练：

```bash
python /tmp/xgnid_r0_full.py --repo . --status
```

回传第一次状态确认启动，再在完成2～3轮后回传同一命令输出，用实际每轮训练加训练集评价耗时估计剩余时间，不由4秒冒烟或另一协议E0线性外推。full没有early stopping，须完整30轮且最终测试/文件保存成功。若失败，保留log/exitcode/weights并审查原因，不自动重跑。

最终验收：status completed、监督退出码0、epochs_completed30、selection last_epoch、test.samples25557、逐类support/预测行数正确、清单和代码/runtime绑定一致、last checkpoint与完整结果日志齐全。启动脚本当前仅已交付，正式训练尚未在本次本地会话执行；以用户接下来服务器回传为准。
