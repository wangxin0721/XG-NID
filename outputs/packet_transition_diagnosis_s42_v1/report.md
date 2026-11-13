# Seed42 冻结特征诊断与报文转移审计

原 C 已逐样本复现，验证 Macro-F1 为 97.553803%。每个线性头使用同一训练顺序训练 30 轮，共 6330 次更新，表中使用固定末轮。标准化统计来自训练集。

| 表示 | 维度 / 参数量 | 验证 Macro-F1 | 相对 C（百分点） | 修复 / 退步 |
|---|---:|---:|---:|---:|
| Flow | 64 / 520 | 90.012686% | -7.541117 | 94 / 679 |
| PacketMean | 64 / 520 | 91.043159% | -6.510645 | 68 / 763 |
| PacketQuery | 64 / 520 | 91.021296% | -6.532508 | 69 / 764 |
| FlowPacketMean | 128 / 1032 | 96.897398% | -0.656405 | 52 / 106 |
| D32Flow | 32 / 264 | 87.512982% | -10.040822 | 93 / 871 |
| D32Packet | 32 / 264 | 87.947115% | -9.606688 | 59 / 1049 |
| D32Fused | 32 / 264 | 96.815205% | -0.738598 | 38 / 89 |
| PacketMeanTransition | 80 / 648 | 91.982146% | -5.571657 | 59 / 681 |

## Recon → WebBased

| 表示 | C 原错分 | 原错分判对 | 新增该方向错分 | 探针该方向错分 |
|---|---:|---:|---:|---:|
| Flow | 43 | 31 | 24 | 34 |
| PacketMean | 43 | 27 | 21 | 34 |
| PacketQuery | 43 | 27 | 18 | 31 |
| FlowPacketMean | 43 | 17 | 11 | 37 |
| D32Flow | 43 | 35 | 21 | 26 |
| D32Packet | 43 | 20 | 48 | 69 |
| D32Fused | 43 | 13 | 11 | 41 |
| PacketMeanTransition | 43 | 17 | 22 | 46 |

## WebBased → Recon

| 表示 | C 原错分 | 原错分判对 | 新增该方向错分 | 探针该方向错分 |
|---|---:|---:|---:|---:|
| Flow | 15 | 2 | 123 | 133 |
| PacketMean | 15 | 4 | 111 | 120 |
| PacketQuery | 15 | 3 | 112 | 122 |
| FlowPacketMean | 15 | 3 | 42 | 53 |
| D32Flow | 15 | 1 | 131 | 141 |
| D32Packet | 15 | 5 | 137 | 143 |
| D32Fused | 15 | 3 | 36 | 48 |
| PacketMeanTransition | 15 | 4 | 70 | 79 |

## Benign → Spoofing

| 表示 | C 原错分 | 原错分判对 | 新增该方向错分 | 探针该方向错分 |
|---|---:|---:|---:|---:|
| Flow | 48 | 22 | 97 | 118 |
| PacketMean | 48 | 11 | 56 | 82 |
| PacketQuery | 48 | 13 | 57 | 80 |
| FlowPacketMean | 48 | 13 | 7 | 40 |
| D32Flow | 48 | 18 | 160 | 184 |
| D32Packet | 48 | 12 | 86 | 111 |
| D32Fused | 48 | 9 | 8 | 45 |
| PacketMeanTransition | 48 | 13 | 54 | 79 |

## Spoofing → Benign

| 表示 | C 原错分 | 原错分判对 | 新增该方向错分 | 探针该方向错分 |
|---|---:|---:|---:|---:|
| Flow | 46 | 12 | 225 | 253 |
| PacketMean | 46 | 12 | 66 | 99 |
| PacketQuery | 46 | 13 | 65 | 97 |
| FlowPacketMean | 46 | 5 | 21 | 61 |
| D32Flow | 46 | 20 | 256 | 278 |
| D32Packet | 46 | 8 | 102 | 135 |
| D32Fused | 46 | 3 | 10 | 53 |
| PacketMeanTransition | 46 | 10 | 62 | 95 |

## C、A、B37 共同的 Recon → WebBased 难例

| 表示 | 找回数 |
|---|---:|
| Flow | 27 |
| PacketMean | 23 |
| PacketQuery | 23 |
| FlowPacketMean | 13 |
| D32Flow | 31 |
| D32Packet | 19 |
| D32Fused | 9 |
| PacketMeanTransition | 15 |

## 有效相邻报文覆盖

| 范围 | 图数 | 无有效转移 | 至少两条有效转移 | 有效 / 预期转移 | 同向 / 换向 |
|---|---:|---:|---:|---:|---:|
| train | 107621 | 15191 | 66402 | 1063099 / 1063099 | 928258 / 134841 |
| val | 13446 | 1898 | 8337 | 134039 / 134039 | 116814 / 17225 |
| Recon -> WebBased | 43 | 0 | 32 | 295 / 295 | 154 / 141 |
| WebBased -> Recon | 15 | 2 | 8 | 77 / 77 | 40 / 37 |
| Benign -> Spoofing | 48 | 3 | 32 | 306 / 306 | 178 / 128 |
| Spoofing -> Benign | 46 | 3 | 25 | 240 / 240 | 153 / 87 |
| common_Recon -> WebBased_C_A_B37 | 38 | 0 | 27 | 257 / 257 | 132 / 125 |
| B37_repaired_Recon | 5 | 0 | 5 | 38 / 38 | 22 / 16 |

FlowPacketMean 找回 17 个原 Recon → WebBased，新增 11 个。D32Fused 找回 13 个，新增 11 个。
PacketMeanTransition 相对 PacketMean 的验证 Macro-F1 变化为 +0.938987 个百分点。原 Recon → WebBased 找回数从 27 到 17，新增该方向错分从 21 到 22。

有效边按时间顺序只计同流 i→i+1。NFStream 方向 0/1 合法，时间单位为毫秒，载荷为字节。0 毫秒和 0 字节保留。缺失或重复的 contain 映射、越界字段和非相邻边单列审计。归档图中没有用于区分零值填补与合法零值的掩码。

本次耗时 983.6 秒。八类指标和混淆矩阵见 `metrics.json`，分真实类别和难例的字段审计见 `field_audit.json`。
