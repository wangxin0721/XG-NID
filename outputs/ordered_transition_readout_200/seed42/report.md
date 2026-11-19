# 有序双尺度报文转移读出

Single 为原单尺度读出，Pointwise 为同参数量逐转移映射，Ordered 编码连续两次转移。

| 主干 seed | Single Macro-F1 | Pointwise Macro-F1 | Ordered Macro-F1 | Ordered−Single（百分点） | Ordered−Pointwise（百分点） |
| ---: | ---: | ---: | ---: | ---: | ---: |
| 42 | 97.825365% | 97.790083% | 97.845839% | +0.020474 | +0.055756 |

ordered-single 平均差为 +0.020474 个百分点。

ordered-pointwise 平均差为 +0.055756 个百分点。
