# 有序双尺度报文转移读出

Single 为原单尺度读出，Pointwise 为同参数量逐转移映射，Ordered 编码连续两次转移。

| 主干 seed | Single Macro-F1 | Pointwise Macro-F1 | Ordered Macro-F1 | Ordered−Single（百分点） | Ordered−Pointwise（百分点） |
| ---: | ---: | ---: | ---: | ---: | ---: |
| 42 | 97.664335% | 97.678574% | 97.704448% | +0.040114 | +0.025874 |

ordered-single 平均差为 +0.040114 个百分点。

ordered-pointwise 平均差为 +0.025874 个百分点。
