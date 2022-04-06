yolox: yolox 的主体

    yolox__fpn(darknet): 主干
    yolox_pafpn(darknet): 主干

    yolox_head: 侦测头 (★★★ 核心技术所在处)

losses:(被 yolo_head 依赖)
    IOUloss: 框的损失

network_blocks: 网络子模块, 同时被主干网络 darknet(yolo_fpn, yolo_pafpn) 和 yolo_head 依赖
    BaseConv: Conv2d + Batchnorm + 激活函数 的组合模块, 特性是 padding 自适应, 保证了 kernel_size 任意时 padding 不改变
    DWConv: 深度可分离卷积 + 卷积 == MobileNetV1 (操作像素的时候不操作通道, 操作通道的时候不操作像素, 速度快)
    Bottleneck: 瓶颈结构
    ResLayer: 残差层, 类似于对称瓶颈模型, 中间层固定压缩一半
    SPPBottleneck: # 中间采用空间金字塔, 金字塔 cat 后再约束到 output_channel
    CSPLayer: csp多瓶颈串联结构 和 1*1卷积 并行后 做 cat, 再用一个 conv3 约束到 output_channel
    Focus: # 聚焦到四个区域：top_left、top_right、bot_left、bot_right, 输入 (b,c,w,h) -> 输出 (b,4c,w/2,h/2)


