# 5090服务器：三路并行

下载根目录的 `SOH_5090_CS2_OXFORD_3LANES.zip`，上传AutoDL数据盘。
解压后进入 `SOH_5090_CS2_OXFORD`，使用已可用的GPU环境运行：

```bash
/root/miniconda3/bin/python start.py
```

不自动安装PyTorch或其他依赖；检查通过后后台启动三路：

| 日志 | 队列 | 训练次数 |
|---|---|---:|
| logs/cs2_other.log | CS2：LSTM、CNN-LSTM、MS-AgentNet、CNN-Transformer | 1664 |
| logs/oxford.log | Oxford：全部五个模型 | 2432 |
| logs/cs2_transformer.log | CS2：仅Transformer | 768 |

监督日志为 `logs/5090.log`；结果和断点为 `results/5090/<数据集>/task_XXXXX/`。
离开前确认三个日志都持续推进、有DONE记录，且磁盘/租用时间充足。

## 从此前两路5090包切换

先确认旧监督进程和全部子训练进程已经停止，不能直接叠加启动本包。
本次只改变任务分配，训练源码、输入、协议指纹和结果目录布局不变。
此前100/1000轮、种子1的同协议结果与断点可以保留续用，不得导入500轮旧协议记录。
异常终止可能留下锁，必须核实进程已结束再处理对应锁，不能盲目删除所有锁。

## 校验范围

三路互斥及总量校验、源码语法和ZIP完整性检查已通过。
模型代码沿用已通过704项模型检查的版本；本机空间不足，未完成100/1000轮端到端训练测试。
默认每50轮保存断点，磁盘剩余不足2GiB会停止。更多并行不保证一定更快。
训练轮次外的设置沿用原方案，包括CosineAnnealingLR(T_max=600)，不擅自改变调度器。
