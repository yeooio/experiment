# H100单卡四数据集并行包

一张H100同时运行四个独立GPU进程：CS2、CX2、MIT、Oxford。每个数据集完整运行五种模型和其全部参数组合，共12,160次训练；每次500轮，种子固定1–10。四个数据集合计48,640次训练。

## 运行前

- 本包必须单独运行，不与本机或其他设备上相同数据集的任务合并。启动H100全量搜索后，应停止其他设备上的CS2、CX2、MIT和Oxford搜索，避免重复。
- 至少预留约8GB用于CUDA环境，另为结果、日志和中断检查点预留空间。完成任务自动删除对应检查点，只保留result.json；运行中的任务仍有一个断点。
- 四进程共享一张H100。每进程一个CPU线程。若服务器为共享队列，需先申请整张H100或遵守调度系统分配的CUDA_VISIBLE_DEVICES。
- 每个数据集有独立目录和独占锁，防止重复启动。

## Windows

准备Python 3.13及NVIDIA驱动，解压到本地可写目录，双击`START_H100.cmd`。首次会创建`.venv`并安装CUDA版PyTorch和依赖，然后完成704项结构检查，再启动四个进程。

## Linux

先在已分配H100的Python 3.13环境中安装与本包一致的依赖。验证`python3 -c "import torch; print(torch.__version__, torch.cuda.is_available(), torch.cuda.get_device_name(0))"`。然后执行：

```bash
chmod +x start_h100.sh
PYTHON_BIN=python3 GPU_INDEX=0 ./start_h100.sh
```

Linux脚本不会擅自重装集群CUDA/PyTorch环境。必须使用PyTorch 2.11.0 CUDA构建及requirements.txt中的版本；如果集群不提供该组合，先创建独立环境。

## 监控与恢复

- 标准输出和错误分别写入`logs/<dataset>.stdout.log`和`logs/<dataset>.stderr.log`。
- 结果位于`dataset_results/<dataset>/task_*/result.json`。
- 每50轮写一次断点。正常完成后删除断点，节省磁盘。
- 任一进程失败，启动器会停止另外三个，保留断点。查明错误后，确认没有遗留Python进程，再删除四个`DATASET_RUNNING.lock`，重新启动会跳过已完成任务并恢复断点。
- 不要编辑runner.py、source或inputs；协议指纹变化后已有结果会被拒绝。

注意：四进程已通过本地RTX4050的同代码路径验证，但H100机器仍必须执行自己的704项预检。是否能在今晚完成取决于H100上的实测吞吐，不能仅由显存容量保证。
