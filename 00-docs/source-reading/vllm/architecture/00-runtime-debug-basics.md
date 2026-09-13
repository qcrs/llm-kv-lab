# Linux 进程观测基础：为 vLLM 源码调试服务

> 目标：不是系统学习 Linux，而是掌握后续阅读 vLLM 多进程架构时真正需要的进程观测能力。
> 当前实验：`04-experiments/vllm-bridge/scripts/smoke_06b.py`，vLLM `0.26.0`，TP=1。

---

## 1. 为什么需要先理解 Linux 进程

nano-vLLM 中，大部分对象都可以直接理解为：

```text
一个 Python 程序
├── Scheduler
├── ModelRunner
└── KV Cache
```

但 vLLM V1 已经出现明显的**进程边界**。

本次实际实验观察到：

```text
bash commands.sh
└── python smoke_06b.py
    └── VLLM::EngineCore
```

实际 PID：

```text
python smoke_06b.py
PID = 3907513

VLLM::EngineCore
PID  = 3908079
PPID = 3907513
```

同时 `nvidia-smi` 显示：

```text
PID       Process             GPU Memory
3908079   VLLM::EngineCore    75170 MiB
```

因此目前可以确认：

```text
Frontend Python Process
PID 3907513
        │
        │ 创建
        ▼
EngineCore Process
PID 3908079
PPID 3907513
        │
        └── 持有主要 GPU 显存 / CUDA 资源
```

这是后续理解：

```text
EngineCore
Scheduler
KVCacheManager
Executor
GPUWorker
ModelRunner
```

对象所有权的基础。

---

# 2. Process、PID 和 PPID

## 2.1 Process

Process，即**进程**。

例如执行：

```bash
python 04-experiments/vllm-bridge/scripts/smoke_06b.py
```

Linux 会创建一个 Python 进程。

每个进程拥有自己的运行状态和资源，例如：

```text
虚拟地址空间
内存
文件描述符
线程
进程状态
```

---

## 2.2 PID

PID：

```text
Process ID
```

即当前进程自己的编号。

例如：

```text
PID = 3908079
```

代表：

```text
当前这次运行中的 EngineCore 进程
```

注意：

> PID 不是 vLLM 固定编号。

每次重新运行程序，Linux 都可能分配新的 PID。

例如：

```text
第一次：
EngineCore PID = 3908079

下一次：
EngineCore PID = 3912008
```

因此调试过程中不能把 PID 写死。

---

## 2.3 PPID

PPID：

```text
Parent Process ID
```

表示：

> 当前进程是谁创建的。

例如：

```text
EngineCore

PID  = 3908079
PPID = 3907513
```

说明：

```text
PID 3907513
    │
    └── 创建了
        PID 3908079
```

本次实验对应：

```text
python smoke_06b.py
PID 3907513
    │
    └── VLLM::EngineCore
        PID 3908079
```

---

# 3. Process 和 Object 不能混淆

这是学习 vLLM 时非常重要的一点。

例如源码中：

```python
scheduler = Scheduler(...)
```

这里只是创建一个：

```text
Scheduler Python Object
```

并不代表 Linux 中创建了：

```text
Scheduler Process
```

同理：

```python
worker = GPUWorker(...)
```

也可能只是：

```text
EngineCore Process
    └── GPUWorker Object
```

而不是：

```text
EngineCore Process
    └── GPUWorker Process
```

因此以后读源码必须分别问两个问题：

```text
问题 1：
这个对象是谁创建、谁持有？

问题 2：
这个对象运行在哪个 Linux process 中？
```

不能把：

```text
class / object
```

和：

```text
process
```

混为一谈。

---

# 4. Process 和 Thread

一个 Process 内部还可以有很多 Thread。

例如：

```bash
cat /proc/3908079/status
```

实际得到：

```text
Name:    VLLM::EngineCor
Pid:     3908079
PPid:    3907513
Threads: 45
```

说明：

```text
EngineCore
```

是 **1 个 Process**，但是里面有：

```text
45 个 Thread
```

---

## 4.1 pstree 中如何区分

本次实际：

```text
VLLM::EngineCor,3908079
├─{VLLM::EngineCor},3908294
├─{VLLM::EngineCor},3908362
├─...
```

其中：

```text
VLLM::EngineCor,3908079
```

是 Process。

而：

```text
{VLLM::EngineCor},3908294
```

带 `{}` 的通常表示 Thread。

因此不能理解成：

```text
启动了几十个 EngineCore
```

正确理解是：

```text
1 个 EngineCore Process
└── 大约 45 个线程
```

这与 `/proc/.../status` 中的：

```text
Threads: 45
```

互相验证。

---

# 5. ps：查看 Linux 进程

`ps` 可以理解为：

```text
Process Status
```

用于查看当前进程。

核心模型：

```text
ps

我要看谁？
├── -p PID
│   └── 只看指定进程
│
└── -e
    └── 看全部进程

我要显示什么？
└── -o
    └── 指定输出字段
```

---

# 6. `ps -p`：只查看指定 PID

例如：

```bash
ps -p 3908079
```

表示：

> 只查看 PID=3908079 的进程。

由于 PID 经常变化，我们通常保存到变量：

```bash
ENGINE_PID=3908079
```

然后：

```bash
ps -p $ENGINE_PID
```

Shell 会将：

```text
$ENGINE_PID
```

替换为：

```text
3908079
```

所以实际上执行的是：

```bash
ps -p 3908079
```

---

# 7. `ps -e`：查看全部 Process

例如：

```bash
ps -e
```

表示：

```text
-e
=
查看系统当前所有进程
```

所以：

```bash
ps -eo pid,ppid,cmd
```

等价于：

```bash
ps -e -o pid,ppid,cmd
```

含义：

```text
查看所有进程
+
只输出 PID / PPID / CMD
```

---

# 8. `ps -o`：控制输出字段

例如：

```bash
ps -p $ENGINE_PID \
  -o pid,ppid,stat,%cpu,%mem,cmd
```

其中：

```text
-o
=
output format
=
指定我想看到哪些字段
```

输出类似：

```text
PID      PPID      STAT   %CPU   %MEM   CMD
3912008  3911521   Sl+    24.5   0.1    VLLM::EngineCore
```

---

# 9. 常用 ps 字段

## PID

```text
当前 Process ID
```

---

## PPID

```text
父 Process ID
```

用于建立父子关系。

---

## CMD

运行的程序 / 命令。

例如：

```text
python 04-experiments/vllm-bridge/scripts/smoke_06b.py
```

或者：

```text
VLLM::EngineCore
```

---

## %CPU

```text
Process 的 CPU 使用率
```

例如：

```text
%CPU = 24.5
```

这里 `%` 是字段名称的一部分。

所以 `ps` 中字段名就是：

```text
%cpu
```

并不是 Shell 特殊语法。

---

## %MEM

表示：

```text
Process 使用的 Host RAM
/
系统总 RAM
```

注意：

```text
%MEM
≠
GPU 显存占用
```

区别：

```text
ps %mem
→ CPU / Host RAM

nvidia-smi
→ GPU VRAM
```

---

# 10. STAT：Process 当前状态

例如：

```text
STAT = Sl+
```

当前阶段重点记：

```text
R
Running
正在运行或等待 CPU 调度

S
Sleeping
正在等待某个事件

D
Uninterruptible Sleep
通常在等待内核 I/O

T
Stopped
进程暂停

Z
Zombie
进程已经结束，但是父进程还未完成回收
```

附加字符：

```text
l
multi-threaded

+
位于前台 process group
```

因此：

```text
Sl+
```

大致可以理解：

```text
当前处于等待状态
+
这是一个多线程进程
+
属于当前前台 process group
```

需要特别注意：

> `S = Sleeping` 不意味着 GPU 没有工作。

例如 GPU 程序可能：

```text
CPU 发起 CUDA Kernel
        ↓
GPU 执行
        ↓
CPU 等待
        ↓
CPU Process 显示 S
```

因此判断 GPU 状态不能只看 `ps STAT`。

---

# 11. `--forest`：查看 Process 父子树

命令：

```bash
ps -eo pid,ppid,stat,cmd --forest
```

其中：

```text
--forest
```

表示：

> 根据父子关系用树形结构展示。

普通输出：

```text
3907511 bash
3907513 python
3908079 EngineCore
3907514 tee
```

不容易直接理解关系。

加 `--forest` 后：

```text
bash commands.sh
├── python smoke_06b.py
│   └── VLLM::EngineCore
└── tee startup.log
```

对理解 vLLM 多进程非常有用。

---

# 12. Pipe：`|`

Linux 中：

```text
|
```

叫 Pipe（管道）。

例如：

```bash
ps -eo pid,ppid,cmd \
| grep EngineCore
```

含义：

```text
ps 输出
    ↓
交给 grep
    ↓
grep 过滤
```

即：

> 先列出 Process，再只留下包含 `EngineCore` 的行。

---

# 13. grep：过滤文本

例如：

```bash
ps -eo pid,ppid,cmd \
| grep EngineCore
```

`grep` 会保留包含：

```text
EngineCore
```

的行。

---

## 多条件匹配

```bash
grep -E "smoke_06b|EngineCore|vllm"
```

其中：

```text
-E
```

启用扩展正则表达式。

里面的：

```text
|
```

表示：

```text
OR
```

所以：

```text
smoke_06b
|
EngineCore
|
vllm
```

就是：

> 只要一行出现其中任意一个，就留下。

---

# 14. 为什么要 `grep -v grep`

执行：

```bash
ps ... | grep EngineCore
```

时，`grep EngineCore` 本身也是一个 Process。

它自己的 command line 中也包含：

```text
EngineCore
```

所以可能把自己显示出来。

因此：

```bash
grep -v grep
```

表示：

```text
-v
=
反向过滤
```

删除包含：

```text
grep
```

的行。

完整：

```bash
ps ...
| grep -E "smoke_06b|EngineCore|vllm"
| grep -v grep
```

可以理解为：

```text
查看 Process
→ 找 vLLM 相关
→ 删除 grep 自己
```

---

# 15. pgrep：更简单地找 PID

很多时候不需要：

```bash
ps | grep
```

可以直接：

```bash
pgrep -af smoke_06b.py
```

例如：

```text
3907513 python 04-experiments/vllm-bridge/scripts/smoke_06b.py
```

其中：

```text
pgrep
=
process grep
```

`-f`：

```text
匹配完整 command line
```

`-a`：

```text
同时显示完整命令
```

因此后续常用：

```bash
pgrep -af EngineCore
```

或者：

```bash
pgrep -af smoke_06b.py
```

---

# 16. pstree：查看父子 Process

例如：

```bash
pstree -ap 3907513
```

含义：

> 从 PID=3907513 开始，显示其全部子进程和线程关系。

本次实验：

```text
python,3907513
  └─VLLM::EngineCor,3908079
      ├─{VLLM::EngineCor}
      ├─{VLLM::EngineCor}
      └─...
```

非常直观地证明：

```text
smoke_06b.py
        ↓
创建
        ↓
EngineCore
```

---

# 17. `/proc/<PID>`：从 Kernel 查询 Process

Linux 有：

```text
/proc
```

这是一个虚拟文件系统。

可以把：

```text
/proc/3908079
```

理解成：

> Linux kernel 暴露的 PID=3908079 的运行信息。

例如：

```bash
cat /proc/3908079/status
```

查看详细 Process 状态。

我们只关注：

```bash
cat /proc/$ENGINE_PID/status \
| grep -E "Name|Pid|PPid|Threads"
```

实际：

```text
Name:       VLLM::EngineCor
Pid:        3908079
PPid:       3907513
Threads:    45
```

这直接验证：

```text
EngineCore
PID = 3908079
父进程 = smoke Python
线程数 = 45
```

---

# 18. Shell 变量：`$ENGINE_PID`

例如：

```bash
ENGINE_PID=3908079
```

创建一个 Shell 变量。

查看：

```bash
echo $ENGINE_PID
```

输出：

```text
3908079
```

因此：

```bash
cat /proc/$ENGINE_PID/status
```

等价于：

```bash
cat /proc/3908079/status
```

Shell 中：

```text
$变量名
```

表示：

> 取这个变量当前保存的值。

---

# 19. `$(...)`：获取命令的输出

例如：

```bash
ENGINE_PID=$(pgrep -f "VLLM::EngineCore" | head -n 1)
```

可以分成：

```text
第一步：

pgrep ...
↓
3908079

第二步：

$(...)
↓
把 3908079 取回来

第三步：

ENGINE_PID=3908079
```

所以：

```text
$(command)
```

可以理解为：

> 执行 command，把 command 的输出作为一个值使用。

---

# 20. nvidia-smi：谁真正占用了 GPU

命令：

```bash
nvidia-smi \
  --query-compute-apps=pid,process_name,used_memory \
  --format=csv
```

只查询 GPU compute process，并输出：

```text
PID
Process Name
GPU Memory
```

本次实际结果：

```text
3908079
VLLM::EngineCore
75170 MiB
```

因此可以确定：

```text
EngineCore PID 3908079
        ↓
持有 CUDA/GPU 资源
        ↓
约占 75 GB GPU memory
```

这里约 75GB 并不是 Qwen3-0.6B 模型权重。

之前启动日志显示：

```text
模型权重 ≈ 1.12 GiB
KV Cache ≈ 71.67 GiB
```

因此 GPU 显存大头实际上是：

```text
预分配 KV Cache
```

而不是模型。

---

# 21. 当前 vLLM 进程结构：实验结论

截至 VB-00，我们已经有足够证据确认：

```text
┌─────────────────────────────────┐
│ Frontend Python Process         │
│ smoke_06b.py                    │
│ PID 3907513                     │
│                                 │
│ LLM(...)                        │
│ generate(...)                   │
└───────────────┬─────────────────┘
                │
                │ 创建子进程
                ▼
┌─────────────────────────────────┐
│ EngineCore Process              │
│ PID 3908079                     │
│ PPID 3907513                    │
│ Threads ≈ 45                    │
│                                 │
│ GPU Memory ≈ 75 GB              │
└─────────────────────────────────┘
```

同时，没有观察到独立：

```text
GPUWorker Process
```

因此当前假设是：

```text
EngineCore Process
├── Scheduler Object
├── KVCacheManager Object
├── Executor Object
├── GPUWorker Object
└── ModelRunner Object
```

其中具体对象关系需要下一阶段通过 vLLM 源码确认。

所以现在应明确区分：

```text
Frontend
EngineCore
→ Process 层次

Scheduler
KVCacheManager
GPUWorker
ModelRunner
→ 目前需要进一步确认其 Object 所有权和进程位置
```

---

# 22. 当前真正需要记住的 5 个命令

不需要记大量 Linux 命令。

后续调试 vLLM，当前只要求熟悉下面五个。

## 查看进程树

```bash
ps -eo pid,ppid,stat,cmd --forest
```

---

## 查一个 Process

```bash
ps -p $ENGINE_PID \
  -o pid,ppid,stat,%cpu,%mem,cmd
```

---

## 按名字寻找 Process

```bash
pgrep -af EngineCore
```

---

## 查看某个 Process 的子进程和线程

```bash
pstree -ap $ENGINE_PID
```

---

## 查看 GPU Process

```bash
nvidia-smi \
  --query-compute-apps=pid,process_name,used_memory \
  --format=csv
```

---

# 23. 当前掌握程度

完成这部分后，应能够独立解释：

```text
PID 和 PPID 是什么？

Process 和 Thread 有什么区别？

Process 和 Python Object 有什么区别？

ps -p 是什么？

ps -e 是什么？

ps -o 是什么？

STAT 是什么？

%MEM 为什么不是 GPU Memory？

如何判断 EngineCore 的父进程？

如何判断哪个进程真正持有 GPU？
```

如果这些能够解释，就不需要继续学习 Linux 进程理论。

下一阶段直接回到 vLLM：

```text
Frontend Python
        ↓
在哪里创建了
        ↓
EngineCore Process？
```

然后从 Linux 运行结果反向进入：

```text
EngineCoreProc
EngineCore
Executor
GPUWorker
```

源码。
