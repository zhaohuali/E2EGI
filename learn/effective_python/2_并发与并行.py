# 并发与并行

# - 并发：计算机似乎是在同一时间做着很多不同的事
# - 并行：计算机确实在同一时间做着很多不同的事
# - 两者关键区别：并行提速了

'''用subprocess模块来管理子进程'''
import subprocess

# 父进程(python解释器)读取子进程的输出信息
'''
proc = subprocess.Popen(
    ['echo', 'Hello from the child!'],
    stdout=subprocess.PIPE)
out, err = proc.communicate()
print(out.decode('utf-8'))
'''

# 父进程与子进程独立，父进程一边读取子进程状态，一边处理其他事务
'''
proc = subprocess.Popen(['sleep', '0.3'])
while proc.poll() is None:
    print('working...')
print(proc.poll())
'''

# 把子进程从父进程中剥离decouple开（解耦），平行地运行多个子进程
'''import time
def run_sleep(period):
    proc = subprocess.Popen(['sleep', str(period)]) # 执行即运行
    return proc

start = time.time()
procs = []
for _ in range(2):
    proc = run_sleep(5)
    procs.append(proc)

time.sleep(10)

for proc in procs:
    proc.communicate()
end = time.time()
print(f'cost time: {end-start}')'''

