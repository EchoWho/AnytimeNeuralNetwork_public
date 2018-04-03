import subprocess

for i in range(3207, 3357):
    subprocess.call(['git', 'rm', 'run_exp_{}.sh'.format(i), '-f'])
