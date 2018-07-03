import subprocess

for i in range(3942, 3967):
    subprocess.call(['git', 'rm', 'cust_exps/run_exp_{}.sh'.format(i), '-f'])
