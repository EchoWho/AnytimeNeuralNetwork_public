# This script only runs with Python2.7

import pycurl
import StringIO
import json
import subprocess
import os
import pdb

def collect_info(cluster_name):
	user = 'dedey'
	password = 'Will2god?'
	cluster = cluster_name
	vc = 'msrlabs'
	status = 'all'
	num_finished_jobs = '2000'

	CMD="https://philly/api/list?jobType=cust&clusterId={}&vcId={}&numFinishedJobs={}&userName={}&status={}".format(cluster, vc, num_finished_jobs, user, status)

	response = StringIO.StringIO()

	curl = pycurl.Curl()
	curl.setopt(pycurl.URL, CMD)
	curl.setopt(pycurl.SSL_VERIFYPEER, 0)

	curl.setopt(pycurl.HTTPAUTH, pycurl.HTTPAUTH_NTLM)
	curl.setopt(pycurl.USERPWD, "{}:{}".format(user, password))
	curl.setopt(pycurl.WRITEFUNCTION, response.write)

	curl.perform()
	curl.close()

	json_data = json.loads(response.getvalue())

	return json_data


def print_minimal_info(json_data=None):

	# Running jobs
	print '------------------------------------------------'
	print 'Running jobs:'
	running_jobs_list = json_data['runningJobs']
	for job in running_jobs_list:
		job_name = job['name']
		job_status = job['status']
		job_progress = job['progress']
		job_model_dir = job['dir']
		job_log_dir = job['scratch']
		ann_in_name = job_name.find('ann')

		if ann_in_name > 0:
			print 'Name: ' + job_name + ' ' + 'Progress: ' + str(job_progress)
			print 'Model dir: ' + job_model_dir
			print 'Log dir: ' + job_log_dir 

	# Queued jobs
	print '------------------------------------------------'
	print 'Queued jobs:'
	queued_jobs_list = json_data['queuedJobs']
	for job in queued_jobs_list:
		job_name = job['name']
		job_status = job['status']
		job_progress = job['progress']
		job_model_dir = job['dir']
		job_log_dir = job['scratch']
		ann_in_name = job_name.find('ann')

		if ann_in_name > 0:
			print 'Name: ' + job_name + ' ' + 'Progress: ' + str(job_progress)
			print 'Model dir: ' + job_model_dir
			print 'Log dir: ' + job_log_dir 


	# Finished jobs
	print '------------------------------------------------'
	print 'Finished jobs:'
	finished_jobs_list = json_data['finishedJobs']
	for job in finished_jobs_list:
		job_name = job['name']
		job_status = job['status']
		job_progress = job['progress']
		job_model_dir = job['dir']
		job_log_dir = job['scratch']
		ann_in_name = job_name.find('ann')

		if ann_in_name > 0:
			print 'Name: ' + job_name + ' ' + 'Progress: ' + str(job_progress)
			print 'Status: ' + job_status
			print 'Model dir: ' + job_model_dir
			print 'Log dir: ' + job_log_dir 


	print '------------------------------------------------'

def main():
	json_data = collect_info('gcr')
	print_minimal_info(json_data=json_data)
	json_data = collect_info('rr1')
	print_minimal_info(json_data=json_data)



if __name__ == '__main__':
	main()