import pycurl
import StringIO
import json
import subprocess
import os
import pdb

def collect_info():
	user = 'dedey'
	password = 'Yoga2god?'
	cluster = 'gcr'
	vc = 'pnrsy'
	status = 'Passed'
	num_finished_jobs = '1000'

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


def copy_passed_logs_from_json(json_data=None):
	# Extract list of finished jobs
	fin_jobs_list = json_data['finishedJobs']

	# Copy jobs over to local disk 
	# if job name has '_ann' and has status 'Pass'
	for job in fin_jobs_list:
		job_name = job['name']
		job_status = job['status']
		ann_in_name = job_name.find('_ann')
		if ann_in_name > 0:
			if job_status == 'Pass':
				
				# The name of the scratch directory on Philly where logs are kept
				philly_scratch_dir = job['scratch']

				# The name of the folder on the local machine to put logs in
				local_dir = os.path.join('/home/dedey/DATADRIVE1/catch_all', job_name.split('.')[0])

				# Call the bash script with the arguments
				print 'Going to copy: ' + philly_scratch_dir + ' to ' + local_dir 
				output = subprocess.check_output(['./copy_logs_from_philly_template.sh', local_dir, philly_scratch_dir])
				print output


def main():
	json_data = collect_info()
	copy_passed_logs_from_json(json_data=json_data)




if __name__ == '__main__':
	main()