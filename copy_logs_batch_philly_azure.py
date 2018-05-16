import pycurl
import StringIO
import json
import subprocess
import os
import errno
import pdb
import wget


def mkdir_p(path):
    try:
        os.makedirs(path)
    except OSError as exc:  # Python >2.5
        if exc.errno == errno.EEXIST and os.path.isdir(path):
            pass
        else:
            raise


def collect_info(cluster=None, status='Pass', num_finished_jobs='500'):
    user = 'dedey'
    password = 'Krishna2god?'
    vc = 'msrlabs'

    cmd = "https://philly/api/list?jobType=cust&clusterId={}&vcId={}&numFinishedJobs={}&userName={}&status={}".format(
        cluster, vc, num_finished_jobs, user, status)
    print cmd

    response = StringIO.StringIO()

    curl = pycurl.Curl()
    curl.setopt(pycurl.URL, cmd)
    curl.setopt(pycurl.SSL_VERIFYPEER, 0)

    curl.setopt(pycurl.HTTPAUTH, pycurl.HTTPAUTH_NTLM)
    curl.setopt(pycurl.USERPWD, "{}:{}".format(user, password))
    curl.setopt(pycurl.WRITEFUNCTION, response.write)

    curl.perform()
    curl.close()

    json_data = json.loads(response.getvalue())

    return json_data

def copy_passed_logs_from_json(json_data=None, local_dir=None):
    # Extract list of finished jobs
    # Finished jobs contain all jobs which are not running
    # including 'Failed', 'Killed' etc
    # But since we acquired only the status of jobs which have
    # 'Pass'ed we have only Passed jobs in the list.
    fin_jobs_list = json_data['finishedJobs']

    # Use wget to get logs back
    # Job logs and models are in 
    # folders which look like
    # https://storage.eu2.philly.selfhost.corp.microsoft.com/msrlabs/sys/jobs/myApplicationJobid/

    # Copy jobs over to local disk
    # if job name has '_ann'
    for job in fin_jobs_list:
        job_name = job['name']
        job_status = job['status']
        ann_in_name = job_name.find('_ann')
        if ann_in_name > 0:
            # The name of the scratch directory on Philly where logs are kept
            philly_scratch_dir = job['scratch']
            # Convert to https path
            # If you don't have the trailing '/' then it will try to get sibling directories as well
            https_link = 'https:' + philly_scratch_dir.replace('\\', '/') + '/'

            print '------------------------------------------'
            print 'Downloading ' + job_name.split('.')[0]
            print '------------------------------------------'

            # The name of the folder on the local machine to put logs in
            local_dir_this = os.path.join(local_dir, job_name.split('.')[0])

            if not os.path.exists(local_dir_this):
                os.makedirs(local_dir_this)

            cmd = 'wget -r --no-parent  -nH --cut-dirs 4 --directory-prefix ' + local_dir_this + '  -R "index.html*","model*" ' + https_link
            output = subprocess.call(cmd, shell=True)


def main():
    # From phillyOnAzure cluster
    json_data = collect_info(cluster='eu1', status='Pass', num_finished_jobs=100)
    local_dir = '/home/dedey/DATADRIVE1/ann_models_logs'
    copy_passed_logs_from_json(json_data=json_data, local_dir=local_dir)

    
if __name__ == '__main__':
    main()
