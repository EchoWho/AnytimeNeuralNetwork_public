import pycurl
import StringIO
import json
import subprocess
import os
import pdb


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


def main():
    # From cam
    json_data = collect_info(cluster='eu1', status='Pass', num_finished_jobs=100)
    pdb.set_trace()

    
if __name__ == '__main__':
    main()
