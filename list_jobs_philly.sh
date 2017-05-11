#!/bin/bash

USERNAME="dedey"
PASSWORD="Yoga2god?"
CLUSTER="gcr"
VC="msrlabs"

CMD="https://philly/api/list?" 
CMD+="jobType=cust&"
CMD+="clusterId=$CLUSTER&" 
CMD+="vcId=$VC&" 
CMD+="numFinished=5"         
 
curl -k --ntlm --user "$USERNAME:$PASSWORD" "$CMD"