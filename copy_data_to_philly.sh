
export PHILLY_VC=msrlabs

local_folder=/home/dedey/DATADRIVE1/ann_data_dir
remote_folder_gcr=hdfs://gcr/msrlabs/dedey/ann_data_dir/
remote_folder_cam=hdfs://cam/msrlabs/dedey/ann_data_dir/
remote_folder_rr1=hdfs://rr1/msrlabs/dedey/
#remote_folder_eu1=gfs://eu1/msrlabs/dedey/

# Copy to rr1
echo "Copying data to rr1"
/home/dedey/Dropbox/philly/linux/philly-fs -cp -r $local_folder $remote_folder_rr1
echo "Finished copying data to rr1"

# Copy to gcr
#echo "Copying data to gcr"
#python /home/dedey/Dropbox/philly/philly_tool/tool_04_19_17_python_2.7/philly-fs.pyc -cp -r $local_folder $remote_folder_gcr
#echo "Finished copying data to gcr"

# Copy to cam
#echo "Copying data to cam"
#python /home/dedey/Dropbox/philly/philly_tool/tool_04_19_17_python_2.7/philly-fs.pyc -cp -r $local_folder $remote_folder_cam
#echo "Finished copying data to cam"

# Copy to eu1
#echo "Copying to eu1"
#python /home/dedey/Dropbox/philly/philly_azure_tool/philly-fs.pyc -cp -r $local_folder $remote_folder_eu1
#echo "Finished copying data to eu1"
