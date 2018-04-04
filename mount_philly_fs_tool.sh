# This mounts the philly_fs tool to /mnt/philly-fs. It assumes that /mnt/philly_fs exists already. See philly documentation on this.

sudo mount -t cifs -o username=dedey,domain=redmond,file_mode=0777,dir_mode=0777,vers=2.0 //scratch2.ntdev.corp.microsoft.com/scratch/Philly/philly-fs /mnt/philly-fs