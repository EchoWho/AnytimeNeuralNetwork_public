#!/bin/bash

#for i in {954..1004}
for i in 474 489 490 520 530 557 567 568 569 583 584 595 610 639 640 641 642 644 647 649 650 651 654 655 656 657 659 661 662 665 666 667 669 670 671 672 695 727 734 735 736 737 739 740 741 742 744 745 746 747 749 781 791 794 796 797 798 799 800 804 805 806 809 812 813 814 815 822 823 824 825 826 827 828 833 835 844 847 851 878 879 880 900 902 932 953 954 956 973 984 986
do

	USERNAME="dedey"
	PASSWORD="GoodCarl2god?"
	CLUSTER="gcr"
	JOBSCRIPT="run_exp_$i.sh"
	SPECIAL_NAME="_ann"
	VC="msrlabs"
	NUM_GPUS="1"

	CMD="https://philly/api/submit?"
	CMD+="buildId=0000&"
	CMD+="customDockerName=custom-tf-0-12-python-2-7-ver2&"
	CMD+="toolType=cust&"
	CMD+="clusterId=$CLUSTER&"
	CMD+="vcId=$VC&"
	CMD+="configFile=$USERNAME%2FAnytimeNeuralNetwork%2F$JOBSCRIPT&"
	CMD+="minGPUs=$NUM_GPUS&"
	CMD+="name=cust-p-$JOBSCRIPT$SPECIAL_NAME!~!~!1&"
	CMD+="isdebug=false&"
	CMD+="iscrossrack=false&"
	CMD+="inputDir=%2Fhdfs%2F$VC%2F$USERNAME%2Fann_data_dir%2F&"
	CMD+="oneProcessPerContainer=true&"
	CMD+="userName=$USERNAME"

	curl -k --ntlm --user "$USERNAME:$PASSWORD" "$CMD"

	echo "$CMD"

	# FOR WHEN YOU NEED IMAGENET	
	# CMD+="inputDir=%2Fhdfs%2F$VC%2F$USERNAME%2Fann_data_dir%2Fimagenet_tfrecords%2F&"

	# FOR WHEN YOU NEED OTHER DATASETS
	# CMD+="inputDir=%2Fhdfs%2F$VC%2F$USERNAME%2Fann_data_dir%2F&"

done
