#!/bin/bash

echo "SCRIPT STARTED:"
dt=$(date '+%d/%m/%Y %H:%M:%S');
echo "$dt"

log_datetime=$(date '+%d_%m_%Y_%H_%M_%S');
temp_log_directory="/home/ikharitonov/Desktop/temp_logs_$log_datetime"
mkdir "$temp_log_directory"

# Getting the command line parameters
while getopts m:f:e: flag
do
    case "${flag}" in
        m) maxjobs=${OPTARG};;
        f) config_file=${OPTARG};;
        e) anaconda_environment=${OPTARG};;
    esac
done

# Activating anaconda environment
eval "$(conda shell.bash hook)"
conda activate "$anaconda_environment"
echo "Activated '$anaconda_environment' anaconda environment."

# Saving the contents of csv file into a temporary space-separated text file
awk -F, '(NR>1) {print $1 " " $2 " " $3 " " $4 " " $5 " " $6}' $config_file >> $temp_log_directory/temp_arguments.txt

# Loading the contents of the text file into xargs and launching each script concurrently
cat $temp_log_directory/temp_arguments.txt | xargs -L 1 -P "$maxjobs" -I {} bash -c './start_and_monitor.sh $0 '"$temp_log_directory" '{}'

wait
echo "All scripts have completed or been terminated."
rm -rf "$temp_log_directory"
echo "SCRIPT FINISHED:"
dt=$(date '+%d/%m/%Y %H:%M:%S');
echo "$dt"