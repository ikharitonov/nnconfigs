#!/bin/bash

echo "SCRIPT STARTED:"
dt=$(date '+%d/%m/%Y %H:%M:%S');
echo "$dt"
working_dir=$(pwd)

# Getting the command line parameters
while getopts m:f:e:l flag
do
    case "${flag}" in
        m) maxjobs=${OPTARG};;
        f) config_file=${OPTARG};;
        e) anaconda_environment=${OPTARG};;
        l) temp_logs_directory=${OPTARG};;
    esac
done

# Creating the temporary logs folder
log_datetime=$(date '+%d_%m_%Y_%H_%M_%S');
# temp_logs_folder="$temp_logs_directory/temp_logs_$log_datetime"
cd $temp_logs_directory
temp_logs_directory=$(pwd)
temp_logs_folder="temp_logs_$log_datetime"
mkdir $temp_logs_folder

# Activating anaconda environment
eval "$(conda shell.bash hook)"
conda activate "$anaconda_environment"
echo "Activated '$anaconda_environment' anaconda environment."

# Saving the contents of csv file into a temporary space-separated text file
awk -F, '(NR>1) {print $1 " " $2 " " $3 " " $4 " " $5 " " $6}' $config_file >> "$temp_logs_folder/temp_arguments.txt"

# Loading the contents of the text file into xargs and launching each script concurrently
cat $temp_logs_folder/temp_arguments.txt | xargs -L 1 -P "$maxjobs" -I {} bash -c ''"$working_dir"'/start_and_monitor.sh $0 '"$temp_logs_directory/$temp_logs_folder" '{}'

wait
echo "All scripts have completed or been terminated."
rm -rf "$temp_logs_folder"
echo "SCRIPT FINISHED:"
dt=$(date '+%d/%m/%Y %H:%M:%S');
echo "$dt"