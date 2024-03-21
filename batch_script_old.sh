#!/bin/bash

echo "SCRIPT STARTED:"
dt=$(date '+%d/%m/%Y %H:%M:%S');
echo "$dt"

kill_process_output="nan"
log_datetime=$(date '+%d_%m_%Y_%H_%M_%S');
temp_log_directory="/home/ikharitonov/Desktop/temp_logs_$log_datetime"
mkdir "$temp_log_directory"

# Function to start a Python script and monitor its output
start_and_monitor() {
    local script_path=$1
    local model_name=$2
    local dataset_name=$3
    local configuration_name=$4
    local configuration_file=$5
    local continue_training=$6
    local output_file="${configuration_name}_output.log"

    # Start the Python script and redirect its output to a log file
    echo $script_path
    python "$script_path" "--model_name" "$model_name" "--dataset_name" "$dataset_name" "--configuration_name" "$configuration_name" "--configuration_file" "$configuration_file" "--continue_training" "$continue_training" > "$temp_log_directory/$output_file" &
    # python "$script_path" &
    local pid=$!

    echo "Started $script_path with PID $pid"

    # # Monitor the output file for a specific text
    # tail -f "$temp_log_directory/$output_file" | grep --line-buffered "$kill_process_output" | while read line; do
    #     echo "'$kill_process_output' output detected in $configuration_name, killing process..."
    #     kill $pid
    #     # pkill -P $$ # Optionally kill the parent process (this script) as well
    #     break
    # done &

    # Instead of tail -f, use a while loop to monitor the log file
    while kill -0 $pid 2>/dev/null; do
        if grep -q "$kill_process_output" "$temp_log_directory/$output_file"; then
            echo "'$kill_process_output' output detected in $configuration_name, killing process..."
            kill $pid
            break
        fi
        sleep 1 # Adjust sleep duration as needed
    done &

    # Limit the number of concurrent jobs
    while [ $(jobs -p | wc -l) -ge $maxjobs ]; do
        # Wait for any background job to finish before continuing
        wait -n
    done
}

# Getting the command line parameters
while getopts m:c:f:e: flag
do
    case "${flag}" in
        m) maxjobs=${OPTARG};;
        c) killcondition=${OPTARG};;
        f) config_file=${OPTARG};;
        e) anaconda_environment=${OPTARG};;
    esac
done

eval "$(conda shell.bash hook)"
conda activate "$anaconda_environment"
echo "Activated '$anaconda_environment' anaconda environment."


awk -F, '(NR>1) {print $1 " " $2 " " $3 " " $4 " " $5 " " $6}' $config_file | while read script_path model_name dataset_name configuration_name configuration_file continue_training
do
    # Start and monitor each Python script in the background
    start_and_monitor "$script_path" "$model_name" "$dataset_name" "$configuration_name" "$configuration_file" "$continue_training"
    # echo "$script_path | $model_name | $dataset_name | $configuration_name | $configuration_file | $continue_training"
done

# Wait for all background jobs to finish
wait
echo "All scripts have completed or been terminated."
# rm -rf "$temp_log_directory"
echo "SCRIPT FINISHED:"
dt=$(date '+%d/%m/%Y %H:%M:%S');
echo "$dt"