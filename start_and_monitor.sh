#!/bin/bash

# Function to start a Python script and monitor its output
script_path=$1
model_name=$2
dataset_name=$3
configuration_name=$4
configuration_file=$5
continue_training=$6
temp_log_directory=$7
output_file="${configuration_name}_output.log"
kill_process_output="nan"

# Start the Python script and redirect its output to a log file
python "$script_path" "--model_name" "$model_name" "--dataset_name" "$dataset_name" "--configuration_name" "$configuration_name" "--configuration_file" "$configuration_file" "--continue_training" "$continue_training" > "$temp_log_directory/$output_file" &
pid=$!

echo "Started $configuration_name with PID $pid"

# Launch a background job for monitoring
(
    while kill -0 $pid 2>/dev/null; do
        if grep -q "$kill_process_output" "$temp_log_directory/$output_file"; then
            echo "'$kill_process_output' output detected in $configuration_name, killing process..."
            kill $pid
            break
        fi
        sleep 1 # Adjust sleep duration as needed
    done
) &

wait $pid  # Wait for the process to complete
echo "Process $pid: $configuration_name has finished."  # Echo text after the process finishes