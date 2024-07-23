#!/bin/bash

FILE_TO_MONITOR="./index.md"
COMMAND_TO_RUN="make build-slides"

# Ensure inotifywait is installed
if ! command -v inotifywait &> /dev/null
then
  echo "inotifywait could not be found, please install inotify-tools"
  exit 1
fi

# Monitor the file for changes and run the command when a change occurs
while true; do
  inotifywait -e modify,create,delete $FILE_TO_MONITOR
  echo "Change detected in $FILE_TO_MONITOR, running command: $COMMAND_TO_RUN"
  $COMMAND_TO_RUN
done
