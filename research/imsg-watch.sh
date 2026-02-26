#!/bin/bash
# Watch multiple iMessage chat IDs and merge NDJSON output to stdout.
# Used by com.research.imsg-watcher.plist as the Program.
#
# Chat 4: +12104265298 (phone number identifier)
# Chat 5: martin.jose.gonzalez@gmail.com (email identifier)

imsg watch --chat-id 4 --json &
PID1=$!

imsg watch --chat-id 5 --json &
PID2=$!

# If either exits, kill the other and exit (launchd will restart us)
trap "kill $PID1 $PID2 2>/dev/null; exit 0" SIGTERM SIGINT

wait -n
kill $PID1 $PID2 2>/dev/null
wait
