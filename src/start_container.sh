#!/bin/bash

echo "Running image" $1

cd $(rospack find verifyobject) && docker compose run --rm $1
