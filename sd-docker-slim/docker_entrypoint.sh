#!/bin/bash

# mkdir models
ln -sf /runpod-volume/models /models
echo "Mount success. Executing CMD now..."

exec "$@"