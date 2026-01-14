#!/bin/bash
bsub -q x86_1h -mem 24G -cores 1x2+1 -interactive 'cd /dccstor/terratorch/users/wanjiru/terratorch && source .venv/bin/activate && cd integrationtests && time pytest test_base_set.py'
/opt/share/exec/jbsub8 -q x86_1h -mem 24G -cores 1x2+1 -interactive cd /dccstor/terratorch/users/wanjiru/terratorch && source .venv/bin/activate && cd integrationtests && time pytest test_base_set.py