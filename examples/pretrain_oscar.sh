# Script to automatically run OSCAR pretraining

####################################################################
###################### MACROS TO SET HERE ##########################

EPOCHS=0            # 0 results in default, else manually sets value

####################################################################
##### YOU SHOULDN'T HAVE TO TOUCH ANYTHING BELOW THIS POINT :) #####

# Setup env variables
export LD_LIBRARY_PATH=~/anaconda3/envs/oscar/lib/
export CUDA_VISIBLE_DEVICES=0

# Setup python interpreter
source activate oscar

# Get current working directory
temp=$( realpath "$0"  )
DIR=$(dirname "$temp")

# Setup local vars
OSCAR_PATH=${DIR}/..
CFG_PATH=${OSCAR_PATH}/oscar/cfg/train

# Run training

# We procedurally load the cfgs necessary, in the order as follows:
# 1. Base cfg
# 2. Sim cfg
# 3. Agent cfg
# 4. Task cfg
# 5. Controller cfg
# 6. Additional cfg (e.g.: OSCAR-specific settings)

python ${OSCAR_PATH}/oscar/train.py \
--cfg_env ${CFG_PATH}/base.yaml \
--cfg_env_add \
${CFG_PATH}/sim/physx.yaml \
${CFG_PATH}/agent/franka.yaml \
${CFG_PATH}/task/pretrace.yaml \
${CFG_PATH}/controller/oscar.yaml \
${CFG_PATH}/controller/oscar_settings/delan_no_pretrained.yaml \
--device GPU \
--ppo_device GPU \
--max_iterations ${EPOCHS} \
--headless \
--logdir ${OSCAR_PATH}/log \
--experiment_name pretrain
