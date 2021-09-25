# Script to automatically test trained model

####################################################################
###################### MACROS TO SET HERE ##########################

CONDA_PATH=~/anaconda3/envs/oscar/lib/  # Should be absolute path to your conda oscar env lib directory
TASK=push                               # Options are: {trace, pour, push}
CONTROLLER=oscar                        # Options are: {oscar, osc, osc_no_vices, ik, joint_tor, joint_vel, joint_pos}
N_ENVS=4                                # Number of test environments to use
N_EPISODES=40                           # Number of test episodes to use
RENDER=true                             # Either {true, false}; whether to render onscreen or not
PRETRAINED_MODEL=default                # Should set this to absolute fpath to checkpoint .pth file. Setting this to "default" results in default pretrained model being loaded
TEST_ZERO_SHOT=false                    # Either {true, false}; whether to test model under zero-shot OOD setting

####################################################################
##### YOU SHOULDN'T HAVE TO TOUCH ANYTHING BELOW THIS POINT :) #####

# Setup env variables
export LD_LIBRARY_PATH="${CONDA_PATH}"
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

# Process macros

# Define config(s) for controller
CONTROLLER_CONFIGS="${CFG_PATH}/controller/${CONTROLLER}.yaml"
if [ ${CONTROLLER} == "oscar" ]
then
  CONTROLLER_CONFIGS+=" ${CFG_PATH}/controller/oscar_settings/delan_residual.yaml"
  CONTROLLER_CONFIGS+=" ${CFG_PATH}/controller/oscar_settings/delan_no_pretrained.yaml"
fi

# Determine agent to use for task
declare -A TASK_AGENT_MAPPING
TASK_AGENT_MAPPING[trace]="franka_pitcher"
TASK_AGENT_MAPPING[pour]="franka_pitcher"
TASK_AGENT_MAPPING[push]="franka"

EVAL_CONFIGS="${CFG_PATH}/common/eval.yaml"
if [ ${TEST_ZERO_SHOT} == "true" ]
then
  EVAL_CONFIGS+=" ${CFG_PATH}/task/${TASK}_zero_shot.yaml"
fi

OTHER_ARGS=" "
if [ ${RENDER} == "false" ]
then
  OTHER_ARGS+="--headless "
fi


# Determine device to use -- Pour must use CPU
DEVICE=""
if [ ${TASK} == "pour" ]
then
  DEVICE+="CPU"
else
  DEVICE+="GPU"
fi

# Load default checkpoint if default is specified
declare -A TASK_MAPPING
TASK_MAPPING[trace]="Trace"
TASK_MAPPING[pour]="Pour"
TASK_MAPPING[push]="Push"
TASK_UPPERCASE=${TASK_MAPPING[${TASK}]}
if [ ${PRETRAINED_MODEL} == "default" ]
then
  PRETRAINED_MODEL="${OSCAR_PATH}/trained_models/train/${TASK_UPPERCASE}/${TASK_UPPERCASE}_${CONTROLLER}__seed_1.pth"
fi


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
${CFG_PATH}/agent/${TASK_AGENT_MAPPING[${TASK}]}.yaml \
${CFG_PATH}/task/${TASK}.yaml \
${CONTROLLER_CONFIGS} \
${EVAL_CONFIGS} \
--device ${DEVICE} \
--ppo_device GPU \
--num_envs ${N_ENVS} \
--test \
--num_test_episodes ${N_EPISODES} \
--checkpoint ${PRETRAINED_MODEL} \
${OTHER_ARGS}
