# MADDPG Installation

## Preparation

conda create -n maddpg python=3.6

conda activate maddpg

conda install tensorflow-gpu==1.15

## MADDPG

git clone https://github.com/Ray0117/maddpg.git

cd maddpg

pip install -e .

## Multi-Agent Particle Environment

cd multiagent-particle-envs

pip install -e .

## Necessary Dependencies

pip install imageio pyglet==1.3.2 gym==0.10.5 //version must be satisfied

