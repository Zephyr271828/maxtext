set -x 

bash scripts/eval/llama3.1-8b_wanda_2:4_L200_S50.sh --lr=3e-5

bash scripts/eval/llama3.1-8b_wanda_2:4_S50.sh --lr=1e-3

bash scripts/eval/llama3.1-8b_sparsegpt_unstructured_L200_S50.sh --lr=3e-5

bash scripts/eval/llama3.1-8b_sparsegpt_unstructured_S50.sh --lr=1e-3

bash scripts/eval/llama3.1-8b_sparsegpt_2:4_S50.sh --lr=1e-3

bash scripts/eval/llama3.1-8b_sparsegpt_2:4_L200_S50.sh --lr=3e-5

bash scripts/eval/llama3.1-4b-depth_S50.sh --lr=5e-4

bash scripts/eval/llama3.1-4b-depth_L200_S50.sh --lr=5e-4

bash scripts/eval/llama3.1-4b-depth_S250.sh --lr=3e-4

bash scripts/eval/llama3.1-4b-width_S50.sh --lr=3e-4

bash scripts/eval/llama3.1-4b-width_L200_S50.sh --lr=0.0003

bash scripts/eval/llama3.1-4b-width_S250.sh --lr=3e-4

bash scripts/eval/llama3.1-1.5b-depth_S50.sh --lr=0.0003

bash scripts/eval/llama3.1-1.5b-depth_L200_S50.sh --lr=0.0003 

bash scripts/eval/llama3.1-2b-depth_S50.sh --lr=0.0003

bash scripts/eval/llama3.1-2b-depth_L200_S50.sh --lr=0.0003     

bash scripts/eval/llama3.1-3b-depth_S50.sh --lr=0.0003

bash scripts/eval/llama3.1-3b-depth_L200_S50.sh --lr=0.0003 