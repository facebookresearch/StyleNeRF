queue=${1:-dev}
hour=${2:-72}
jname=${3:-critical}
comment=${4:-ACLACL}
constrain=${5:-volta32gb}

srun --job-name=${jname} --gres=gpu:8 -c 48 -C ${constrain} -v --partition=${queue} \
    --exclude 'learnfair7596,learnfair7636,learnfair5056,learnfair5065,learnfair5300,learnfair5133,learnfair5098,learnfair7483,learnfair7498,learnfair0702,learnfair5122,learnfair7611,learnfair5124,learnfair5156,learnfair5036,learnfair5258,learnfair5205,learnfair5201,learnfair5240,learnfair5087,learnfair5119,learnfair5246,learnfair7474,learnfair7585,learnfair5150,learnfair5166,learnfair5215,learnfair5142,learnfair5070,learnfair5236,learnfair7523,learnfair7526' \
    --comment ${comment} --time=${hour}:00:00 --mem 400GB --pty  \
bash
