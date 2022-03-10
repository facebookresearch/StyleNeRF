# training AFHQ data
export NCCL_SOCKET_IFNAME=

DATA=ffhq
OUTDIR=/checkpoint/jgu/space/gan/$DATA/clean
ARCH=stylenerf
mkdir -p ${OUTDIR}

python run_train.py \
    launcher=spawn \
    outdir=${OUTDIR} \
    data=/private/home/jgu/work/stylegan2/datasets/ffhq_512.zip \
    mirror=True \
    spec=paper512 \
    model=stylenerf_ffhq \
    debug=True \
    seed=100 \
    snap=25 \
    aug=noaug \
    +prefix=example \


    