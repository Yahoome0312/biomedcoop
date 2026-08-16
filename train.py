import argparse
import torch

from dassl.utils import setup_logger, set_random_seed, collect_env_info
from dassl.config import get_cfg_default
from dassl.engine import build_trainer


import datasets.busi
import datasets.lungcolon
import datasets.chmnist
import datasets.covid
import datasets.btmri
import datasets.ctkidney
import datasets.kvasir
import datasets.retina
import datasets.kneexray
import datasets.dermamnist 
import datasets.octmnist

import trainers.Zeroshot.zeroshot
import trainers.CoOp.coop_clip
import trainers.CoOp.coop_biomedclip
import trainers.CoOp.coop_vpt_biomedclip
import trainers.CoOp.coop_pubmedclip
import trainers.CoOp.coop_pmcclip
import trainers.CoCoOp.cocoop_clip
import trainers.CoCoOp.cocoop_biomedclip
import trainers.CoCoOp.cocoop_pubmedclip
import trainers.CoCoOp.cocoop_pmcclip
import trainers.KgCoOp.kgcoop_clip
import trainers.KgCoOp.kgcoop_biomedclip
import trainers.KgCoOp.kgcoop_pubmedclip
import trainers.KgCoOp.kgcoop_pmcclip
import trainers.ProGrad.prograd_clip
import trainers.ProGrad.prograd_biomedclip
import trainers.ProGrad.prograd_pubmedclip
import trainers.ProGrad.prograd_pmcclip
import trainers.BiomedCoOp.biomedcoop_clip
import trainers.BiomedCoOp.biomedcoop_biomedclip
import trainers.BiomedCoOp.biomedcoop_pubmedclip
import trainers.BiomedCoOp.biomedcoop_pmcclip


def print_args(args, cfg):
    print("***************")
    print("** Arguments **")
    print("***************")
    optkeys = list(args.__dict__.keys())
    optkeys.sort()
    for key in optkeys:
        print("{}: {}".format(key, args.__dict__[key]))
    print("************")
    print("** Config **")
    print("************")
    print(cfg)


def reset_cfg(cfg, args):
    if args.root:
        cfg.DATASET.ROOT = args.root

    if args.output_dir:
        cfg.OUTPUT_DIR = args.output_dir

    if args.resume:
        cfg.RESUME = args.resume

    if args.seed:
        cfg.SEED = args.seed

    if args.source_domains:
        cfg.DATASET.SOURCE_DOMAINS = args.source_domains

    if args.target_domains:
        cfg.DATASET.TARGET_DOMAINS = args.target_domains

    if args.transforms:
        cfg.INPUT.TRANSFORMS = args.transforms

    if args.trainer:
        cfg.TRAINER.NAME = args.trainer

    if args.backbone:
        cfg.MODEL.BACKBONE.NAME = args.backbone

    if args.head:
        cfg.MODEL.HEAD.NAME = args.head



def extend_cfg(cfg):
    """
    Add new config variables.

    E.g.
        from yacs.config import CfgNode as CN
        cfg.TRAINER.MY_MODEL = CN()
        cfg.TRAINER.MY_MODEL.PARAM_A = 1.
        cfg.TRAINER.MY_MODEL.PARAM_B = 0.5
        cfg.TRAINER.MY_MODEL.PARAM_C = False
    """
    from yacs.config import CfgNode as CN

    cfg.DATASET.SUBSAMPLE_CLASSES = "all"  # all, base or new

    cfg.TRAINER.COOP = CN()
    cfg.TRAINER.COOP.N_CTX = 4  # number of context vectors
    cfg.TRAINER.COOP.CSC = False  # class-specific context
    cfg.TRAINER.COOP.CTX_INIT = ""  # initialization words
    cfg.TRAINER.COOP.PREC = "fp32"  # fp16, fp32, amp
    cfg.TRAINER.COOP.CLASS_TOKEN_POSITION = "end"  # 'middle' or 'end' or 'front'

    cfg.TRAINER.COOPVPT = CN()
    cfg.TRAINER.COOPVPT.PREC = "fp32"
    cfg.TRAINER.COOPVPT.VPT_ENABLED = False
    cfg.TRAINER.COOPVPT.VPT_MODE = "deep"
    cfg.TRAINER.COOPVPT.VPT_N_CTX = 5
    cfg.TRAINER.COOPVPT.VPT_DROPOUT = 0.0
    cfg.TRAINER.COOPVPT.VPT_INIT = "uniform"
    cfg.TRAINER.COOPVPT.TEXT_VPT_ENABLED = False
    cfg.TRAINER.COOPVPT.TEXT_VPT_MODE = "deep"
    cfg.TRAINER.COOPVPT.TEXT_VPT_N_CTX = 4
    cfg.TRAINER.COOPVPT.TEXT_VPT_DROPOUT = 0.0
    cfg.TRAINER.COOPVPT.TEXT_VPT_INIT = "normal"
    # One AdamW configuration and one shared LR cover CoOp and VPT prompts.
    cfg.TRAINER.COOPVPT.OPTIM = cfg.OPTIM.clone()
    cfg.TRAINER.COOPVPT.OPTIM.NAME = "adamw"
    cfg.TRAINER.COOPVPT.OPTIM.LR = 0.002
    cfg.TRAINER.COOPVPT.OPTIM.WEIGHT_DECAY = 5e-4
    cfg.TRAINER.COOPVPT.OPTIM.MAX_EPOCH = 100
    cfg.TRAINER.COOPVPT.OPTIM.LR_SCHEDULER = "cosine"
    cfg.TRAINER.COOPVPT.OPTIM.WARMUP_EPOCH = 1
    cfg.TRAINER.COOPVPT.OPTIM.WARMUP_TYPE = "constant"
    cfg.TRAINER.COOPVPT.OPTIM.WARMUP_CONS_LR = 1e-5

    # Textual-based Class-aware Prompt tuning (TCP). These settings are kept
    # separate so disabling TCP preserves the original CoOp+VPT code path.
    cfg.TRAINER.TCP = CN()
    cfg.TRAINER.TCP.ENABLED = False
    cfg.TRAINER.TCP.NUM_TOKENS = 4
    cfg.TRAINER.TCP.BOTTLENECK_DIM = 128
    cfg.TRAINER.TCP.INSERT_LAYER = 8
    cfg.TRAINER.TCP.FUSION_MODE = "replace"
    cfg.TRAINER.TCP.FUSION_WEIGHT = 1.0
    cfg.TRAINER.TCP.KG_WEIGHT = 8.0
    cfg.TRAINER.TCP.KG_MODE = "raw_cosine"
    cfg.TRAINER.TCP.PRIOR_TEMPLATE = "a photo of a {}."
    cfg.TRAINER.TCP.PRIOR_SOURCE = "single_template"
    cfg.TRAINER.TCP.DESCRIPTION_COUNT = 50
    cfg.TRAINER.TCP.DESCRIPTION_BATCH_SIZE = 64
    cfg.TRAINER.TCP.DESCRIPTION_CACHE = ""
    cfg.TRAINER.TCP.LAYER_DESCRIPTION_CACHE = ""
    cfg.TRAINER.TCP.PRIOR_REPRESENTATION = "projected_text"
    cfg.TRAINER.TCP.AGGREGATION = "feature_mean"
    cfg.TRAINER.TCP.CONNECTION = "late_residual"
    cfg.TRAINER.TCP.CONSENSUS_TEMPERATURE = 0.07
    cfg.TRAINER.TCP.GATE_INIT = 0.1
    cfg.TRAINER.TCP.RESIDUAL_WARMUP_EPOCHS = 0
    # Optional warm-start trust region. It penalizes directional drift of the
    # existing CoOp/VPT prompts while the new TCP branch is learned.
    cfg.TRAINER.TCP.PROMPT_ANCHOR_WEIGHT = 0.0
    # Optional scale-aware term inside the same prompt trust region. The
    # squared distance is normalized by the frozen baseline parameter energy.
    cfg.TRAINER.TCP.PROMPT_ANCHOR_L2_WEIGHT = 0.0
    # Include the exact, zero-residual warm-start as epoch 0 in both best-model
    # histories. This is an early-stopping candidate, not another model.
    cfg.TRAINER.TCP.EVAL_WARMSTART = False
    cfg.TRAINER.TCP.DESCRIPTION_KD_WEIGHT = 0.0
    cfg.TRAINER.TCP.DESCRIPTION_KD_TEMPERATURE = 1.5
    cfg.TRAINER.TCP.DESCRIPTION_KD_TAU = 1.5
    # Training-only image-to-description-prior supervision. It updates the
    # same visual prompt branch and is absent from inference logits.
    cfg.TRAINER.TCP.IMAGE_PRIOR_WEIGHT = 0.0
    # Optional internal cross-class alignment: each learned text prototype is
    # contrasted against all frozen 50-description class priors.
    cfg.TRAINER.TCP.PRIOR_CONTRASTIVE_WEIGHT = 0.0
    cfg.TRAINER.TCP.PRIOR_CONTRASTIVE_TEMPERATURE = 0.1
    # Optional layer-8 token supervision from five ordered groups of ten
    # BiomedCoOp descriptions. Available only with the layer_cls bank.
    cfg.TRAINER.TCP.LAYER_TOKEN_ALIGNMENT_WEIGHT = 0.0
    # Training-only, class-balanced alignment between per-batch image
    # centroids and the same model's learned text prototypes. No inference
    # branch or logit fusion is introduced.
    cfg.TRAINER.TCP.CROSS_MODAL_PROTO_WEIGHT = 0.0
    cfg.TRAINER.TCP.CROSS_MODAL_PROTO_TEMPERATURE = 0.1
    # Training-only class-balanced ranking constraint on the same cosine
    # logits used by the classifier. It adds no inference branch or logits.
    cfg.TRAINER.TCP.HARD_NEGATIVE_MARGIN_WEIGHT = 0.0
    cfg.TRAINER.TCP.HARD_NEGATIVE_MARGIN = 0.05
    cfg.TRAINER.TCP.HARD_NEGATIVE_TEMPERATURE = 0.02
    # Optional single-run settling phase: the optimizer still owns all three
    # prompt groups, but only TKE is stepped for the first N epochs.
    cfg.TRAINER.TCP.BASE_PROMPT_FREEZE_EPOCHS = 0
    # Optional one-time initialization of CoOp and Visual VPT from a matching
    # baseline prompt bundle. This is not a resume: TCP stays newly initialized
    # and optimizer/scheduler state always starts from epoch zero.
    cfg.TRAINER.TCP.INIT_BASELINE_CHECKPOINT = ""

    cfg.TRAINER.COCOOP = CN()
    cfg.TRAINER.COCOOP.N_CTX = 4  # number of context vectors
    cfg.TRAINER.COCOOP.CSC = False  # class-specific context
    cfg.TRAINER.COCOOP.CTX_INIT = ""  # initialization words
    cfg.TRAINER.COCOOP.PREC = "fp32"  # fp16, fp32, amp
    cfg.TRAINER.COCOOP.CLASS_TOKEN_POSITION = "end"  # 'middle' or 'end' or 'front'

    cfg.TRAINER.BIOMEDCOOP = CN()
    cfg.TRAINER.BIOMEDCOOP.CTX_INIT = "a photo of a"  # initialization words
    cfg.TRAINER.BIOMEDCOOP.CSC = False  # class-specific context
    cfg.TRAINER.BIOMEDCOOP.CLASS_TOKEN_POSITION = "end"  # 'middle' or 'end' or 'front'
    cfg.TRAINER.BIOMEDCOOP.N_CTX = 4  # number of context vectors
    cfg.TRAINER.BIOMEDCOOP.PREC = "fp32"  # fp16, fp32, amp
    cfg.TRAINER.BIOMEDCOOP.SCCM_LAMBDA = 1.0
    cfg.TRAINER.BIOMEDCOOP.KDSP_LAMBDA = 1.0
    cfg.TRAINER.BIOMEDCOOP.TAU = 1.5
    cfg.TRAINER.BIOMEDCOOP.N_PROMPTS = 50

    cfg.TRAINER.KGCOOP = CN()
    cfg.TRAINER.KGCOOP.CTX_INIT = "a photo of a"  # initialization words
    cfg.TRAINER.KGCOOP.CSC = False  # class-specific context
    cfg.TRAINER.KGCOOP.N_CTX = 4  # number of context vectors
    cfg.TRAINER.KGCOOP.CLASS_TOKEN_POSITION = "end"  # 'middle' or 'end' or 'front'
    cfg.TRAINER.KGCOOP.PREC = "fp32"  # fp16, fp32, amp
    cfg.TRAINER.KGCOOP.W = 1.0

    cfg.TRAINER.PROGRAD = CN()
    cfg.TRAINER.PROGRAD.CTX_INIT = "a photo of a"  # initialization words
    cfg.TRAINER.PROGRAD.CSC = False  # class-specific context
    cfg.TRAINER.PROGRAD.CLASS_TOKEN_POSITION = "end"  # 'middle' or 'end' or 'front'
    cfg.TRAINER.PROGRAD.N_CTX = 4  # number of context vectors
    cfg.TRAINER.PROGRAD.PREC = "fp32"  # fp16, fp32, amp
    cfg.TRAINER.PROGRAD.GM = False
    cfg.TRAINER.PROGRAD.NAME = ""
    cfg.TRAINER.PROGRAD.ALPHA = 0.
    cfg.TRAINER.PROGRAD.T = 1.
    cfg.TRAINER.PROGRAD.LAMBDA = 1.

def setup_cfg(args):
    cfg = get_cfg_default()
    extend_cfg(cfg)

    # 1. From the dataset config file
    if args.dataset_config_file:
        cfg.merge_from_file(args.dataset_config_file)

    # 2. From the method config file
    if args.config_file:
        cfg.merge_from_file(args.config_file)

    # 3. From input arguments
    reset_cfg(cfg, args)

    # 4. From optional input arguments
    cfg.merge_from_list(args.opts)

    cfg.freeze()

    return cfg


def main(args):
    cfg = setup_cfg(args)
    if cfg.SEED >= 0:
        print("Setting fixed seed: {}".format(cfg.SEED))
        set_random_seed(cfg.SEED)
    setup_logger(cfg.OUTPUT_DIR)

    if torch.cuda.is_available() and cfg.USE_CUDA:
        torch.backends.cudnn.benchmark = True

    print_args(args, cfg)
    print("Collecting env info ...")
    # [Reproduction compatibility] PyTorch's collect_env helper can return
    # None/raise on Windows when a system query is unavailable. Environment
    # collection must not prevent the actual experiment from running.
    try:
        env_info = collect_env_info()
    except Exception as exc:
        env_info = "Environment info unavailable: {}: {}".format(
            type(exc).__name__, exc
        )
    print("** System info **\n{}\n".format(env_info))

    trainer = build_trainer(cfg)
    print("Trainer built successfully.")

    if args.eval_only:
        trainer.load_model(args.model_dir, epoch=args.load_epoch)
        trainer.test()
        return

    if not args.no_train:
        trainer.train()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=str, default="", help="path to dataset")
    parser.add_argument("--output-dir", type=str, default="", help="output directory")
    parser.add_argument(
        "--resume",
        type=str,
        default="",
        help="checkpoint directory (from which the training resumes)",
    )
    parser.add_argument(
        "--seed", type=int, default=-1, help="only positive value enables a fixed seed"
    )
    parser.add_argument(
        "--source-domains", type=str, nargs="+", help="source domains for DA/DG"
    )
    parser.add_argument(
        "--target-domains", type=str, nargs="+", help="target domains for DA/DG"
    )
    parser.add_argument(
        "--transforms", type=str, nargs="+", help="data augmentation methods"
    )
    parser.add_argument(
        "--config-file", type=str, default="", help="path to config file"
    )
    parser.add_argument(
        "--dataset-config-file",
        type=str,
        default="",
        help="path to config file for dataset setup",
    )
    parser.add_argument("--trainer", type=str, default="", help="name of trainer")
    parser.add_argument("--backbone", type=str, default="", help="name of CNN backbone")
    parser.add_argument("--head", type=str, default="", help="name of head")
    parser.add_argument("--eval-only", action="store_true", help="evaluation only")
    parser.add_argument(
        "--model-dir",
        type=str,
        default="",
        help="load model from this directory for eval-only mode",
    )
    parser.add_argument(
        "--load-epoch", type=int, help="load model weights at this epoch for evaluation"
    )
    parser.add_argument(
        "--no-train", action="store_true", help="do not call trainer.train()"
    )
    parser.add_argument(
        "opts",
        default=None,
        nargs=argparse.REMAINDER,
        help="modify config options using the command-line",
    )
    args = parser.parse_args()
    main(args)
