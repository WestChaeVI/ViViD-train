import os
os.environ["CUDA_VISIBLE_DEVICES"] = "3,4,5,6"
os.environ['NCCL_P2P_LEVEL'] = 'NVL'
os.environ['NCCL_SOCKET_IFNAME'] = 'eth0'
os.environ['NCCL_SOCKET_IFNAME'] = 'enp3s0f1'
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "max_split_size_mb:128"
os.environ["TORCH_DISTRIBUTED_DEBUG"] = "DETAIL"

import cv2
import copy
import math
import glob
import time
import mlflow
import random
import logging
import inspect
import warnings
import builtins
import argparse
import deepspeed
from datetime import datetime
from PIL import Image
from tqdm.auto import tqdm
from copy import deepcopy
from pathlib import Path
from tempfile import TemporaryDirectory
from einops import rearrange
from omegaconf import OmegaConf
from collections import OrderedDict
from typing import Callable, List, Optional, Union

import torch
import torch.nn as nn
import torch.optim as optim
import torch.distributed as dist
import torch.utils.checkpoint
import torch.nn.functional as F
import torch.multiprocessing as mp
from torchvision import transforms
from torch.utils.tensorboard import SummaryWriter
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader, Dataset, random_split

from accelerate import Accelerator
from accelerate.logging import get_logger
from accelerate.utils import DistributedDataParallelKwargs
from torch.cuda.amp import autocast, GradScaler

import diffusers
from diffusers.utils import check_min_version
from diffusers import AutoencoderKL, DDIMScheduler
from diffusers.optimization import get_scheduler
from diffusers.image_processor import VaeImageProcessor
from diffusers.utils.import_utils import is_xformers_available
from diffusers.schedulers import (
    DDIMScheduler,
    # DPMSolverMultistepScheduler,
    # EulerAncestralDiscreteScheduler,
    # EulerDiscreteScheduler,
    # LMSDiscreteScheduler,
    # PNDMScheduler,
)
import traceback
import transformers

from dataset import LouisVTONDataset, LouisVTONDataset_front

from transformers import CLIPVisionModelWithProjection, CLIPImageProcessor

from src.models.pose_guider import PoseGuider
from src.models.unet_2d_condition import UNet2DConditionModel
from src.models.unet_3d import UNet3DConditionModel
from src.pipelines.pipeline_pose2vid_long import Pose2VideoPipeline
from src.models.mutual_self_attention import ReferenceAttentionControl
from src.pipelines.context import get_context_scheduler
from src.pipelines.utils import get_tensor_interpolation_method
from src.utils.util import (
    get_fps,
    delete_additional_ckpt,
    import_filename,
    read_frames,
    save_videos_grid,
    seed_everything
)

warnings.filterwarnings("ignore")


logger = get_logger(__name__, log_level="INFO")

class Net(nn.Module):
    def __init__(
        self,
        reference_unet: UNet2DConditionModel,
        denoising_unet: UNet3DConditionModel,
        pose_guider: PoseGuider,
        reference_control_writer,
        reference_control_reader,
    ):
        super().__init__()
        self.reference_unet = reference_unet
        self.denoising_unet = denoising_unet
        self.pose_guider = pose_guider
        self.reference_control_writer = reference_control_writer
        self.reference_control_reader = reference_control_reader

    def forward(
        self,
        noisy_latents,
        timesteps,
        ref_image_latents,
        clip_image_embeds,
        pose_img,
        # uncond_fwd: bool = False
    ):
        pose_cond_tensor = pose_img.to(device="cuda")
        pose_fea = self.pose_guider(pose_cond_tensor)
        
        b, c, f, h, w = noisy_latents.shape
        
        # if not uncond_fwd:
        #     ref_timesteps = torch.zeros_like(timesteps)
        #     self.reference_unet(
        #         ref_image_latents,
        #         ref_timesteps,
        #         encoder_hidden_states=clip_image_embeds,
        #         return_dict=False,
        #     )
        #     self.reference_control_reader.update(self.reference_control_writer)
        
        ref_timesteps = torch.zeros_like(timesteps)
        self.reference_unet(
            ref_image_latents,
            ref_timesteps,
            encoder_hidden_states=clip_image_embeds,
            return_dict=False,
        )
        self.reference_control_reader.update(self.reference_control_writer)

        model_pred = self.denoising_unet(
            noisy_latents,
            timesteps,
            pose_cond_fea=pose_fea,
            encoder_hidden_states=clip_image_embeds[:b],
        ).sample
        
        if model_pred is None:
            raise ValueError("model_pred is None. Check denoising_unet output.")

        return model_pred
    
    
    
def compute_snr(noise_scheduler, timesteps):
    """
    Computes SNR as per
    https://github.com/TiankaiHang/Min-SNR-Diffusion-Training/blob/521b624bd70c67cee4bdf49225915f5945a872e3/guided_diffusion/gaussian_diffusion.py#L847-L849
    """
    alphas_cumprod = noise_scheduler.alphas_cumprod
    sqrt_alphas_cumprod = alphas_cumprod**0.5
    sqrt_one_minus_alphas_cumprod = (1.0 - alphas_cumprod) ** 0.5

    # Expand the tensors.
    # Adapted from https://github.com/TiankaiHang/Min-SNR-Diffusion-Training/blob/521b624bd70c67cee4bdf49225915f5945a872e3/guided_diffusion/gaussian_diffusion.py#L1026
    sqrt_alphas_cumprod = sqrt_alphas_cumprod.to(device=timesteps.device)[
        timesteps
    ].float()
    while len(sqrt_alphas_cumprod.shape) < len(timesteps.shape):
        sqrt_alphas_cumprod = sqrt_alphas_cumprod[..., None]
    alpha = sqrt_alphas_cumprod.expand(timesteps.shape)

    sqrt_one_minus_alphas_cumprod = sqrt_one_minus_alphas_cumprod.to(
        device=timesteps.device
    )[timesteps].float()
    while len(sqrt_one_minus_alphas_cumprod.shape) < len(timesteps.shape):
        sqrt_one_minus_alphas_cumprod = sqrt_one_minus_alphas_cumprod[..., None]
    sigma = sqrt_one_minus_alphas_cumprod.expand(timesteps.shape)

    # Compute SNR.
    snr = (alpha / sigma) ** 2
    return snr


def log_validation(
    vae,
    image_enc,
    net,
    scheduler,
    accelerator,
    width,
    height,
    clip_length=30,
    generator=None,
):
    logger.info("Running validation... ")

    ori_net = accelerator.unwrap_model(net)
    reference_unet = ori_net.reference_unet.to(dtype=torch.float16)
    denoising_unet = ori_net.denoising_unet.to(dtype=torch.float16)
    pose_guider = ori_net.pose_guider.to(dtype=torch.float16)

    if generator is None:
        generator = torch.manual_seed(42)
    # tmp_denoising_unet = copy.deepcopy(denoising_unet)
    # tmp_denoising_unet = tmp_denoising_unet.to(dtype=torch.float16)

    pipe = Pose2VideoPipeline(
        vae=vae,
        image_encoder=image_enc,
        reference_unet=reference_unet,
        denoising_unet=denoising_unet,
        pose_guider=pose_guider,
        scheduler=scheduler,
    )
    pipe = pipe.to(accelerator.device)

    test_cases = [
        (
            "/home/vton/westchaevi/ViViD/data_cleve/test/cloth/0050.jpg",           # cloth
            "/home/vton/westchaevi/ViViD/data_cleve/test/videos/0050.mp4",          # video
        ),
        (
            "/home/vton/westchaevi/ViViD/data_cleve/test/cloth/0150.jpg",           # cloth
            "/home/vton/westchaevi/ViViD/data_cleve/test/videos/0150.mp4",          # video
        ),
    ]

    results = []
    for test_case in test_cases:
        ref_image_path, video_path = test_case
        video_name = Path(video_path).stem
        
        ref_image_mask_path = ref_image_path.replace("cloth","cloth_mask")
        agnostic_path = video_path.replace("videos","agnostic")
        agn_mask_path = video_path.replace("videos","agnostic_mask")
        densepose_path = video_path.replace("videos","densepose")
        
        transform = transforms.Compose(
            [transforms.Resize((height, width)), transforms.ToTensor()]
        )
        
        video_tensor_list=[]
        video_images=read_frames(video_path)
        for vid_image_pil in video_images[:clip_length]:
            video_tensor_list.append(transform(vid_image_pil))

        video_tensor = torch.stack(video_tensor_list, dim=0)  # (f, c, h, w)
        video_tensor = video_tensor.transpose(0, 1)
        
        agnostic_list=[]
        agnostic_images=read_frames(agnostic_path)
        for agnostic_image_pil in agnostic_images[:clip_length]:
            agnostic_list.append(agnostic_image_pil)

        agn_mask_list=[]
        agn_mask_images=read_frames(agn_mask_path)
        for agn_mask_image_pil in agn_mask_images[:clip_length]:
            agn_mask_list.append(agn_mask_image_pil)

        pose_list=[]
        pose_images=read_frames(densepose_path)
        for pose_image_pil in pose_images[:clip_length]:
            pose_list.append(pose_image_pil)
            
        cloth_image_pil = Image.open(ref_image_path).convert("RGB")
        cloth_mask_pil = Image.open(ref_image_mask_path).convert("RGB")
        

        pipeline_output = pipe(
            agnostic_list,
            agn_mask_list,
            cloth_image_pil,
            cloth_mask_pil,
            pose_list,
            width,
            height,
            clip_length,
            num_inference_steps=20,
            guidance_scale=3.5,
            generator=generator,
            interpolation_factor=1,
        )
        video = pipeline_output.videos

        video_tensor = video_tensor.unsqueeze(0)
        video = torch.cat([video, video_tensor], dim=0)

        results.append({"name": f"{video_name}", "vid": video})

    # del tmp_denoising_unet
    del pipe
    torch.cuda.empty_cache()

    return results

def initialize_model(cfg, infer_config, weight_dtype):
    vae = AutoencoderKL.from_pretrained(cfg.vae_model_path).to("cuda", dtype=weight_dtype)
    image_enc = CLIPVisionModelWithProjection.from_pretrained(cfg.image_encoder_path).to(dtype=weight_dtype, device="cuda")
    reference_unet = UNet2DConditionModel.from_pretrained_2d(cfg.base_model_path, subfolder="unet", unet_additional_kwargs={"in_channels": 5}).to(device="cuda")
    denoising_unet = UNet3DConditionModel.from_pretrained_2d(cfg.base_model_path, cfg.mm_path, subfolder="unet", unet_additional_kwargs=OmegaConf.to_container(infer_config.unet_additional_kwargs)).to(device="cuda")
    pose_guider = PoseGuider(conditioning_embedding_channels=320, block_out_channels=(16, 32, 96, 256)).to(device="cuda", dtype=torch.float32)

    # dist.barrier()
    denoising_unet.load_state_dict(torch.load(cfg.denoising_unet_path, map_location="cpu"), strict=False)
    reference_unet.load_state_dict(torch.load(cfg.reference_unet_path, map_location="cpu"), strict=False)
    pose_guider.load_state_dict(torch.load(cfg.pose_guider_path, map_location="cpu"), strict=False)
    

    return vae, image_enc, reference_unet, denoising_unet, pose_guider


def verify_model_parameters(model):
    param_count = sum(p.numel() for p in model.parameters())
    param_count_tensor = torch.tensor(param_count, dtype=torch.int64, device='cuda')
    dist.all_reduce(param_count_tensor)
    if param_count_tensor.item() != param_count * dist.get_world_size():
        raise ValueError(f"Model parameters do not match across all processes. "
                         f"Rank {dist.get_rank()} has {param_count} params, "
                         f"expected {param_count_tensor.item() / dist.get_world_size()}.")
        
    dist.barrier()
    

        

def main(cfg):  # cfg : /home/vton/westchaevi/ViViD/configs/train/train_stage.yaml

    if os.path.exists(cfg.output_dir):
        remove_checkpoints(cfg.output_dir)
        
    kwargs = DistributedDataParallelKwargs(find_unused_parameters=True)
    accelerator = Accelerator(
        gradient_accumulation_steps=cfg.solver.gradient_accumulation_steps,
        mixed_precision=cfg.solver.mixed_precision,
        log_with="mlflow",
        project_dir="./mlruns",
        kwargs_handlers=[kwargs],
    )
    
    if accelerator.is_main_process:
        remove_checkpoints(cfg.output_dir)

    # Make one log on every process with the configuration for debugging.
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        level=logging.INFO,
    )
    logger.info(accelerator.state, main_process_only=False)
    if accelerator.is_local_main_process:
        transformers.utils.logging.set_verbosity_warning()
        diffusers.utils.logging.set_verbosity_info()
    else:
        transformers.utils.logging.set_verbosity_error()
        diffusers.utils.logging.set_verbosity_error()

    # If passed along, set the training seed now.
    if cfg.seed is not None:
        seed_everything(cfg.seed)

    exp_name = cfg.exp_name
    save_dir = f"{cfg.output_dir}/{exp_name}"
    if accelerator.is_main_process:
        if not os.path.exists(save_dir):
            os.makedirs(save_dir)
    
    inference_config_path = "./configs/train/train.yaml"
    infer_config = OmegaConf.load(inference_config_path)

    if cfg.weight_dtype == "fp16":
        weight_dtype = torch.float16
    elif cfg.weight_dtype == "fp32":
        weight_dtype = torch.float32
    else:
        raise ValueError(
            f"Do not support weight dtype: {cfg.weight_dtype} during training"
        )

    sched_kwargs = OmegaConf.to_container(cfg.noise_scheduler_kwargs)
    if cfg.enable_zero_snr:
        sched_kwargs.update(
            rescale_betas_zero_snr=True,
            timestep_spacing="trailing",
            prediction_type="v_prediction",
        )
    val_noise_scheduler = DDIMScheduler(**sched_kwargs)
    sched_kwargs.update({"beta_schedule": "scaled_linear"})
    train_noise_scheduler = DDIMScheduler(**sched_kwargs)
    
    
    print(f"Rank {dist.get_rank()} initializing models")
    vae, image_enc, reference_unet, denoising_unet, pose_guider = initialize_model(cfg, infer_config, weight_dtype)
    print(f"Rank {dist.get_rank()} models initialized")

    verify_model_parameters(reference_unet)
    print(f"Rank {dist.get_rank()} reference_unet parameters verified")
    verify_model_parameters(denoising_unet)
    print(f"Rank {dist.get_rank()} denoising_unet parameters verified")
    verify_model_parameters(pose_guider)
    print(f"Rank {dist.get_rank()} pose_guider parameters verified")
    
    verify_model_parameters(reference_unet)
    verify_model_parameters(denoising_unet)
    verify_model_parameters(pose_guider)
    
    clip_image_processor = CLIPImageProcessor()
    vae_scale_factor = 2 ** (len(vae.config.block_out_channels) - 1)
    ref_image_processor = VaeImageProcessor(vae_scale_factor=vae_scale_factor, do_convert_rgb=True)
    
    # Freeze
    vae.requires_grad_(False)
    image_enc.requires_grad_(False)
    
#     #  Some top layer parames of reference_unet don't need grad
#     for name, param in reference_unet.named_parameters():
#         if "up_blocks.3" in name:
#             param.requires_grad_(False)
#         else:
#             param.requires_grad_(True)
            
    reference_unet.requires_grad_(True)
    denoising_unet.requires_grad_(True)
    pose_guider.requires_grad_(True)
    
    reference_control_writer = ReferenceAttentionControl(
        reference_unet,
        do_classifier_free_guidance=False,
        mode="write",
        fusion_blocks="full",
    )
    reference_control_reader = ReferenceAttentionControl(
        denoising_unet,
        do_classifier_free_guidance=False,
        mode="read",
        fusion_blocks="full",
    )
    
    
    net = Net(
        reference_unet,
        denoising_unet,
        pose_guider,
        reference_control_writer,
        reference_control_reader,
    )
    
    if cfg.solver.enable_xformers_memory_efficient_attention:
        if is_xformers_available():
            reference_unet.enable_xformers_memory_efficient_attention()
            denoising_unet.enable_xformers_memory_efficient_attention()
        else:
            raise ValueError(
                "xformers is not available. Make sure it is installed correctly"
            )

    if cfg.solver.gradient_checkpointing:
        reference_unet.enable_gradient_checkpointing()
        denoising_unet.enable_gradient_checkpointing()


    if cfg.solver.scale_lr:
        learning_rate = (
            cfg.solver.learning_rate
            * cfg.solver.gradient_accumulation_steps
            * cfg.data.train_bs
            * accelerator.num_processes
        )
    else:
        learning_rate = cfg.solver.learning_rate

    # Initialize the optimizer
    if cfg.solver.use_8bit_adam:
        try:
            import bitsandbytes as bnb
        except ImportError:
            raise ImportError(
                "Please install bitsandbytes to use 8-bit Adam. You can do so by running `pip install bitsandbytes`"
            )

        optimizer_cls = bnb.optim.AdamW8bit
    else:
        optimizer_cls = torch.optim.AdamW
        
        
    trainable_params = list(filter(lambda p: p.requires_grad, net.parameters()))
    logger.info(f"Total trainable params {len(trainable_params)}")
    optimizer = deepspeed.ops.adam.FusedAdam(
        trainable_params,
        lr=learning_rate,
        betas=(cfg.solver.adam_beta1, cfg.solver.adam_beta2),
        weight_decay=cfg.solver.adam_weight_decay,
        eps=cfg.solver.adam_epsilon,
    )
    

    # Scheduler
    lr_scheduler = get_scheduler(
        cfg.solver.lr_scheduler,
        optimizer=optimizer,
        num_warmup_steps=cfg.solver.lr_warmup_steps
        * cfg.solver.gradient_accumulation_steps,
        num_training_steps=cfg.solver.max_train_steps
        * cfg.solver.gradient_accumulation_steps,
    )
    
    data_transform = transforms.Compose([
        transforms.Resize((cfg.data.train_height, cfg.data.train_width)),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.RandomAffine(degrees=0, translate=(0.2, 0.2), scale=(0.8, 1.2)),
        transforms.ToTensor()])
    
    # Train loader
    train_dataset = LouisVTONDataset_front(base_dir=cfg.data_dir, mode='train', transforms=data_transform, clip_length=cfg.data.n_sample_frames)
    # train_sampler = torch.utils.data.distributed.DistributedSampler(train_dataset)
    train_dataloader = DataLoader(train_dataset, batch_size=cfg.data.train_bs, num_workers=8, pin_memory=True, drop_last=True)
    print("Load train_loader")

    context_schedule="uniform"
    context_frames=30,
    context_stride=1,
    context_overlap=4,
    context_batch_size=1,
    interpolation_factor=1,

    # Prepare everything with our `accelerator`.
    (
        net,
        optimizer,
        train_dataloader,
        lr_scheduler,
    ) = accelerator.prepare(
        net,
        optimizer,
        train_dataloader,
        lr_scheduler,
    )
    
    # We need to recalculate our total training steps as the size of the training dataloader may have changed.
    num_update_steps_per_epoch = math.ceil(
        len(train_dataloader) / cfg.solver.gradient_accumulation_steps
    )
    # Afterwards we recalculate our number of training epochs
    num_train_epochs = math.ceil(
        cfg.solver.max_train_steps / num_update_steps_per_epoch
    )

    print("initialize the trackers")
    # We need to initialize the trackers we use, and also store our configuration.
    # The trackers initializes automatically on the main process.
    if accelerator.is_main_process:
        run_time = datetime.now().strftime("%Y%m%d-%H%M")
        accelerator.init_trackers(
            exp_name,
            init_kwargs={"mlflow": {"run_name": run_time}},
        )
        # dump config file
        mlflow.log_dict(OmegaConf.to_container(cfg), "config.yaml")

    # Train!
    total_batch_size = (
        cfg.data.train_bs
        * accelerator.num_processes
        * cfg.solver.gradient_accumulation_steps
    )

    logger.info("***** Running training *****")
    logger.info(f"  Num examples = {len(train_dataset)}")
    logger.info(f"  Num Epochs = {num_train_epochs}")
    logger.info(f"  Instantaneous batch size per device = {cfg.data.train_bs}")
    logger.info(
        f"  Total train batch size (w. parallel, distributed & accumulation) = {total_batch_size}"
    )
    logger.info(
        f"  Gradient Accumulation steps = {cfg.solver.gradient_accumulation_steps}"
    )
    logger.info(f"  Total optimization steps = {cfg.solver.max_train_steps}")
    global_step = 0
    first_epoch = 0

    # Potentially load in the weights and states from a previous save
    if cfg.resume_from_checkpoint:
        if cfg.resume_from_checkpoint != "latest":
            resume_dir = cfg.resume_from_checkpoint
        else:
            resume_dir = save_dir
        # Get the most recent checkpoint
        dirs = os.listdir(resume_dir)
        dirs = [d for d in dirs if d.startswith("checkpoint")]
        dirs = sorted(dirs, key=lambda x: int(x.split("-")[1]))
        path = dirs[-1]
        accelerator.load_state(os.path.join(resume_dir, path))
        accelerator.print(f"Resuming from checkpoint {path}")
        global_step = int(path.split("-")[1])

        first_epoch = global_step // num_update_steps_per_epoch
        resume_step = global_step % num_update_steps_per_epoch

    # Only show the progress bar once on each machine.
    progress_bar = tqdm(
        range(global_step, cfg.solver.max_train_steps),
        disable=not accelerator.is_local_main_process,
        miniters=1,
    )
    progress_bar.set_description("Steps")
    

    for epoch in range(first_epoch, num_train_epochs):
        train_loss = 0.0
        t_data_start = time.time()
        
        for step, batch in enumerate(train_dataloader):
            t_data = time.time() - t_data_start
            print(f"Rank: {dist.get_rank()}, Epoch: {epoch}, Step: {step}")
            
            with accelerator.accumulate(net):
                video_frames = batch['video_frames'].to(weight_dtype)
                agnostic_frames = batch['agnostic_frames']
                agn_mask_frames = batch['agn_mask_frames']
                pose_frames = batch['densepose_frames']
                cloth_images = batch['cloth_image']
                cloth_masks = batch['cloth_mask']

                with torch.no_grad():
                    video_length = video_frames.shape[1]
                    video_frames = rearrange(video_frames, "b f c h w -> (b f) c h w")
                    latents = vae.encode(video_frames).latent_dist.sample()
                    latents = rearrange(latents, "(b f) c h w -> b c f h w", f=video_length)
                    latents = latents * 0.18215
                    latents = latents.to(denoising_unet.device)

                noise = torch.randn_like(latents).to(denoising_unet.device)
                if cfg.noise_offset > 0:
                    noise += cfg.noise_offset * torch.randn((latents.shape[0], latents.shape[1], 1, 1, 1), device=denoising_unet.device)
                bsz = latents.shape[0]
                timesteps = torch.randint(0, train_noise_scheduler.num_train_timesteps, (bsz,), device=denoising_unet.device).long()
                uncond_fwd = random.random() < cfg.uncond_ratio

                clip_image_list = []
                for cloth in cloth_images:
                    cloth = transforms.ToPILImage()(cloth)
                    cloth_resized = cloth.resize((224, 224))
                    clip_image = clip_image_processor.preprocess(cloth_resized, return_tensors="pt").pixel_values
                    clip_image_list.append(clip_image)
                clip_img = torch.cat(clip_image_list).to(device="cuda", dtype=weight_dtype)
                clip_image_embeds = image_enc(clip_img.to(dtype=image_enc.dtype)).image_embeds
                encoder_hidden_states = clip_image_embeds.unsqueeze(1)

                ref_image_list = []
                for cloth in cloth_images:
                    with torch.no_grad():
                        ref_image_tensor = ref_image_processor.preprocess(cloth, height=cfg.data.train_height, width=cfg.data.train_width)
                        ref_image_tensor = ref_image_tensor.to(dtype=vae.dtype, device=vae.device)
                        ref_image_latents = vae.encode(ref_image_tensor).latent_dist.sample()
                        ref_image_latents = ref_image_latents * 0.18215
                        ref_image_list.append(ref_image_latents)
                ref_image_latents = torch.cat(ref_image_list)

                agn_tensor_list = []
                for agn in agnostic_frames:
                    agn_tensor = ref_image_processor.preprocess(agn, height=cfg.data.train_height, width=cfg.data.train_width)
                    agn_tensor_list.append(agn_tensor)
                agn_tensor = torch.cat(agn_tensor_list, dim=0).to(dtype=vae.dtype, device=vae.device)
                with torch.no_grad():
                    agnostic_image_latents = vae.encode(agn_tensor).latent_dist.sample()
                    agnostic_image_latents = agnostic_image_latents * 0.18215
                    agnostic_image_latents = rearrange(agnostic_image_latents, "(b f) c h w -> b c f h w", f=video_length)

                agn_mask_list = []
                for mask in agn_mask_frames:
                    mask = ref_image_processor.preprocess(mask, height=cfg.data.train_height, width=cfg.data.train_width)
                    mask = ref_image_processor.normalize(mask)
                    mask = ref_image_processor.binarize(mask)
                    mask = mask[:, 0:1, :, :]
                    agn_mask_list.append(mask)
                agn_mask = torch.cat(agn_mask_list, dim=0)
                agn_mask = torch.nn.functional.interpolate(agn_mask, size=(agn_mask.shape[-2] // 8, agn_mask.shape[-1] // 8))
                agn_mask = rearrange(agn_mask, "(b f) c h w -> b c f h w", f=video_length)

                cloth_mask_list = []
                for cloth_mask in cloth_masks:
                    cloth_mask = ref_image_processor.preprocess(cloth_mask, height=cfg.data.train_height, width=cfg.data.train_width)
                    cloth_mask = ref_image_processor.normalize(cloth_mask)
                    cloth_mask = ref_image_processor.binarize(cloth_mask)
                    cloth_mask = cloth_mask[:, 0:1, :, :]
                    cloth_mask_list.append(cloth_mask)
                cloth_mask = torch.cat(cloth_mask_list)
                cloth_mask = torch.nn.functional.interpolate(cloth_mask, size=(cloth_mask.shape[-2] // 8, cloth_mask.shape[-1] // 8))
                cloth_mask = cloth_mask.to(dtype=reference_unet.dtype, device=ref_image_latents.device)

                agnostic_image_latents = agnostic_image_latents.to(dtype=denoising_unet.dtype)
                agn_mask = agn_mask.to(dtype=denoising_unet.dtype, device=denoising_unet.device)

                cond_image_processor = VaeImageProcessor(vae_scale_factor=vae_scale_factor, do_convert_rgb=True, do_normalize=False)

                pose_cond_tensor_list = []
                for pose_image in pose_frames:
                    pose_cond_tensor = cond_image_processor.preprocess(pose_image, height=cfg.data.train_height, width=cfg.data.train_width)
                    pose_cond_tensor_list.append(pose_cond_tensor)
                pose_cond_tensor = torch.cat(pose_cond_tensor_list, dim=0)
                pose_cond_tensor = pose_cond_tensor.to(device=pose_guider.device, dtype=pose_guider.dtype)
                pose_cond_tensor = rearrange(pose_cond_tensor, "(b f) c h w -> b c f h w", f=video_length)

                if train_noise_scheduler.prediction_type == "epsilon":
                    target = noise.to(accelerator.device)
                elif train_noise_scheduler.prediction_type == "v_prediction":
                    target = train_noise_scheduler.get_velocity(latents, noise, timesteps).to(accelerator.device)
                else:
                    raise ValueError(f"Unknown prediction type {train_noise_scheduler.prediction_type}")

                
                ref_image_latents = torch.cat([ref_image_latents, cloth_mask], dim=1)
                noisy_latents = train_noise_scheduler.add_noise(latents, noise, timesteps)
                latents_cat = torch.cat([noisy_latents.to(accelerator.device), agnostic_image_latents, agn_mask.to(accelerator.device)], dim=1)

                noisy_latents = latents_cat
                pose_img = pose_cond_tensor
                
                # model_pred = net(noisy_latents, timesteps, ref_image_latents, encoder_hidden_states, pose_img, uncond_fwd=uncond_fwd)
                model_pred = net(noisy_latents, timesteps, ref_image_latents, encoder_hidden_states, pose_img)

                if cfg.snr_gamma == 0:
                    loss = F.mse_loss(model_pred.float(), target.float(), reduction="mean")
                else:
                    snr = compute_snr(train_noise_scheduler, timesteps)
                    if train_noise_scheduler.prediction_type == "v_prediction":
                        snr = snr + 1
                    mse_loss_weights = (torch.stack([snr, cfg.snr_gamma * torch.ones_like(timesteps)], dim=1).min(dim=1)[0] / snr).to(accelerator.device)
                    loss = F.mse_loss(model_pred.float(), target.float(), reduction="none")
                    loss = (loss.mean(dim=list(range(1, len(loss.shape)))) * mse_loss_weights).mean()


                # Gather the losses across all processes for logging (if we use distributed training).
                avg_loss = accelerator.gather(loss.repeat(cfg.data.train_bs)).mean()
                train_loss += avg_loss.item() / cfg.solver.gradient_accumulation_steps
    
    
                # Backpropagate
                accelerator.backward(loss)
                if accelerator.sync_gradients:
                    accelerator.clip_grad_norm_(
                        trainable_params,
                        cfg.solver.max_grad_norm,
                    )
                optimizer.step()
                lr_scheduler.step()
                optimizer.zero_grad()
                
            if accelerator.sync_gradients:
                reference_control_reader.clear()
                reference_control_writer.clear()
                progress_bar.update(1)
                global_step += 1
                accelerator.log({"train_loss": train_loss}, step=global_step)
                train_loss = 0.0
                # if global_step % cfg.checkpointing_steps == 0:
                # if accelerator.is_main_process:
                    # save_path = os.path.join(save_dir, f"checkpoint-{global_step}")
                    # delete_additional_ckpt(save_dir, 1)
                    # accelerator.save_state(save_path)

                if global_step % cfg.val.validation_steps == 0:
                    if accelerator.is_main_process:
                        generator = torch.Generator(device=accelerator.device)
                        generator.manual_seed(cfg.seed)

                        sample_dicts = log_validation(
                            vae=vae,
                            image_enc=image_enc,
                            net=net,
                            scheduler=val_noise_scheduler,
                            accelerator=accelerator,
                            width=cfg.data.train_width,
                            height=cfg.data.train_height,
                            clip_length=cfg.data.n_sample_frames,
                            generator=generator,
                        )

                        for sample_id, sample_dict in enumerate(sample_dicts):
                            sample_name = sample_dict["name"]
                            vid = sample_dict["vid"]
                            with TemporaryDirectory() as temp_dir:
                                out_file = Path(f"{temp_dir}/{global_step:06d}-{sample_name}.gif")
                                save_videos_grid(vid, out_file, n_rows=2, fps=8)   # 비디오 속도 8 (slow mode)
                                mlflow.log_artifact(out_file)
            logs = {
                "step_loss": loss.detach().item(),
                "lr": lr_scheduler.get_last_lr()[0],
                "td": f"{t_data:.2f}s",
            }
            t_data_start = time.time()
            progress_bar.set_postfix(**logs)

            if global_step >= cfg.solver.max_train_steps:
                break
        
        
        if accelerator.is_main_process:
            save_path = os.path.join(save_dir, f"checkpoint-{global_step}")
            delete_additional_ckpt(save_dir, 1)
            accelerator.save_state(save_path)

            unwrap_net = accelerator.unwrap_model(net)
            save_checkpoint(unwrap_net.reference_unet, save_dir, "reference_unet", global_step, total_limit=3)
            save_checkpoint(unwrap_net.denoising_unet, save_dir, "denoising_unet", global_step, total_limit=3)
            save_checkpoint(unwrap_net.denoising_unet, save_dir, "motion_module", global_step, total_limit=3)
            save_checkpoint(unwrap_net.pose_guider, save_dir, "pose_guider", global_step, total_limit=3)
            accelerator.wait_for_everyone()
        accelerator.wait_for_everyone()
            
    accelerator.wait_for_everyone()
    accelerator.end_training()
    
    
def remove_checkpoints(directory):
    for root, dirs, files in os.walk(directory):
        for dir_name in dirs:
            if dir_name == ".ipynb_checkpoints":
                shutil.rmtree(os.path.join(root, dir_name))
                
def save_checkpoint(model, save_dir, prefix, ckpt_num, total_limit=None):
    save_path = osp.join(save_dir, f"{prefix}-{ckpt_num}.pth")

    if total_limit is not None:
        checkpoints = os.listdir(save_dir)
        checkpoints = [d for d in checkpoints if d.startswith(prefix)]
        checkpoints = sorted(
            checkpoints, key=lambda x: int(x.split("-")[1].split(".")[0])
        )

        if len(checkpoints) >= total_limit:
            num_to_remove = len(checkpoints) - total_limit + 1
            removing_checkpoints = checkpoints[0:num_to_remove]
            logger.info(
                f"{len(checkpoints)} checkpoints already exist, removing {len(removing_checkpoints)} checkpoints"
            )
            logger.info(f"removing checkpoints: {', '.join(removing_checkpoints)}")

            for removing_checkpoint in removing_checkpoints:
                removing_checkpoint = os.path.join(save_dir, removing_checkpoint)
                os.remove(removing_checkpoint)

    state_dict = model.state_dict()
    torch.save(state_dict, save_path)
    


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="/home/vton/westchaevi/ViViD/configs/train/train_stage.yaml")
    args = parser.parse_args()

    if args.config[-5:] == ".yaml":
        config = OmegaConf.load(args.config)
    elif args.config[-3:] == ".py":
        config = import_filename(args.config).cfg
    else:
        raise ValueError("Do not support this format config file")
    main(config)