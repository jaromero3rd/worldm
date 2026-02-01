from spr.utils import Config, DatasetConfig, PolicyDecoderConfig, PolicyEncoderConfig, TrainerConfig, VAEConfig

# Define default environments
ENV_CONFIGS = {
    "mo-halfcheetah-v5": Config(
        trainer=TrainerConfig(
            vae_epochs=100,
            coef_kl_end=0.05,
            rnc_temperature=0.5,
            coef_ortho=5.0,
            coef_contrastive=1.0,
            contrastive_loss_type="rnc",
        ),
        dataset=DatasetConfig(
            max_checkpoints=2400,
        ),
        vae=VAEConfig(
            context_length=32,
            encoder=PolicyEncoderConfig(
                encoder_type="transformer",
                n_embd=32,
                obj_n_embd=4,
                # obj_hidden_dims=[16, 16],
            ),
            decoder=PolicyDecoderConfig(),
        ),
    ),
    "mo-walker2d-v5": Config(
        trainer=TrainerConfig(
            vae_epochs=200,
        ),
        dataset=DatasetConfig(
            max_checkpoints=2400,
        ),
        vae=VAEConfig(
            context_length=32,
            encoder=PolicyEncoderConfig(),
            decoder=PolicyDecoderConfig(),
        ),
    ),
    "mo-humanoid-v5": Config(
        trainer=TrainerConfig(vae_epochs=200),
        dataset=DatasetConfig(
            max_checkpoints=2100,
        ),
        vae=VAEConfig(
            context_length=32,
            encoder=PolicyEncoderConfig(),
            decoder=PolicyDecoderConfig(),
        ),
    ),
    "mo-hopper-2obj-v5": Config(
        trainer=TrainerConfig(vae_epochs=200),
        dataset=DatasetConfig(),
        vae=VAEConfig(
            context_length=32,
            encoder=PolicyEncoderConfig(),
            decoder=PolicyDecoderConfig(),
        ),
    ),
    "mo-ant-2obj-v5": Config(
        trainer=TrainerConfig(vae_epochs=200),
        dataset=DatasetConfig(),
        vae=VAEConfig(
            context_length=32,
            encoder=PolicyEncoderConfig(),
            decoder=PolicyDecoderConfig(),
        ),
    ),
    # ---------------------------- 3 objectives ------------------------
    "mo-hopper-v5": Config(
        trainer=TrainerConfig(vae_epochs=200),
        dataset=DatasetConfig(),
        vae=VAEConfig(
            context_length=32,
            encoder=PolicyEncoderConfig(),
            decoder=PolicyDecoderConfig(),
        ),
    ),
    "mo-ant-v5": Config(
        trainer=TrainerConfig(vae_epochs=200),
        dataset=DatasetConfig(),
        vae=VAEConfig(
            context_length=32,
            encoder=PolicyEncoderConfig(),
            decoder=PolicyDecoderConfig(),
        ),
    ),
}
