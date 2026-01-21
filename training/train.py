import os

import jax
import jax.numpy as jnp
import optax
import seaborn as sns
from dotenv import load_dotenv
from flax import nnx
from matplotlib import pyplot as plt

import wandb
from models import ConsistentAutoencoder, DynamicAutoencoder, VanillaAutoencoder
from systems.lorenz import lorenz
from utils import prepare_data

load_dotenv()

WANDB_PROJECT = os.getenv("WANDB_PROJECT")
WANDB_ENTITY = os.getenv("WANDB_ENTITY")


def log_rollout(test_rollout, pred_rollout, window_size):
    """
    test_rollout: (window, D)
    pred_rollout: (window, D)
    """
    t = list(range(window_size))
    labels = ["x", "y", "z"]

    # Per-component plots
    for i, label in enumerate(labels):
        wandb.log(
            {
                f"test/{label}": wandb.plot.line_series(
                    xs=t,
                    ys=[
                        test_rollout[:, i].tolist(),
                        pred_rollout[:, i].tolist(),
                    ],
                    keys=["ground_truth", "prediction"],
                    title=f"{label} component",
                    xname="t",
                )
            }
        )

    # MSE over time
    mse_per_step = jnp.mean((test_rollout - pred_rollout) ** 2, axis=1)  # (window,)
    wandb.log(
        {
            "test/mse_over_time": wandb.plot.line_series(
                xs=t,
                ys=[mse_per_step.tolist()],
                keys=["mse"],
                title="MSE over time",
                xname="t",
            )
        }
    )


@nnx.jit
def train_step(model, optimizer, batch):
    def loss_fn(model):
        return model.forward_and_loss_function(batch)

    (loss, aux), grads = nnx.value_and_grad(loss_fn, has_aux=True)(model)
    optimizer.update(model, grads)
    return loss, aux


@nnx.jit
def validate_step(model, batch):
    loss, aux = model.forward_and_loss_function(batch)
    return loss, aux


def main():
    # region dataset
    system = lorenz
    num_steps = 1_000
    num_trajectories = 100
    window_size = 100

    train_data = prepare_data(
        system=system,
        num_steps=num_steps,
        num_trajectories=num_trajectories,
        window_size=window_size,
        rngs=jax.random.PRNGKey(42),
    )

    val_data = prepare_data(
        system=system,
        num_steps=num_steps,
        num_trajectories=5,
        window_size=window_size,
        rngs=jax.random.PRNGKey(2024),
    )

    test_rollout, _ = prepare_data(
        system=system,
        num_steps=num_steps,
        num_trajectories=1,
        window_size=window_size,
        shuffle=False,
        output_rollouts=True,
        rngs=jax.random.PRNGKey(24),
    )
    # endregion

    # region model
    # model = ConsistentAutoencoder(
    #     input_dim=3,
    #     hidden_dim=16,
    #     koopman_dim=6,
    #     dt=0.01,
    #     rngs=nnx.Rngs(10),
    # )

    # model = DynamicAutoencoder(
    #     input_dim=3,
    #     hidden_dim=16,
    #     koopman_dim=6,
    #     dt=0.01,
    #     rngs=nnx.Rngs(10),
    # )

    model = VanillaAutoencoder(
        input_dim=3,
        hidden_dim=16,
        koopman_dim=6,
        dt=0.01,
        rngs=nnx.Rngs(10),
    )
    # endregion

    # region wandb config
    wandb_run = wandb.init(
        entity=WANDB_ENTITY,
        project=WANDB_PROJECT,
        config={
            "model": model.__class__.__name__,
        },
    )
    # endregion

    # region training
    epochs = 100
    print_epoch = epochs // 10
    learning_rate = 3e-4
    batch_size = 128
    optimizer = nnx.Optimizer(model, optax.adamw(learning_rate), wrt=nnx.Param)

    for epoch in range(epochs):
        perm = jax.random.permutation(jax.random.PRNGKey(epoch), train_data.shape[0])
        train_data = train_data[perm]

        metrics_list = []
        for i in range(0, train_data.shape[0], batch_size):
            batch = train_data[i : i + batch_size]
            loss, aux = train_step(model, optimizer, batch)

            # logging
            if i == 0:
                for _, v in aux.items():
                    metric = nnx.metrics.Average()
                    metric.update(values=v)
                    metrics_list.append(metric)
            else:
                for i, (_, v) in enumerate(aux.items()):
                    metrics_list[i].update(values=v)

        if epoch % print_epoch == 0:
            print(f"Epoch {epoch}, Loss: {loss:.6f}")

            # validation
            val_perm = jax.random.permutation(jax.random.PRNGKey(epoch + 1000), val_data.shape[0])
            val_batch = val_data[val_perm][:batch_size]
            val_loss, val_aux = validate_step(model, val_batch)

            # wandb logging
            wandb_run.log({"train/loss": loss}, step=epoch)
            for i, (k, _) in enumerate(aux.items()):
                wandb_run.log({f"train/{k}": metrics_list[i].compute()}, step=epoch)
            wandb_run.log({"val/loss": val_loss}, step=epoch)
            for k, v in val_aux.items():
                wandb_run.log({f"val/{k}": v}, step=epoch)
        # endregion

    # region test rollout
    test_t_start = 500
    encoded = model.encoder(test_rollout[:, test_t_start, :])
    pred_rollout = model.koopman_operator(encoded, T=window_size)
    decoded = jax.vmap(model.decoder)(pred_rollout[0])
    log_rollout(
        test_rollout.squeeze()[test_t_start : test_t_start + window_size],
        decoded.squeeze(),
        window_size=window_size,
    )
    # endregion

    wandb_run.finish()


if __name__ == "__main__":
    main()
