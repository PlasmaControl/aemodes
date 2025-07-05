import matplotlib.pyplot as plt
from multiprocessing import Pool, cpu_count
import shutil
from glob import glob
import wandb
import hydra
from pathlib import Path

import imageio.v3 as iio

import torch
from torch.utils.data import DataLoader
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

from ..utils.dataset import load_dataset


def generate_frame(index, input_image, ground_truth, prediction, cache_dir):
    """Generates and saves a single frame for the GIF."""
    plt.figure(figsize=(10, 8))
    plt.subplot(4, 1, 1)
    plt.imshow(input_image[0].T, cmap='gist_heat', origin='lower', aspect='auto')
    plt.axis("off")
    plt.title("Input Image")

    plt.subplot(4, 1, 2)
    plt.imshow(ground_truth.T, origin='lower', aspect='auto', interpolation='none')
    plt.axis("off")
    plt.title("Ground Truth")

    plt.subplot(4, 1, 3)
    plt.imshow(prediction.T, origin='lower', aspect='auto', interpolation='none')
    plt.axis("off")
    plt.title("Model Prediction")

    plt.subplot(4, 1, 4)
    plt.imshow(prediction.T > 0.5, origin='lower', aspect='auto', interpolation='none')
    plt.axis("off")
    plt.title("Thresholded Prediction")

    plt.tight_layout()
    output_path = cache_dir / f"{index}.png"
    plt.savefig(output_path)
    plt.close()

    return str(output_path)


def create_gif(image_files, output_file, duration):
    """Creates a GIF from a list of image files."""
    images = [iio.imread(img) for img in image_files]
    iio.imwrite(output_file, images, duration=duration, loop=0)
    print(f"GIF saved at: {output_file}")


def generate_inference_gif(cfg, model, output_dir):
    """Generates an inference GIF and logs it to wandb."""
    model.eval()
    model.to(device)  # Ensure model is on the correct device
    
    _, valid_dataset = load_dataset(cfg.train.data_file)
    
    loader = DataLoader(
        valid_dataset, 
        batch_size=cfg.train.batch_size, 
        shuffle=False, 
        num_workers=cfg.train.num_workers
    )
    
    cache_dir = output_dir / "cache"
    cache_dir.mkdir(parents=True, exist_ok=True)
    
    frame_paths = []
    
    with torch.no_grad():
        with Pool(cpu_count()) as pool:
            tasks = []
            for batch_idx, batch in enumerate(loader):
                print(f"Processing video batch {batch_idx + 1}/{len(loader)}", flush=True)
                inputs = batch['X'].to(device)  # Ensure inputs are moved to the correct device
                labels = batch['y']
                
                predictions = model(inputs)
                predictions = torch.sigmoid(predictions).cpu().detach().numpy()
                
                for i in range(len(inputs)):
                    sample_idx = batch_idx * cfg.train.batch_size + i
                    task = (
                        sample_idx,
                        inputs[i].cpu().numpy(),
                        labels[i].cpu().numpy(),
                        predictions[i],
                        cache_dir
                    )
                    tasks.append(task)
            
            frame_paths = pool.starmap(generate_frame, tasks)

    # Sort frame paths numerically to ensure correct order in GIF
    frame_paths.sort(key=lambda f: int(Path(f).stem))

    gif_path = output_dir / "inference_animation.gif"
    create_gif(frame_paths, gif_path, duration=0.8)
    
    if cfg.wandb.mode != 'disabled':
        wandb.log({"inference_animation": wandb.Video(str(gif_path), fps=3, format="gif")})
        
    # Clean up cache directory
    shutil.rmtree(cache_dir)
    
    
def evaluate_model(cfg, output_dir):
    """Evaluate the model and generate inference GIF."""
    model = hydra.utils.instantiate(cfg.model).to(device)  # Ensure model is moved to the correct device
    state_dict = torch.load(output_dir / cfg.train.ckpt_file, map_location=device)
    model.load_state_dict(state_dict)
    
    generate_inference_gif(cfg, model, output_dir)
    
    return