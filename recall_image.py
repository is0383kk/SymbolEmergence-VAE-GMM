"""
Image recall module for Symbol Emergence VAE+GMM model.
This module handles post-training image reconstruction and visualization.
"""

import argparse
from pathlib import Path
from typing import List, Tuple

import torch
from PIL import Image
from torch.utils.data.dataset import Subset
from torchvision import datasets, transforms

import cnn_vae_module_mnist
from tool import visualize_gmm


class Config:
    """Configuration class for recall image parameters."""

    def __init__(self):
        self.batch_size = 10
        self.vae_iter = 100
        self.mh_iter = 100
        self.category = 28
        self.no_cuda = False
        self.seed = 1
        self.angle_a = 0
        self.angle_b = 45
        self.data_fraction = 1 / 6
        self.file_name = "debug"
        self.model_dir = "./model"

    def setup_device(self) -> torch.device:
        """Setup computing device (CPU/CUDA)."""
        self.cuda = not self.no_cuda and torch.cuda.is_available()
        torch.manual_seed(self.seed)
        return torch.device("cuda" if self.cuda else "cpu")


class DirectoryManager:
    """Manages directory creation and path handling."""

    def __init__(self, config: Config):
        self.config = config
        self.base_dir = Path(config.model_dir) / config.file_name
        self._setup_directories()

    def _setup_directories(self) -> None:
        """Create all necessary directories."""
        directories = [
            self.config.model_dir,
            self.base_dir,
            self.base_dir / "graphA",
            self.base_dir / "graphB",
            self.base_dir / "pth",
            self.base_dir / "npy",
            self.base_dir / "reconA" / "graph_dist",
            self.base_dir / "reconB" / "graph_dist",
            self.base_dir / "log",
            self.base_dir / "result",
        ]

        for directory in directories:
            Path(directory).mkdir(parents=True, exist_ok=True)

    @property
    def dir_name(self) -> str:
        """Get base directory name as string for compatibility."""
        return str(self.base_dir)


class DatasetManager:
    """Manages dataset creation and loading."""

    def __init__(self, config: Config):
        self.config = config
        self.train_loader1 = None
        self.train_loader2 = None
        self.all_loader1 = None
        self.all_loader2 = None
        self._setup_datasets()

    def _create_transforms(self) -> Tuple[transforms.Compose, transforms.Compose]:
        """Create transformation pipelines for both agents."""
        trans_ang1 = transforms.Compose(
            [
                transforms.RandomRotation(
                    degrees=(-self.config.angle_a, -self.config.angle_a)
                ),
                transforms.ToTensor(),
            ]
        )

        trans_ang2 = transforms.Compose(
            [
                transforms.RandomRotation(
                    degrees=(self.config.angle_b, self.config.angle_b)
                ),
                transforms.ToTensor(),
            ]
        )

        return trans_ang1, trans_ang2

    def _setup_datasets(self) -> None:
        """Setup datasets and data loaders for both agents."""
        trans_ang1, trans_ang2 = self._create_transforms()

        # Create datasets
        trainval_dataset1 = datasets.MNIST(
            "./../data", train=True, transform=trans_ang1, download=False
        )
        trainval_dataset2 = datasets.MNIST(
            "./../data", train=True, transform=trans_ang2, download=False
        )

        # Calculate subset sizes
        n_samples = len(trainval_dataset1)
        D = int(n_samples * self.config.data_fraction)

        # Create subsets
        subset1_indices = list(range(0, D))
        train_dataset1 = Subset(trainval_dataset1, subset1_indices)
        train_dataset2 = Subset(trainval_dataset2, subset1_indices)

        # Create data loaders
        self.train_loader1 = torch.utils.data.DataLoader(
            train_dataset1, batch_size=self.config.batch_size, shuffle=False
        )
        self.train_loader2 = torch.utils.data.DataLoader(
            train_dataset2, batch_size=self.config.batch_size, shuffle=False
        )
        self.all_loader1 = torch.utils.data.DataLoader(
            train_dataset1, batch_size=D, shuffle=False
        )
        self.all_loader2 = torch.utils.data.DataLoader(
            train_dataset2, batch_size=D, shuffle=False
        )


class ImageRecaller:
    """Main class for handling image recall functionality."""

    def __init__(self, config: Config):
        self.config = config
        self.device = config.setup_device()
        self.dir_manager = DirectoryManager(config)
        self.dataset_manager = DatasetManager(config)

    def _load_and_resize_images(
        self, dir_name: str, agent: str, num_images: int = 10
    ) -> List[Image.Image]:
        """Load and resize images for concatenation."""
        images = []

        for i in range(num_images):
            image_path = Path(dir_name) / f"recon{agent}" / f"manual_{i}.png"

            if not image_path.exists():
                raise FileNotFoundError(f"Image not found: {image_path}")

            try:
                image = Image.open(image_path)
                images.append(image)
            except Exception as e:
                raise IOError(f"Error loading image {image_path}: {e}")

        return images

    def _resize_images_to_min_height(
        self, images: List[Image.Image]
    ) -> List[Image.Image]:
        """Resize all images to minimum height while maintaining aspect ratio."""
        if not images:
            return images

        min_height = min(img.height for img in images)
        resized_images = []

        for img in images:
            new_width = int(img.width * min_height / img.height)
            resized_img = img.resize((new_width, min_height), Image.BICUBIC)
            resized_images.append(resized_img)

        return resized_images

    def concatenate_images_horizontally(self, dir_name: str, agent: str) -> None:
        """
        Concatenate multiple reconstruction images horizontally.

        Args:
            dir_name: Directory containing the images
            agent: Agent identifier ("A" or "B")
        """
        try:
            images = self._load_and_resize_images(dir_name, agent)
            resized_images = self._resize_images_to_min_height(images)

            if not resized_images:
                raise ValueError("No images to concatenate")

            # Calculate total width and create destination image
            total_width = sum(img.width for img in resized_images)
            height = resized_images[0].height
            dst = Image.new("RGB", (total_width, height))

            # Paste images horizontally
            pos_x = 0
            for img in resized_images:
                dst.paste(img, (pos_x, 0))
                pos_x += img.width

            # Save concatenated image
            output_path = Path(dir_name) / f"recon{agent}" / "concat.png"
            output_path.parent.mkdir(parents=True, exist_ok=True)
            dst.save(output_path)

        except Exception as e:
            print(f"Error concatenating images for agent {agent}: {e}")
            raise

    def decode_from_gmm(
        self,
        load_iteration: int = 0,
        sigma: float = 0,
        K: int = 10,
        sample_num: int = 1,
        manual: bool = True,
    ) -> None:
        """
        Decode images from GMM components for both agents.

        Args:
            load_iteration: Model iteration to load
            sigma: Sigma parameter for GMM
            K: Number of GMM components
            sample_num: Number of samples to generate
            manual: Whether to use manual mode
        """
        dir_name = self.dir_manager.dir_name

        try:
            for i in range(K):
                # Process Agent A
                sample_d_a = visualize_gmm(
                    iteration=load_iteration,
                    sigma=sigma,
                    K=K,
                    decode_k=i,
                    sample_num=sample_num,
                    manual=manual,
                    model_dir=dir_name,
                    agent="A",
                )

                cnn_vae_module_mnist.decode(
                    iteration=load_iteration,
                    decode_k=i,
                    sample_num=sample_num,
                    sample_d=sample_d_a,
                    manual=manual,
                    model_dir=dir_name,
                    agent="A",
                )

                # Process Agent B
                sample_d_b = visualize_gmm(
                    iteration=load_iteration,
                    sigma=sigma,
                    K=K,
                    decode_k=i,
                    sample_num=sample_num,
                    manual=manual,
                    model_dir=dir_name,
                    agent="B",
                )

                cnn_vae_module_mnist.decode(
                    iteration=load_iteration,
                    decode_k=i,
                    sample_num=sample_num,
                    sample_d=sample_d_b,
                    manual=manual,
                    model_dir=dir_name,
                    agent="B",
                )

        except Exception as e:
            print(f"Error in GMM decoding: {e}")
            raise

    def run_recall(self, load_iteration: int = 0) -> None:
        """
        Run the complete image recall process.

        Args:
            load_iteration: Model iteration to load for recall
        """
        try:
            # Decode images from GMM
            self.decode_from_gmm(
                load_iteration=load_iteration,
                sigma=0,
                K=10,
                sample_num=1,
                manual=True,
            )

            # Concatenate images for both agents
            dir_name = self.dir_manager.dir_name
            self.concatenate_images_horizontally(dir_name, "A")
            self.concatenate_images_horizontally(dir_name, "B")

            print(f"Image recall completed successfully. Results saved in {dir_name}")

        except Exception as e:
            print(f"Error during image recall: {e}")
            raise


def parse_arguments() -> Config:
    """Parse command line arguments and return configuration."""
    parser = argparse.ArgumentParser(
        description="Symbol emergence based on VAE+GMM Example - Image Recall"
    )

    parser.add_argument(
        "--batch-size",
        type=int,
        default=10,
        metavar="B",
        help="input batch size for training",
    )
    parser.add_argument(
        "--vae-iter", type=int, default=100, metavar="V", help="number of VAE iteration"
    )
    parser.add_argument(
        "--mh-iter",
        type=int,
        default=100,
        metavar="M",
        help="number of M-H mgmm iteration",
    )
    parser.add_argument(
        "--category",
        type=int,
        default=28,
        metavar="K",
        help="number of category for GMM module",
    )
    parser.add_argument(
        "--no-cuda", action="store_true", default=False, help="enables CUDA training"
    )
    parser.add_argument("--seed", type=int, default=1, metavar="S", help="random seed")

    args = parser.parse_args()

    # Create config object
    config = Config()
    config.batch_size = args.batch_size
    config.vae_iter = args.vae_iter
    config.mh_iter = args.mh_iter
    config.category = args.category
    config.no_cuda = args.no_cuda
    config.seed = args.seed

    return config


def main() -> None:
    """Main function to run image recall."""
    try:
        config = parse_arguments()
        recaller = ImageRecaller(config)
        recaller.run_recall(load_iteration=0)

    except Exception as e:
        print(f"Error in main execution: {e}")
        return 1

    return 0


if __name__ == "__main__":
    exit(main())
