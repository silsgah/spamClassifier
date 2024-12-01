from gpt_download import download_and_load_gpt2
from previous_chapters import GPTModel, load_weights_into_gpt


class ModelConfig:
    def __init__(self, model_name, train_dataset_max_length):
        self.model_name = model_name
        self.base_config = {
            "vocab_size": 50257,     # Vocabulary size
            "context_length": 1024,  # Context length
            "drop_rate": 0.0,        # Dropout rate
            "qkv_bias": True,        # Query-key-value bias
        }

        # Model-specific configurations
        self.model_configs = {
            "gpt2-small (124M)": {"emb_dim": 768, "n_layers": 12, "n_heads": 12},
            "gpt2-medium (355M)": {"emb_dim": 1024, "n_layers": 24, "n_heads": 16},
            "gpt2-large (774M)": {"emb_dim": 1280, "n_layers": 36, "n_heads": 20},
            "gpt2-xl (1558M)": {"emb_dim": 1600, "n_layers": 48, "n_heads": 25},
        }

        self._validate_model_name()
        self.base_config.update(self.model_configs[self.model_name])

        self._validate_context_length(train_dataset_max_length)

    def _validate_model_name(self):
        if self.model_name not in self.model_configs:
            raise ValueError(
                f"Invalid model name '{self.model_name}'. Choose from: "
                f"{list(self.model_configs.keys())}"
            )

    def _validate_context_length(self, train_dataset_max_length):
        if train_dataset_max_length > self.base_config["context_length"]:
            raise ValueError(
                f"Dataset length {train_dataset_max_length} exceeds model's "
                f"context length {self.base_config['context_length']}. "
                f"Reinitialize data sets with max_length="
                f"{self.base_config['context_length']}"
            )

    def get_config(self):
        return self.base_config


def initialize_gpt_model(model_name, train_dataset_max_length, models_dir="gpt2"):
    """
    Initializes the GPT model with pre-trained weights.

    Args:
        model_name (str): Name of the GPT model.
        train_dataset_max_length (int): Max sequence length in training data.
        models_dir (str): Directory to store downloaded models.

    Returns:
        torch.nn.Module: The GPT model in evaluation mode.
    """
    # Step 1: Load configuration
    config = ModelConfig(model_name, train_dataset_max_length).get_config()

    # Step 2: Download weights and settings
    model_size = model_name.split(" ")[-1].lstrip("(").rstrip(")")
    settings, params = download_and_load_gpt2(model_size=model_size, models_dir=models_dir)

    # Step 3: Initialize GPT model
    model = GPTModel(config)
    load_weights_into_gpt(model, params)

    # Set to evaluation mode
    model.eval()

    return model


# Main execution
if __name__ == "__main__":
    CHOOSE_MODEL = "gpt2-small (124M)"
    INPUT_PROMPT = "Every effort moves"

    # Initialize the GPT model
    model = initialize_gpt_model(
        model_name=CHOOSE_MODEL,
        train_dataset_max_length=train_dataset.max_length,
        models_dir="gpt2"
    )

    print(f"{CHOOSE_MODEL} initialized and ready for evaluation.")
