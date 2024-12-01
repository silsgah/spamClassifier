import torch
import torch
import time
from training import train_classifier_simple
from loss_utils import calc_loss_loader, calc_loss_batch
from torch.utils.data import DataLoader
from data_processing import (
    SpamDataDownloader,
    load_and_check_data,
    create_balanced_dataset,
    random_split,
)
from dataset import SpamDataset
from model_config import initialize_gpt_model
import tiktoken
from previous_chapters import text_to_token_ids, token_ids_to_text, generate_text_simple
from config import CHOOSE_MODEL, BASE_CONFIG
from plot_utils import plot_values
from classification import classify_review

def prepare_datasets_and_dataloaders(
    train_csv="train.csv",
    val_csv="validation.csv",
    test_csv="test.csv",
    tokenizer=None,
    batch_size=8,
    num_workers=0,
):
    """
    Prepares datasets and DataLoaders for train, validation, and test.

    Args:
        train_csv (str): Path to the training CSV file.
        val_csv (str): Path to the validation CSV file.
        test_csv (str): Path to the test CSV file.
        tokenizer: Tokenizer instance.
        batch_size (int): Batch size for the DataLoader.
        num_workers (int): Number of workers for the DataLoader.

    Returns:
        tuple: (train_loader, val_loader, test_loader)
    """
    # Load datasets
    train_dataset = SpamDataset(csv_file=train_csv, tokenizer=tokenizer)
    val_dataset = SpamDataset(
        csv_file=val_csv, max_length=train_dataset.max_length, tokenizer=tokenizer
    )
    test_dataset = SpamDataset(
        csv_file=test_csv, max_length=train_dataset.max_length, tokenizer=tokenizer
    )

    # Create DataLoaders
    train_loader = DataLoader(
        dataset=train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        drop_last=True,
    )

    val_loader = DataLoader(
        dataset=val_dataset,
        batch_size=batch_size,
        num_workers=num_workers,
        drop_last=False,
    )

    test_loader = DataLoader(
        dataset=test_dataset,
        batch_size=batch_size,
        num_workers=num_workers,
        drop_last=False,
    )

    return train_loader, val_loader, test_loader

def test_generate_text_examples(model, tokenizer, base_config):
    """Test the GPT model with predefined input texts."""
    # Example 1
    text_1 = "Every effort moves you"
    token_ids_1 = generate_text_simple(
        model=model,
        idx=text_to_token_ids(text_1, tokenizer),
        max_new_tokens=15,
        context_size=base_config["context_length"]
    )
    print(f"Input 1: {text_1}")
    print(f"Generated Text 1: {token_ids_to_text(token_ids_1, tokenizer)}\n")

    # Example 2
    text_2 = (
        "Is the following text 'spam'? Answer with 'yes' or 'no':"
        " 'You are a winner you have been specially"
        " selected to receive $1000 cash or a $2000 award.'"
    )
    token_ids_2 = generate_text_simple(
        model=model,
        idx=text_to_token_ids(text_2, tokenizer),
        max_new_tokens=23,
        context_size=base_config["context_length"]
    )
    print(f"Input 2: {text_2}")
    print(f"Generated Text 2: {token_ids_to_text(token_ids_2, tokenizer)}\n")

def calc_accuracy_loader(data_loader, model, device, num_batches=None):
    """
    Calculates the accuracy of the model on a given data loader.

    Args:
        data_loader: DataLoader object for the dataset.
        model: PyTorch model to evaluate.
        device: Device to perform computation ('cpu', 'cuda', 'mps').
        num_batches: Number of batches to process. If None, processes all batches.

    Returns:
        float: Accuracy of the model on the provided dataset.
    """
    model.eval()
    correct_predictions, num_examples = 0, 0

    if num_batches is None:
        num_batches = len(data_loader)
    else:
        num_batches = min(num_batches, len(data_loader))

    for i, (input_batch, target_batch) in enumerate(data_loader):
        if i < num_batches:
            input_batch, target_batch = input_batch.to(device), target_batch.to(device)

            with torch.no_grad():
                logits = model(input_batch)[:, -1, :]  # Logits of the last output token
            predicted_labels = torch.argmax(logits, dim=-1)

            num_examples += predicted_labels.shape[0]
            correct_predictions += (predicted_labels == target_batch).sum().item()
        else:
            break

    return correct_predictions / num_examples

def calc_loss_batch(input_batch, target_batch, model, device):
    """
    Calculate the loss for a single batch.

    Args:
        input_batch: Tensor containing input data.
        target_batch: Tensor containing target labels.
        model: The PyTorch model.
        device: Device to perform computation ('cpu', 'cuda', 'mps').

    Returns:
        torch.Tensor: Loss for the batch.
    """
    input_batch, target_batch = input_batch.to(device), target_batch.to(device)
    logits = model(input_batch)[:, -1, :]  # Logits of last output token
    loss = torch.nn.functional.cross_entropy(logits, target_batch)
    return loss


def calc_loss_loader(data_loader, model, device, num_batches=None):
    """
    Calculate the average loss over a data loader.

    Args:
        data_loader: DataLoader object for the dataset.
        model: The PyTorch model.
        device: Device to perform computation ('cpu', 'cuda', 'mps').
        num_batches: Number of batches to process. If None, processes all batches.

    Returns:
        float: Average loss over the provided dataset.
    """
    total_loss = 0.0
    if len(data_loader) == 0:
        return float("nan")  # Avoid divide-by-zero error

    if num_batches is None:
        num_batches = len(data_loader)
    else:
        # Limit num_batches to the number of batches available in the data loader
        num_batches = min(num_batches, len(data_loader))

    for i, (input_batch, target_batch) in enumerate(data_loader):
        if i < num_batches:
            loss = calc_loss_batch(input_batch, target_batch, model, device)
            total_loss += loss.item()
        else:
            break

    return total_loss / num_batches

# Main execution
if __name__ == "__main__":
    # Step 1: Download and extract data
    downloader = SpamDataDownloader(
        url="https://archive.ics.uci.edu/static/public/228/sms+spam+collection.zip",
        zip_path="sms_spam_collection.zip",
        extracted_path="sms_spam_collection",
        data_file_name="SMSSpamCollection.tsv",
    )
    downloader.download_and_extract()

    # Step 2: Load and process data
    data_file_path = downloader.data_file_path
    df = load_and_check_data(data_file_path)
    balanced_df = create_balanced_dataset(df)
    train_df, val_df, test_df = random_split(balanced_df, 0.7, 0.1)
    train_df.to_csv("train.csv", index=None)
    val_df.to_csv("validation.csv", index=None)
    test_df.to_csv("test.csv", index=None)

    print("Data preparation completed!")

    # Step 3: Prepare datasets and DataLoaders
    tokenizer = tiktoken.get_encoding("gpt2")
    train_loader, val_loader, test_loader = prepare_datasets_and_dataloaders(
        train_csv="train.csv",
        val_csv="validation.csv",
        test_csv="test.csv",
        tokenizer=tokenizer,
        batch_size=8,
        num_workers=0,
    )

    # Check DataLoader details
    print("Train loader:")
    for input_batch, target_batch in train_loader:
        print("Input batch dimensions:", input_batch.shape)
        print("Label batch dimensions:", target_batch.shape)
        break  # Only check the first batch for dimensions

    print(f"{len(train_loader)} training batches")
    print(f"{len(val_loader)} validation batches")
    print(f"{len(test_loader)} test batches")
    
    # Step 7: Create the datasets
    train_dataset = SpamDataset(
        csv_file="train.csv",
        max_length=None,  # Let the dataset determine the max length dynamically
        tokenizer=tokenizer
    )

    # Step 8: Initialize the GPT model
    CHOOSE_MODEL = "gpt2-small (124M)"
    model = initialize_gpt_model(
        model_name=CHOOSE_MODEL,
        train_dataset_max_length=train_dataset.max_length,
        models_dir="gpt2"
    )
    BASE_CONFIG.update(BASE_CONFIG["model_specs"][CHOOSE_MODEL])
    # Run the examples
    test_generate_text_examples(model, tokenizer, BASE_CONFIG) 
    # print(model)
    for param in model.parameters():
        param.requires_grad = False
    
    torch.manual_seed(123)

    num_classes = 2
    model.out_head = torch.nn.Linear(in_features=BASE_CONFIG["emb_dim"], out_features=num_classes)
    for param in model.trf_blocks[-1].parameters():

        param.requires_grad = True

    for param in model.final_norm.parameters():
        param.requires_grad = True
    
    inputs = tokenizer.encode("Do you have time")
    inputs = torch.tensor(inputs).unsqueeze(0)
    print("Inputs:", inputs)
    print("Inputs dimensions:", inputs.shape) # shape: (batch_size, num_tokens)
    with torch.no_grad():
        outputs = model(inputs)

    print("Outputs:\n", outputs)
    print("Outputs dimensions:", outputs.shape) # shape: (batch_size, num_tokens, num_classes)
    print("Last output token:", outputs[:, -1, :])
    probas = torch.softmax(outputs[:, -1, :], dim=-1)
    label = torch.argmax(probas)
    print("Class label:", label.item())
    logits = outputs[:, -1, :]
    label = torch.argmax(logits)
    print("Class label:", label.item())

    # Set up the device
    if torch.cuda.is_available():
        device = torch.device("cuda")
    elif torch.backends.mps.is_available():
        device = torch.device("mps")
    else:
        device = torch.device("cpu")
    print(f"Running on {device} device.")

    # Move the model to the device
    model.to(device)

    # Ensure reproducibility
    torch.manual_seed(123)

    # Calculate accuracies
    train_accuracy = calc_accuracy_loader(train_loader, model, device, num_batches=10)
    val_accuracy = calc_accuracy_loader(val_loader, model, device, num_batches=10)
    test_accuracy = calc_accuracy_loader(test_loader, model, device, num_batches=10)

    # Print accuracies
    print(f"Training accuracy: {train_accuracy*100:.2f}%")
    print(f"Validation accuracy: {val_accuracy*100:.2f}%")
    print(f"Test accuracy: {test_accuracy*100:.2f}%")

    # Calculate and print losses
    with torch.no_grad():  # Disable gradient tracking for efficiency
        train_loss = calc_loss_loader(train_loader, model, device, num_batches=5)
        val_loss = calc_loss_loader(val_loader, model, device, num_batches=5)
        test_loss = calc_loss_loader(test_loader, model, device, num_batches=5)

    print(f"Training loss: {train_loss:.3f}")
    print(f"Validation loss: {val_loss:.3f}")
    print(f"Test loss: {test_loss:.3f}")



    # Training setup
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    torch.manual_seed(123)

    optimizer = torch.optim.AdamW(model.parameters(), lr=5e-5, weight_decay=0.1)
    num_epochs = 5

    # Start training
    print("Starting training...")
    start_time = time.time()

    train_losses, val_losses, train_accs, val_accs, examples_seen = train_classifier_simple(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        optimizer=optimizer,
        device=device,
        num_epochs=num_epochs,
        eval_freq=50,
        eval_iter=5,
    )

    end_time = time.time()
    execution_time_minutes = (end_time - start_time) / 60
    print(f"Training completed in {execution_time_minutes:.2f} minutes.")
    print(f"Final Training Accuracy: {train_accs[-1]*100:.2f}%")
    print(f"Final Validation Accuracy: {val_accs[-1]*100:.2f}%")
    print(f"Final Training Loss: {train_losses[-1]:.3f}")
    print(f"Final Validation Loss: {val_losses[-1]:.3f}")

    # Generate tensors for plotting
    epochs_tensor = torch.linspace(0, num_epochs, len(train_losses))
    examples_seen_tensor = torch.linspace(0, examples_seen, len(train_losses))

    # # Plot loss values
    # plot_values(epochs_tensor, examples_seen_tensor, train_losses, val_losses, label="loss")

    # # Plot accuracy values
    # plot_values(epochs_tensor, examples_seen_tensor, train_accs, val_accs, label="accuracy")
   
    train_accuracy = calc_accuracy_loader(train_loader, model, device)
    val_accuracy = calc_accuracy_loader(val_loader, model, device)
    test_accuracy = calc_accuracy_loader(test_loader, model, device)

    print(f"Training accuracy: {train_accuracy*100:.2f}%")
    print(f"Validation accuracy: {val_accuracy*100:.2f}%")
    print(f"Test accuracy: {test_accuracy*100:.2f}%")


    # Example text inputs for classification
    text_1 = (
        "You are a winner you have been specially"
        " selected to receive $1000 cash or a $2000 award."
    )

    classification_result_1 = classify_review(
        text_1, model, tokenizer, device, max_length=train_dataset.max_length
    )
    print(f"Text 1 Classification: {classification_result_1}")

    text_2 = (
        "Hey, just wanted to check if we're still on"
        " for dinner tonight? Let me know!"
    )

    classification_result_2 = classify_review(
        text_2, model, tokenizer, device, max_length=train_dataset.max_length
    )
    print(f"Text 2 Classification: {classification_result_2}")

    torch.save(model.state_dict(), "review_classifier.pth")

    model_state_dict = torch.load("review_classifier.pth", map_location=device, weights_only=True)
    model.load_state_dict(model_state_dict)
    # pip install chainlit
    # chainlit run app.py