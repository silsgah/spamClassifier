import torch

def classify_review(text, model, tokenizer, device, max_length=None, pad_token_id=50256):
    """
    Classifies a review as 'spam' or 'not spam'.

    Args:
        text (str): The input text to classify.
        model (torch.nn.Module): The trained model.
        tokenizer: The tokenizer for encoding text.
        device: The device ('cuda' or 'cpu') to run inference on.
        max_length (int): Maximum context length for the model.
        pad_token_id (int): Token ID used for padding shorter sequences.

    Returns:
        str: Classification result ('spam' or 'not spam').
    """
    model.eval()

    # Prepare inputs to the model
    input_ids = tokenizer.encode(text)
    supported_context_length = model.pos_emb.weight.shape[0]

    # Truncate sequences if they are too long
    input_ids = input_ids[:min(max_length, supported_context_length)]

    # Pad sequences to the maximum length
    input_ids += [pad_token_id] * (max_length - len(input_ids))
    input_tensor = torch.tensor(input_ids, device=device).unsqueeze(0)  # Add batch dimension

    # Model inference
    with torch.no_grad():
        logits = model(input_tensor)[:, -1, :]  # Logits of the last output token
    predicted_label = torch.argmax(logits, dim=-1).item()

    # Return the classified result
    return "spam" if predicted_label == 1 else "not spam"
