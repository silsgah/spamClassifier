import torch
from loss_utils import calc_loss_batch, calc_loss_loader, calc_accuracy_loader

def train_classifier_simple(model, train_loader, val_loader, optimizer, device, num_epochs, eval_freq, eval_iter):
    """
    Trains a classifier model with periodic evaluation.
    Args:
        model: PyTorch model to train.
        train_loader: DataLoader for training data.
        val_loader: DataLoader for validation data.
        optimizer: Optimizer for model training.
        device: Device to train on (e.g., 'cuda' or 'cpu').
        num_epochs: Number of epochs to train.
        eval_freq: Evaluation frequency during training (steps).
        eval_iter: Number of batches for evaluation.
    Returns:
        train_losses, val_losses, train_accs, val_accs, examples_seen
    """
    train_losses, val_losses, train_accs, val_accs = [], [], [], []
    examples_seen, global_step = 0, -1

    for epoch in range(num_epochs):
        model.train()
        for input_batch, target_batch in train_loader:
            optimizer.zero_grad()
            loss = calc_loss_batch(input_batch, target_batch, model, device)
            loss.backward()
            optimizer.step()

            examples_seen += input_batch.shape[0]
            global_step += 1

            if global_step % eval_freq == 0:
                train_loss, val_loss = evaluate_model(model, train_loader, val_loader, device, eval_iter)
                train_losses.append(train_loss)
                val_losses.append(val_loss)
                print(f"Ep {epoch+1} (Step {global_step:06d}): "
                      f"Train loss {train_loss:.3f}, Val loss {val_loss:.3f}")

        train_acc = calc_accuracy_loader(train_loader, model, device, num_batches=eval_iter)
        val_acc = calc_accuracy_loader(val_loader, model, device, num_batches=eval_iter)
        train_accs.append(train_acc)
        val_accs.append(val_acc)
        print(f"Training accuracy: {train_acc*100:.2f}% | Validation accuracy: {val_acc*100:.2f}%")

    return train_losses, val_losses, train_accs, val_accs, examples_seen


def evaluate_model(model, train_loader, val_loader, device, eval_iter):
    """
    Evaluates a model on training and validation loaders.
    Args:
        model: PyTorch model.
        train_loader: Training DataLoader.
        val_loader: Validation DataLoader.
        device: Device to run on.
        eval_iter: Number of batches to evaluate.
    Returns:
        train_loss, val_loss
    """
    model.eval()
    with torch.no_grad():
        train_loss = calc_loss_loader(train_loader, model, device, num_batches=eval_iter)
        val_loss = calc_loss_loader(val_loader, model, device, num_batches=eval_iter)
    model.train()
    return train_loss, val_loss
