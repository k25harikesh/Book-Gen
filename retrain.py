import re
import torch
import train

# hyperparameters
batch_size = 16
block_size = 32
max_iters = 1000
eval_interval = 200
learning_rate = 1e-4
device = 'cuda' if torch.cuda.is_available() else 'cpu'
checkpoint_interval = 500

model = train.BigramLanguageModel()


def train_model(dataset_text, max_iters=max_iters, eval_interval=eval_interval, checkpoint_interval=checkpoint_interval, model=model):
  # optimizer
  optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)

  # Define your checkpoint path
  checkpoint_path = './checkpoints/retrained_checkpoint.pth'

  # Convert dataset text to tensor
  data = torch.tensor(train.encode(dataset_text), dtype=torch.long)
  n = int(0.9 * len(data))  # first 90% will be train, rest val
  train_data = data[:n]
  val_data = data[n:]

  # Define function to get batch from the dataset
  def get_batch_from_dataset(split):
    data = train_data if split == 'train' else val_data
    ix = torch.randint(len(data) - block_size, (batch_size,))
    x = torch.stack([data[i:i+block_size] for i in ix])
    y = torch.stack([data[i+1:i+block_size+1] for i in ix])
    x, y = x.to(device), y.to(device)
    return x, y

  # Training loop
  for iter in range(max_iters):
    # every once in a while evaluate the loss on train and val sets
    if iter % eval_interval == 0 or iter == max_iters - 1:
      losses = train.estimate_loss()
      print(
          f"step {iter}: train loss {losses['train']:.4f}, val loss {losses['val']:.4f}")

    # sample a batch of data
    xb, yb = get_batch_from_dataset('train')

    # evaluate the loss
    logits, loss = model(xb, yb)
    optimizer.zero_grad(set_to_none=True)
    loss.backward()
    optimizer.step()

    # Save checkpoint every few iterations
    if (iter + 1) % checkpoint_interval == 0 or iter == max_iters - 1:
      torch.save({
          'iteration': iter,
          'model_state_dict': model.state_dict(),
          'optimizer_state_dict': optimizer.state_dict(),
          'loss': loss
      }, checkpoint_path)

  return model
