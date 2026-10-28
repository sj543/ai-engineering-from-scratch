import torch
import torch.nn as nn
from torch.utils.tensorboard import SummaryWriter

# 1. Create synthetic data (small dataset easily overfitted)
torch.manual_seed(42)
X_train = torch.randn(20, 10)
y_train = torch.randint(0, 2, (20,))
X_val = torch.randn(20, 10)
y_val = torch.randint(0, 2, (20,))

# 2. Oversized model for 20 samples
model = nn.Sequential(
    nn.Linear(10, 64),
    nn.ReLU(),
    nn.Linear(64, 2)
)
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.01)

# Exercise 4: TensorBoard SummaryWriter
writer = SummaryWriter("runs/overfitting_demo")

for epoch in range(50):
    # Training step
    model.train()
    optimizer.zero_grad()
    train_out = model(X_train)
    train_loss = criterion(train_out, y_train)
    train_loss.backward()

    # Exercise 5: Test breakpoint() at epoch 0
    if epoch == 0:
        print("\n--- Exercise 5: Dropping into breakpoint() ---")
        print("Try commands: p X_train.shape, p X_train.device, p model[0].weight.grad")
        print("Type 'c' and press Enter to resume training.\n")
        breakpoint()

    optimizer.step()

    # Validation step
    model.eval()
    with torch.no_grad():
        val_out = model(X_val)
        val_loss = criterion(val_out, y_val)

    # Log to TensorBoard
    writer.add_scalar("loss/train", train_loss.item(), epoch)
    writer.add_scalar("loss/val", val_loss.item(), epoch)

writer.close()
print("Training complete. Run: tensorboard --logdir=runs")
