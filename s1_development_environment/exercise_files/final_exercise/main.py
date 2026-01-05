import torch
from torch import nn

import typer

import matplotlib.pyplot as plt
from tqdm import tqdm

from data import corrupt_mnist
from model import MyAwesomeModel

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu")

app = typer.Typer()


@app.command()
def train(lr: float = 1e-3) -> None:
    """Train a model on MNIST."""
    print("Training day and night")
    print(lr)

    # TODO: Implement training loop here
    model = MyAwesomeModel().to(DEVICE)
    train_set, _ = corrupt_mnist()
    train_dataloader = torch.utils.data.DataLoader(train_set, batch_size=32)

    criterion = nn.NLLLoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=0.005)

    epochs = 15
    steps = 0

    train_losses = []
    for _ in tqdm(range(epochs)):
        model.train()
        batch_loss = 0
        for images, labels in train_dataloader:
            images, labels = images.to(DEVICE), labels.to(DEVICE)
            optimizer.zero_grad()

            log_ps = model(images)
            loss = criterion(log_ps, labels)
            loss.backward()
            optimizer.step()
            batch_loss += loss.item()

        train_losses.append(batch_loss)

    torch.save(model.state_dict(), "models/s1_model.pt")

    plt.plot(range(0,epochs),train_losses)
    plt.title("Training loss")
    plt.show()


@app.command()
def evaluate(model_checkpoint: str) -> None:
    """Evaluate a trained model."""
    print("Evaluating like my life depends on it")
    print(model_checkpoint)

    # TODO: Implement evaluation logic here
    model = MyAwesomeModel().to(DEVICE)
    model.load_state_dict(torch.load(model_checkpoint))
    model.eval()
    _, test_set = corrupt_mnist()
    test_dataloader = torch.utils.data.DataLoader(test_set, batch_size=32)

    with torch.no_grad():
        batch_test_loss = 0
        batch_accuracy = []
        for images, labels in test_dataloader:
            images, labels = images.to(DEVICE), labels.to(DEVICE)
            log_ps = model(images)
            ps = torch.exp(log_ps)

            top_p, top_class = ps.topk(k=1, dim=1)
            equals = top_class == labels.view(*top_class.shape)

            accuracy = torch.mean(equals.type(torch.FloatTensor))
            batch_accuracy.append(accuracy.item())
        accuracy = torch.mean(torch.Tensor(batch_accuracy))
        print(100*accuracy.item(), "%")



if __name__ == "__main__":
    app()
