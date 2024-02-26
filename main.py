import torch
import torch.nn as nn
import logging
from torch.utils.data import random_split
from torch.utils.data import DataLoader
from models.unet_tranformer import PPGUnet
from models.custom_dataloader import ENTCDataset
from models.log import readable_time
from torch.optim.lr_scheduler import ExponentialLR

# logging file configuration
logging.basicConfig(filename=f"logs/{readable_time()}.log", level=logging.INFO)

# device configuration
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# dataset configuration
ENTC_dataset = ENTCDataset("dataset/paraqum-dataset/")
train_dataset, valid_dataset, test_dataset = random_split(ENTC_dataset, [0.6, 0.2, 0.2], generator=torch.Generator().manual_seed(42))

# model configuration
num_epochs = 300
model = PPGUnet(in_channels=1).to(device)
criterion = nn.L1Loss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
scheduler = ExponentialLR(optimizer, gamma=0.992354)

# dataloader configuration
train_dataloader = DataLoader(train_dataset, batch_size=64, shuffle=True)
valid_dataloader = DataLoader(valid_dataset, batch_size=64, shuffle=False)
test_dataloader = DataLoader(test_dataset, batch_size=64, shuffle=False)


def main():
    for epoch in range(num_epochs):
        model.train()
        train_loss = 0.0
        for i, data_sample in enumerate(train_dataloader):

            input = data_sample[:, 0, :].unsqueeze(1)/1024
            target = data_sample[:, 1, :].unsqueeze(1)/1024   

            input = input.to(device)
            target = target.to(device)

            optimizer.zero_grad()
            output = model(input)
            loss = criterion(output, target)
            loss.backward()
            optimizer.step()

            train_loss += loss.item() * input.size(0)

        train_loss = train_loss / len(train_dataloader.dataset)

        logging.info(f"Epoch {epoch+1} / {num_epochs}, Train Loss: {train_loss:.4f}")
        print(f"Epoch {epoch+1} / {num_epochs}, Train Loss: {train_loss:.4f}")

        # Adjusting learning rate
        scheduler.step()

        model.eval()

        valid_loss = 0.0

        with torch.no_grad():
            for i, data_sample in enumerate(valid_dataloader):
                input = data_sample[:, 0, :].unsqueeze(1)/1024
                target = data_sample[:, 1, :].unsqueeze(1)/1024

                input = input.to(device)
                target = target.to(device)

                output = model(input)
                loss = criterion(output, target)
                valid_loss += loss.item() * input.size(0)

        valid_loss = valid_loss / len(test_dataloader.dataset)
        
        logging.info(f"Epoch {epoch+1} / {num_epochs}, Valid Loss: {valid_loss:.4f}")
        print(f"Epoch {epoch+1} / {num_epochs}, Valid Loss: {valid_loss:.4f}")


        model_save_path = f"checkpoints/PPGUnet_epoch{epoch}.pth"
        torch.save(model.state_dict(), model_save_path)



if __name__ == "__main__":
    main()