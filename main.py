import torch
import torch.nn as nn
import logging
import numpy as np
from torch.utils.data import random_split
from torch.utils.data import DataLoader
from models.unet_tranformer import PPGUnet
from models.custom_dataloader import ENTCDataset
from models.custom_dataloader import UCIBPDataset
from models.custom_dataloader import UCIBPDatasetNPY
from models.log import readable_time
from models.resample import resample
from torch.optim.lr_scheduler import ExponentialLR

# logging file configuration
logging.basicConfig(filename=f"logs/{readable_time()}.log", level=logging.INFO)

# device configuration
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def train_entc():
        
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

        # Change save directory based on the experiment
        if (epoch+1) % 5 == 0:
            model_save_path = f"checkpoints/vatta-pitta/PPGUnet_epoch{epoch}.pth"
            torch.save(model.state_dict(), model_save_path)

def train_ucibp(dataset="ucibp"):

        # dataset configuration

        if dataset == "ucibp":
            UCIBP_dataset = UCIBPDataset("dataset/uci-bp")
        elif dataset == "ucibp-npy":
            UCIBP_dataset = UCIBPDatasetNPY("dataset/uci-bp-processed")
            
        train_dataset, valid_dataset, test_dataset = random_split(UCIBP_dataset, [0.6, 0.2, 0.2], generator=torch.Generator().manual_seed(42))

        # model configuration
        num_epochs = 300
        model = PPGUnet(in_channels=1).to(device)
        criterion = nn.L1Loss()
        optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
        scheduler = ExponentialLR(optimizer, gamma=0.992354)

        # dataloader configuration
        train_dataloader = DataLoader(train_dataset, batch_size=128, shuffle=True)
        valid_dataloader = DataLoader(valid_dataset, batch_size=128, shuffle=False)
        test_dataloader = DataLoader(test_dataset, batch_size=128, shuffle=False)

        for epoch in range(num_epochs):
            model.train()
            train_loss = 0.0
            for i, data_sample in enumerate(train_dataloader):
                print(i)
                
                if dataset=="ucibp":
                    input, target = data_sample

                elif dataset=="ucibp-npy":
                    input, target, pressures = data_sample

                input = input.unsqueeze(1).to(device)
                target = target.unsqueeze(1).to(device)

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
                    
                    if dataset=="ucibp":
                        input, target = data_sample

                    elif dataset=="ucibp-npy":
                        input, target, pressures = data_sample

                    input = input.unsqueeze(1).to(device)
                    target = target.unsqueeze(1).to(device)

                    output = model(input)
                    loss = criterion(output, target)
                    valid_loss += loss.item() * input.size(0)

            valid_loss = valid_loss / len(test_dataloader.dataset)
            
            logging.info(f"Epoch {epoch+1} / {num_epochs}, Valid Loss: {valid_loss:.4f}")
            print(f"Epoch {epoch+1} / {num_epochs}, Valid Loss: {valid_loss:.4f}")

            # Change save directory based on the experiment
            # if (epoch+1) % 5 == 0:
            #     model_save_path = f"checkpoints/vatta-pitta/PPGUnet_epoch{epoch}.pth"
            #     torch.save(model.state_dict(), model_save_path)
     


def main(train_dataset="entc"):
    if train_dataset == "entc":
        train_entc()
    elif train_dataset == "ucibp":
        train_ucibp(dataset="ucibp")
    elif train_dataset=="ucibp-npy":
        train_ucibp(dataset="ucibp-npy")




if __name__ == "__main__":
    main(train_dataset="ucibp-npy")