# scripts/finetune.py
import os  
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import transforms, datasets, models
import mlflow
from app.model import load_model, setup_mlflow
import argparse
from tqdm import tqdm

def create_datasets(data_dir, batch_size=32):
    data_transforms = {
        'train': transforms.Compose([
            transforms.RandomResizedCrop(224),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ]),
        'val': transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ]),
    }
    
    image_datasets = {
        x: datasets.ImageFolder(os.path.join(data_dir, x), data_transforms[x])
        for x in ['train', 'val']
    }
    
    dataloaders = {
        x: DataLoader(image_datasets[x], batch_size=batch_size, shuffle=True, num_workers=4)
        for x in ['train', 'val']
    }
    
    class_names = image_datasets['train'].classes
    
    return dataloaders['train'], dataloaders['val'], class_names

def train_model(model, train_loader, val_loader, criterion, optimizer, scheduler, 
                num_epochs=10, device='cuda'):
    model = model.to(device)
    best_acc = 0.0
    
    for epoch in range(num_epochs):
        print(f'Epoch {epoch+1}/{num_epochs}')
        print('-' * 10)
        
        model.train()
        running_loss = 0.0
        running_corrects = 0
        
        pbar = tqdm(train_loader, desc=f"Training Epoch {epoch+1}")
        for inputs, labels in pbar:
            inputs = inputs.to(device)
            labels = labels.to(device)
            
            optimizer.zero_grad()
            
            with torch.set_grad_enabled(True):
                outputs = model(inputs)
                _, preds = torch.max(outputs, 1)
                loss = criterion(outputs, labels)
                
                loss.backward()
                optimizer.step()
            
            running_loss += loss.item() * inputs.size(0)
            running_corrects += torch.sum(preds == labels.data)
            
            pbar.set_postfix(loss=loss.item())
        
        epoch_loss = running_loss / len(train_loader.dataset)
        epoch_acc = running_corrects.double() / len(train_loader.dataset)
        
        print(f'Train Loss: {epoch_loss:.4f} Acc: {epoch_acc:.4f}')
        
        mlflow.log_metric(f"train_loss", epoch_loss, step=epoch)
        mlflow.log_metric(f"train_acc", epoch_acc.item(), step=epoch)
        
        model.eval()
        running_loss = 0.0
        running_corrects = 0
        
        pbar = tqdm(val_loader, desc=f"Validation Epoch {epoch+1}")
        for inputs, labels in pbar:
            inputs = inputs.to(device)
            labels = labels.to(device)
            
            with torch.no_grad():
                outputs = model(inputs)
                _, preds = torch.max(outputs, 1)
                loss = criterion(outputs, labels)
            
            running_loss += loss.item() * inputs.size(0)
            running_corrects += torch.sum(preds == labels.data)
            
            pbar.set_postfix(loss=loss.item())
        
        scheduler.step()
        
        epoch_loss = running_loss / len(val_loader.dataset)
        epoch_acc = running_corrects.double() / len(val_loader.dataset)
        
        print(f'Val Loss: {epoch_loss:.4f} Acc: {epoch_acc:.4f}')
        
        mlflow.log_metric(f"val_loss", epoch_loss, step=epoch)
        mlflow.log_metric(f"val_acc", epoch_acc.item(), step=epoch)
        
        if epoch_acc > best_acc:
            best_acc = epoch_acc
            torch.save(model.state_dict(), "models/best_model.pth")
            mlflow.log_artifact("models/logs/best_model.pth")
    
    return model

def register_finetuned_model(model_path, model_name, class_names):
    from mlflow.tracking import MlflowClient
    
    model = models.resnet18(pretrained=False)
    model.fc = nn.Linear(model.fc.in_features, len(class_names))
    model.load_state_dict(torch.load(model_path))
    model.eval()
    
    with open("class_names.txt", "w") as f:
        for class_name in class_names:
            f.write(f"{class_name}\n")
    
    mlflow.log_artifact("class_names.txt")
    
    model_info = mlflow.pytorch.log_model(
        model,
        artifact_path="finetuned_model",
        registered_model_name=model_name
    )
    
    client = MlflowClient()
    
    for mv in client.search_model_versions(f"run_id='{mlflow.active_run().info.run_id}'"):
        model_version = mv.version
        client.transition_model_version_stage(
            name=model_name,
            version=model_version,
            stage="Production"
        )
        return model_version
    
    return None

def finetune_model(data_dir, num_epochs=10, batch_size=32, learning_rate=0.001, 
                   output_model_name="finetuned_resnet18"):
    setup_mlflow()
    
    with mlflow.start_run(run_name="model_finetuning"):
        mlflow.log_params({
            "epochs": num_epochs,
            "batch_size": batch_size,
            "learning_rate": learning_rate,
            "data_directory": data_dir
        })
        
        base_model, _ = load_model()
        
        train_loader, val_loader, class_names = create_datasets(data_dir, batch_size)
        num_classes = len(class_names)
        
        print(f"Fine-tuning model for {num_classes} classes: {class_names}")
        mlflow.log_param("num_classes", num_classes)
        mlflow.log_param("classes", class_names)
        
        model = models.resnet18(pretrained=False)
        model.load_state_dict(base_model.state_dict())
        model.fc = nn.Linear(model.fc.in_features, num_classes)
        
        # # Uncomment below to freeze some layers and only train the final ones
        # for param in list(model.parameters())[:-2]:
        #     param.requires_grad = False
        
        criterion = nn.CrossEntropyLoss()
        optimizer = optim.SGD(model.parameters(), lr=learning_rate, momentum=0.9)
        scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=7, gamma=0.1)
        
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"Using device: {device}")
        
        model = train_model(
            model, train_loader, val_loader, criterion, optimizer, scheduler,
            num_epochs=num_epochs, device=device
        )
        
        torch.save(model.state_dict(), "models/finetuned_model.pth")
        mlflow.log_artifact("models/logs/finetuned_model.pth")
        
        model_version = register_finetuned_model(
            "models/best_model.pth",
            output_model_name,
            class_names
        )
        
        print(f"Fine-tuning complete. Model version {model_version} registered.")
        print(f"To use the new model, update the model_name parameter in load_model()")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Fine-tune ResNet model")
    parser.add_argument("--data_dir", type=str, required=True, 
                        help="Directory with train and val folders")
    parser.add_argument("--epochs", type=int, default=10,
                        help="Number of training epochs")
    parser.add_argument("--batch_size", type=int, default=32,
                        help="Training batch size")
    parser.add_argument("--lr", type=float, default=0.001,
                        help="Initial learning rate")
    parser.add_argument("--model_name", type=str, default="finetuned_resnet18",
                        help="Name for the model in MLflow")
    
    args = parser.parse_args()
    
    finetune_model(
        args.data_dir,
        num_epochs=args.epochs,
        batch_size=args.batch_size,
        learning_rate=args.lr,
        output_model_name=args.model_name
    )