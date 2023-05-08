if __name__ == "__main__":

    # Set device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Create generator and discriminator
    generator = UNETGenerator()
    discriminator = WassersteinDiscriminator()

    # Create the training model
    model = TrainingModel(generator, discriminator, device)

    # Prepare data
    transform = transforms.Compose([
        transforms.Resize((128, 128)),
        transforms.ToTensor(),
    ])
    dataset = ImageFolder("path/to/dataset", transform=transform)
    train_loader = DataLoader(dataset, batch_size=64, shuffle=True, num_workers=4)

    # Train the model
    model.train(train_loader, num_epochs=10)
