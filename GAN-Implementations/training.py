# Define the training model
class TrainingModel:

    def __init__(self, generator, discriminator, device):
        self.generator = generator
        self.discriminator = discriminator
        self.device = device

        self.generator.to(device)
        self.discriminator.to(device)

        self.generator_optimizer = optim.Adam(generator.parameters(), lr=0.0002, betas=(0.5, 0.999))
        self.discriminator_optimizer = optim.Adam(discriminator.parameters(), lr=0.0002, betas=(0.5, 0.999))

        self.loss_fn = nn.BCELoss()

    def train(self, train_loader, num_epochs):
        for epoch in range(num_epochs):
            for batch_idx, (real_images, _) in enumerate(train_loader):
                batch_size = real_images.size(0)

                # Train the discriminator
                self.discriminator_optimizer.zero_grad()

                real_images = real_images.to(self.device)
                real_labels = torch.ones(batch_size, 1).to(self.device)
                fake_labels = torch.zeros(batch_size, 1).to(self.device)

                real_outputs = self.discriminator(real_images)
                real_loss = self.loss_fn(real_outputs, real_labels)
                real_loss.backward()

                # Generate fake images
                noise = torch.randn(batch_size, 100, 1, 1).to(self.device)
                fake_images = self.generator(noise)

                fake_outputs = self.discriminator(fake_images.detach())
                fake_loss = self.loss_fn(fake_outputs, fake_labels)
                fake_loss.backward()

                self.discriminator_optimizer.step()

                # Train the generator
                self.generator_optimizer.zero_grad()

                fake_outputs = self.discriminator(fake_images)
                generator_loss = self.loss_fn(fake_outputs, real_labels)
                generator_loss.backward()

                self.generator_optimizer.step()

                if batch_idx % 100 == 0:
                    print(f"Epoch [{epoch+1}/{num_epochs}], Batch [{batch_idx}/{len(train_loader)}], "
                          f"Discriminator Loss: {real_loss+fake_loss}, Generator Loss: {generator_loss}")
