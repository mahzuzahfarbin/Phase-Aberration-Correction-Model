import torch
import torch.optim as optim
from torchvision.utils import save_image
from model import Generator, Discriminator, wasserstein_loss, compute_gradient_penalty
import torchvision.datasets as datasets
from torch.utils.data import DataLoader
import torchvision.transforms as transforms

# Define the transforms for 3D to 2D image translation
transforms_3d_to_2d = transforms.Compose([
    # Resize the 3D volume to the desired 2D image size
    transforms.Resize((128, 128)),
    # Convert the 3D volume to a 2D image by selecting a specific slice
    transforms.Lambda(lambda x: x[:, :, 64]),  # Assuming the slice is at index 64
    # Convert the 2D image to tensor
    transforms.ToTensor(),
    # Normalize the pixel values of the 2D image between -1 and 1
    transforms.Normalize((0.5,), (0.5,))
])


# Initialize generator and discriminator
generator = Generator()
discriminator = Discriminator()

# Define loss function
loss_fn = wasserstein_loss

# Define optimizer for generator and discriminator
generator_optimizer = optim.Adam(generator.parameters(), lr=0.0002, betas=(0.5, 0.999))
discriminator_optimizer = optim.Adam(discriminator.parameters(), lr=0.0002, betas=(0.5, 0.999))

# Training loop
num_epochs = 100
batch_size = 64
lambda_gp = 10  # Coefficient for gradient penalty
n_critic = 5  # Number of times to update the critic for every generator update, arbitrary
sample_interval = 200 # Set the interval to 200 iterations
log_interval = 100  # Interval for logging training progress
latent_dim = 100 # Define the size of the fixed noise vector
fixed_noise = torch.randn(batch_size, latent_dim, 1, 1) # Generate fixed noise samples
dataset = datasets.ImageFolder(root='Images', transform=transforms_3d_to_2d)
dataloader = DataLoader(dataset=dataset, batch_size=batch_size, shuffle=True)
device = torch.device("gpu" if torch.cuda.is_available() else "cpu")


for epoch in range(num_epochs):

    for batch_idx, real_samples in enumerate(dataloader):
        real_samples = real_samples.to(device)

        # Update discriminator
        discriminator_optimizer.zero_grad()

        # Generate fake samples
        fake_samples = generator(real_samples)

        # Compute discriminator outputs
        real_outputs = discriminator(real_samples)
        fake_outputs = discriminator(fake_samples.detach())

        # Compute gradient penalty
        gradient_penalty = compute_gradient_penalty(discriminator, real_samples, fake_samples)

        # Compute discriminator loss
        discriminator_loss = loss_fn(real_outputs, fake_outputs) + lambda_gp * gradient_penalty

        # Backward pass and optimizer step
        discriminator_loss.backward()
        discriminator_optimizer.step()

        # Update generator
        if batch_idx % n_critic == 0:
            generator_optimizer.zero_grad()

            # Generate fake samples
            fake_samples = generator(real_samples)

            # Compute discriminator outputs
            fake_outputs = discriminator(fake_samples)

            # Compute generator loss
            generator_loss = -torch.mean(fake_outputs)

            # Backward pass and optimizer step
            generator_loss.backward()
            generator_optimizer.step()

        # Print training progress
        if (batch_idx + 1) % log_interval == 0:
            print(f"Epoch [{epoch + 1}/{num_epochs}], Batch [{batch_idx + 1}/{len(dataloader)}], "
                  f"Discriminator Loss: {discriminator_loss.item():.4f}, "
                  f"Generator Loss: {generator_loss.item():.4f}")

    # Save generated samples
            # Print/display curated fake images
            if epoch % 50 == 0:
                curated_images = fake_images[:num_samples]

                # Convert tensors to numpy arrays and move to CPU
                curated_images = curated_images.detach().cpu().numpy()

                # Display the curated images
                for i in range(num_samples):
                    plt.subplot(1, num_samples, i+1)
                    plt.imshow(curated_images[i].squeeze(), cmap='gray')
                    plt.axis('off')

                plt.show()
