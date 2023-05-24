def compute_gradient_penalty(discriminator, real_samples, fake_samples):
    """Gradient Penalty computation."""
    alpha = torch.rand(real_samples.size(0), 1, 1, 1, device=real_samples.device)
    interpolated_samples = alpha * real_samples + (1 - alpha) * fake_samples
    interpolated_samples.requires_grad_(True)
    interpolated_outputs = discriminator(interpolated_samples)
    gradients = torch.autograd.grad(outputs=interpolated_outputs, inputs=interpolated_samples,
                                    grad_outputs=torch.ones(interpolated_outputs.size(), device=real_samples.device),
                                    create_graph=True, retain_graph=True, only_inputs=True)[0]
    gradients = gradients.view(gradients.size(0), -1)
    gradient_penalty = torch.mean((gradients.norm(2, dim=1) - 1) ** 2)
    return gradient_penalty
