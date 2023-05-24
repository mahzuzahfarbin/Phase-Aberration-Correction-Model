def wasserstein_loss(real_output, fake_output):
    """The Wasserstein critic:
    Critic Loss = [avg. critic score on real image] - [avg. critic score on
    fake image]"""
    return torch.mean(real_output) - torch.mean(fake_output)
