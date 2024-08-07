def barlow_twins_loss(self, z_unmasked, z_masked, lambda_param=0.005):
    z_unmasked = (z_unmasked - z_unmasked.mean(0)) / z_unmasked.std(0)
    z_masked = (z_masked - z_masked.mean(0)) / z_masked.std(0)

    N, D = z_unmasked.size()
    c = torch.mm(z_unmasked.T, z_masked) / N

    c_diff = (c - torch.eye(D, device=c.device)).pow(2)
    on_diag = torch.diagonal(c_diff).sum()
    off_diag = (c_diff.sum() - on_diag)

    loss = on_diag + lambda_param * off_diag
    return loss

def training_step(self, batch, optimizer, lambda_param=0.005):
    elem_weights, elem_fea, self_idx, nbr_idx, cry_elem_idx = batch
    mask_idx = torch.randint(0, elem_fea.size(0), (1,)).item()

    z_unmasked, z_masked = self.forward(
        elem_weights, elem_fea, self_idx, nbr_idx, cry_elem_idx, mask_idx
    )

    loss = self.barlow_twins_loss(z_unmasked[0], z_masked[0], lambda_param)

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    return loss.item()
