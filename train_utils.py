import torch as t
from torchvision import utils

from torch.cuda import amp
from tqdm import tqdm


def save_checkpoint(state, filepath):
    print("Saving checkpoint =>")
    t.save(state, filepath)


def load_checkpoint(model, filepath):
    print("=> Loading checkpoint")

    checkpoint = t.load(filepath)
    model.load_state_dict(checkpoint["state_dict"])

    return model


def check_accuracy(loader, model, device):
    model = model.to(device)
    model.eval()

    dice_score = 0.

    with t.no_grad():
        for images, masks in loader:
            images = images.to(device)
            masks = masks.unsqueeze(1).to(device)

            preds = t.sigmoid(model(images))
            preds = (preds > 0.5).float()

            dice_score += 2 * (preds * masks).sum() / ((preds + masks).sum() + 1e-8)

    dice_score = (dice_score / len(loader)).item()
    print(f'Dice Score: {round(dice_score, 4)}')


def save_preds_as_imgs(loader, model, device, folder):
    model = model.to(device)
    model.eval()

    with t.no_grad():
        for idx, (images, masks) in enumerate(loader):
            if (idx > 1): # save pred for only 2 batches
                break
                

            images = images.to(device)

            preds = t.sigmoid(model(images))
            preds = (preds > 0.5).float()

            utils.save_image(preds, f"{folder}/pred_{idx}.png")
            utils.save_image(masks.unsqueeze(1), f"{folder}/mask_{idx}.png")


def train_fn(loader, model, optimizer, loss_fn, scaler, device):
    model = model.to(device)
    model.train()

    loop = tqdm(loader)

    for images, masks in loop:
        images = images.to(device)
        masks = masks.unsqueeze(1).to(device)

        # forward
        with amp.autocast():
            preds = model(images)
            loss = loss_fn(preds, masks)

        # backprop
        optimizer.zero_grad()
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()

        # update tqdm loop
        loop.set_postfix(loss=loss.item())
