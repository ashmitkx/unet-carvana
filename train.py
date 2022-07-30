from torch import optim, nn
from torch.cuda import amp

from config import *
from train_utils import *
from unet import UNet
from data import CarvanaDataset


def train():
    train_loader, val_loader = CarvanaDataset.get_dataloaders(
        IMAGE_HEIGHT, IMAGE_WIDTH, TRAIN_IMG_DIR, TRAIN_MASK_DIR, VAL_IMG_DIR, VAL_MASK_DIR,
        BATCH_SIZE, PIN_MEMORY, NUM_WORKERS
    )
    model = UNet(in_ch=3, out_ch=1)
    loss_fn = nn.BCEWithLogitsLoss()
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)
    scaler = amp.GradScaler()

    if LOAD_MODEL:
        model = load_checkpoint(model, filepath=MODEL_RESTORE_PATH)
        check_accuracy(val_loader, model, device=DEVICE)

    for epoch in range(NUM_EPOCHS):
        print(f'{epoch+1}/{NUM_EPOCHS}:')

        # one training pass
        train_fn(train_loader, model, optimizer, loss_fn, scaler, device=DEVICE)

        # save model
        state = {'state_dict': model.state_dict(), 'optimizer': optimizer.state_dict()}
        save_checkpoint(state, filepath=MODEL_SAVE_PATH)

        # check accuracy
        check_accuracy(val_loader, model, device=DEVICE)

    # save preds
    save_preds_as_imgs(val_loader, model, device=DEVICE, folder=PRED_EX_SAVE_PATH)


if __name__ == '__main__':
    train()
