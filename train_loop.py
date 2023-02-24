import torch.nn as nn
import torch
import numpy as np



def train_batch(model, data, optimizer, criterion):
    model.train()
    ims, ce_masks = data
    _masks = model(ims)
    optimizer.zero_grad()
    loss, acc = criterion(_masks, ce_masks)
    loss.backward()
    optimizer.step()
    return loss.item(), acc.item()

@torch.no_grad()
def validate_batch(model, data, criterion):
    model.eval()
    ims, masks = data
    _masks = model(ims)
    loss, acc = criterion(_masks, masks)
    return loss.item(), acc.item()


def train_model(trn_dl, val_dl, batch_size, n_epochs, optimizer, criterion, model):
    # Train the model over increasing epochs
    log_train_loss = []
    log_train_acc = []
    log_val_loss = []
    log_val_acc = []

    for ex in range(n_epochs):
        print(f'START EPOCH {ex+1}')
        N = len(trn_dl)
        loss_avg = 0
        acc_avg = 0
        for bx, data in enumerate(trn_dl):
            loss, acc = train_batch(model, data, optimizer, criterion)
            print(f'{(ex+(bx+1)/N):.2f}',"  training loss=", f'{loss:.2f}', "training_accuracy=", f'{acc:.2f}')
            loss_avg+=loss
            acc_avg += batch_size * acc
        log_train_loss.append(loss_avg/N)
        log_train_acc.append(acc_avg/(N*batch_size))

        N = len(val_dl)
        loss_avg = 0
        acc_avg = 0
        for bx, data in enumerate(val_dl):
            loss, acc = validate_batch(model, data, criterion)
            loss_avg+=loss
            acc_avg += batch_size * acc
            print(f'{(ex+(bx+1)/N):.2f}',"  test_loss=", f'{loss:.2f}', "test_accuracy=", f'{acc:.2f}')
        log_val_loss.append(loss_avg/N)
        log_val_acc.append(acc_avg/(N*batch_size))

        with open(f'../dummy_dataset/model_state/log_train_loss', 'wb') as f:
            np.save(f, log_train_loss)

        with open(f'../dummy_dataset/model_state/log_train_acc', 'wb') as f:
            np.save(f, log_train_acc)

        with open(f'../dummy_dataset/model_state/log_val_loss', 'wb') as f:
            np.save(f, log_val_loss)

        with open(f'../dummy_dataset/model_state/log_val_acc', 'wb') as f:
            np.save(f, log_val_acc)

        with open(f'../dummy_dataset/model_state/n_epochs', 'wb') as f:
            np.save(f, n_epochs)


    torch.save(model.state_dict(), f'../dummy_dataset/model_state/model.pt')
