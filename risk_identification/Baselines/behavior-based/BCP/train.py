
from datasets import GCNDataLayer as DataLayer
from models import GCN as Model
import utils as utl
import config as cfg

from tqdm import tqdm
from torchvision import transforms
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
import torch.optim as optim
import torch.nn as nn
import torch
import numpy as np
import json
import copy
import time
import os


def to_device(x, device):
    return x.to(device)


def count_parameters(model):

    params = sum([np.prod(p.size()) for p in model.parameters()])
    print('Total parameters: ', params)

    model_parameters = filter(lambda p: p.requires_grad, model.parameters())
    params = sum([np.prod(p.size()) for p in model_parameters])
    print('Total trainable parameters: ', params)


def load_weight(model, checkpoint):

    state_dict = torch.load(checkpoint)
    state_dict_copy = {}
    for key in state_dict.keys():
        state_dict_copy[key[7:]] = state_dict[key]

    model.load_state_dict(state_dict_copy)
    return copy.deepcopy(model)


def create_model(args, device):

    model = Model(args.time_steps, pretrained=args.pretrained,
                  partialConv=args.partial_conv, use_intention=args.use_intention, NUM_BOX=args.num_box)

    if args.ckpt_path != "":
        model = load_weight(model, args.ckpt_path)

    model = nn.DataParallel(model).to(device)
    count_parameters(model)

    return model


def create_data_loader(args):

    camera_transforms = transforms.Compose([
        # transforms.Resize(args.img_resize, interpolation=InterpolationMode.NEAREST),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225]),
    ])

    data_sets = {
        phase: DataLayer(
            data_root=args.data_root,
            behavior_root=args.behavior_root,
            state_root=args.state_root,
            scenario=getattr(args, phase+"_session_set"),
            camera_transforms=camera_transforms,
            num_box=args.num_box,
            raw_img_size=args.img_size,
            img_resize=args.img_resize,
            time_steps=args.time_steps,
            data_augmentation=args.data_augmentation,
            phase=phase
        )
        for phase in args.phases
    }

    data_loaders = {
        phase: DataLoader(
            data_sets[phase],
            batch_size=args.batch_size,
            shuffle=True,
            num_workers=args.num_workers,
            pin_memory=True,
        )
        for phase in args.phases
    }

    return data_loaders


def write_result(args, epoch, vel_metrics, target_metrics, losses, loss_go, loss_stop):

    result = {}
    result['Epoch'] = epoch
    result['lr'] = args.lr
    result['loss_weights'] = args.loss_weights

    for phase in args.phases:
        phase_result = utl.compute_result(
            args.class_index, vel_metrics[phase], target_metrics[phase])

        phase_result['loss'] = losses[phase] / \
            len(data_loaders[phase].dataset)
        phase_result['loss_go'] = loss_go[phase] / \
            sum(phase_result['confusion_matrix'][0])
        phase_result['loss_stop'] = loss_stop[phase] / \
            sum(phase_result['confusion_matrix'][1])

        result[phase] = phase_result
        print(phase_result)

        # save training logs in tensorboard
        writer.add_scalars(main_tag=f'Loss/{phase}',
                           tag_scalar_dict={'total': phase_result['loss'],
                                            'go': phase_result['loss_go'],
                                            'stop': phase_result['loss_stop'],
                                            },
                           global_step=epoch)

        writer.add_scalars(main_tag=f'AP/{phase}',
                           tag_scalar_dict={'mAP': phase_result['mAP'],
                                            'go': phase_result['AP']['go'],
                                            'stop': phase_result['AP']['stop'],
                                            },
                           global_step=epoch)

        writer.add_scalars(main_tag=f'Accuracy/{phase}',
                           tag_scalar_dict={'total': phase_result['ACC_total'],
                                            'go': phase_result['ACC_go'],
                                            'stop': phase_result['ACC_stop'],
                                            },
                           global_step=epoch)

    # save training logs in json type
    with open(args.result_path) as f:
        history_result = json.load(f)
    history_result.append(result)
    with open(args.result_path, "w") as f:
        json.dump(history_result, f, indent=4)

    torch.save(model.state_dict(), os.path.join(
        args.ckpt_folder, f"epoch-{epoch}.pth"))

    return result


def train(args, model, data_loaders, device):

    # softmax = nn.Softmax(dim=1).to(device)
    weights = args.loss_weights
    # class_weights = torch.FloatTensor(weights).to(device)
    # criterion = nn.BCELoss(weight=class_weights, reduction='none').to(device)

    optimizer = optim.Adam(model.parameters(), lr=args.lr,
                           weight_decay=args.weight_decay)

    for epoch in range(1, args.epochs+1):

        losses = {phase: 0.0 for phase in args.phases}
        loss_go = {phase: 0.0 for phase in args.phases}
        loss_stop = {phase: 0.0 for phase in args.phases}
        vel_metrics = {phase: [] for phase in args.phases}
        target_metrics = {phase: [] for phase in args.phases}

        start = time.time()
        for phase in args.phases:

            is_training = (phase == "train")
            if not is_training and epoch % args.val_interval != 0:
                continue

            torch.set_grad_enabled(is_training)
            with tqdm(data_loaders[phase], unit="batch") as tepoch:
                for batch_idx, (camera_inputs, trackers,
                                mask, vel_target, intention_inputs, state_inputs) in enumerate(tepoch):
                    tepoch.set_description(f"Epoch {epoch:2}/{args.epochs}")
                    batch_size = camera_inputs.shape[0]

                    # BxTxCxHxW
                    camera_inputs = to_device(camera_inputs, device)
                    mask = mask.to(device)
                    # BxTxNx4
                    trackers = to_device(trackers, device)
                    # B
                    vel_target = to_device(vel_target, device).view(-1)
                    # Bx10
                    intention_inputs = to_device(intention_inputs, device)
                    # BxTxNx2
                    state_inputs = to_device(state_inputs, device)

                    vel = model(camera_inputs, trackers, mask,
                                intention_inputs, state_inputs, device)

                    # dynamic loss weight
                    weight = torch.where(
                        vel_target == 0, weights[0], weights[1])
                    criterion = nn.BCELoss(
                        weight=weight, reduction='none').to(device)

                    loss = criterion(vel, vel_target.float())

                    for j in range(len(vel_target)):
                        if vel_target[j].item() == 0:
                            loss_go[phase] += loss[j].item()
                        elif vel_target[j].item() == 1:
                            loss_stop[phase] += loss[j].item()

                    loss = torch.mean(loss)
                    losses[phase] += (loss.item()*batch_size)

                    pred = torch.where(vel > 0.5, 1, 0)

                    print(pred)
                    print(vel_target)
                    print()

                    vel = np.array(vel.detach().to(
                        'cpu').numpy()).reshape((-1, 1))
                    vel_target = np.array(
                        vel_target.detach().to('cpu').numpy())
                    vel = np.concatenate((1-vel, vel), 1)

                    vel_metrics[phase].extend(vel)
                    target_metrics[phase].extend(vel_target)

                    if is_training:
                        optimizer.zero_grad()
                        loss.backward()
                        optimizer.step()

                    tepoch.set_postfix(loss=loss.item())

        end = time.time()

        result = write_result(args, epoch, vel_metrics,
                              target_metrics, losses, loss_go, loss_stop)

        print(f"Epoch {epoch:2d} | train loss: {result['train']['loss']:.5f} | \
            validation loss: {result['test']['loss']:.5f} mAP: {result['test']['mAP']:.5f} | running time: {end-start:.2f} sec")


if __name__ == '__main__':

    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_type', default='all', type=str, required=True)
    parser.add_argument('--phases', default=['train', 'test'], type=list)
    parser.add_argument('--gpu', default='0,1,2,3', type=str)
    parser.add_argument('--epochs', default=40, type=int)
    parser.add_argument('--batch_size', default=8, type=int)
    parser.add_argument('--lr', default=1e-07, type=float)
    parser.add_argument('--weight_decay', default=1e-02, type=float)
    parser.add_argument('--loss_weights', default=[1.0, 1.55], type=list)
    parser.add_argument('--num_workers', default=4, type=int)
    parser.add_argument('--time_steps', default=5, type=int)
    parser.add_argument('--pretrained', default=True, type=bool)
    parser.add_argument('--partial_conv', default=True, type=bool)
    parser.add_argument('--use_intention', action='store_true', default=False)
    parser.add_argument('--data_augmentation', default=True, type=bool)
    parser.add_argument('--ckpt_path', default="", type=str)
    parser.add_argument('--Method', default="", type=str)

    args = cfg.parse_args(parser)
    args = cfg.read_data(args, valiation=True)

    os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    model = create_model(args, device)
    data_loaders = create_data_loader(args)

    writer = SummaryWriter(args.log_folder)
    train(args, model, data_loaders, device)
    writer.close()
