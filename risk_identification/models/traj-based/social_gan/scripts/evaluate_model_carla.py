import argparse
import os
import sys
import numpy
import torch

from attrdict import AttrDict

from sgan.data.loader_v2 import data_loader
from sgan.models import TrajectoryGenerator
from sgan.losses import displacement_error, final_displacement_error
from sgan.utils import relative_to_abs, get_dset_path

parser = argparse.ArgumentParser()
parser.add_argument('--model_path', type=str)
parser.add_argument('--result_dir', type=str)
parser.add_argument('--infer_data', type=str)
parser.add_argument('--sc_type', type=str)
parser.add_argument('--num_samples', default=1, type=int)
parser.add_argument('--pred_len', default=30, type=int)
parser.add_argument('--dset_type', default='test', type=str)


def get_generator(checkpoint, new_args):
    args = AttrDict(checkpoint['args'])
    generator = TrajectoryGenerator(
        obs_len=args.obs_len,
        pred_len=new_args.pred_len,
        embedding_dim=args.embedding_dim,
        encoder_h_dim=args.encoder_h_dim_g,
        decoder_h_dim=args.decoder_h_dim_g,
        mlp_dim=args.mlp_dim,
        num_layers=args.num_layers,
        noise_dim=args.noise_dim,
        noise_type=args.noise_type,
        noise_mix_type=args.noise_mix_type,
        pooling_type=args.pooling_type,
        pool_every_timestep=args.pool_every_timestep,
        dropout=args.dropout,
        bottleneck_dim=args.bottleneck_dim,
        neighborhood_size=args.neighborhood_size,
        grid_size=args.grid_size,
        batch_norm=args.batch_norm)
    generator.load_state_dict(checkpoint['g_state'])
    generator.cuda()
    generator.train()
    return generator


def evaluate_helper(error, seq_start_end):
    sum_ = 0
    error = torch.stack(error, dim=1)

    for (start, end) in seq_start_end:
        start = start.item()
        end = end.item()
        _error = error[start:end]
        _error = torch.sum(_error, dim=0)
        _error = torch.min(_error)
        sum_ += _error
    return sum_


def evaluate(args, loader, generator, num_samples, infer_file, result_dir):
    ade_outer, fde_outer = [], []
    total_traj = 0
    secenario = infer_file.split('-with-')[0]
    variant = infer_file.split('-with-')[1].split('.')[0]
    scenario_path = os.path.join(result_dir, secenario)
    variant_path = os.path.join(scenario_path, variant)
    if not os.path.exists(scenario_path):
        os.mkdir(scenario_path)
    if not os.path.exists(variant_path):
        os.mkdir(variant_path)
    
            
    
    with torch.no_grad():
        for batch in loader:
            batch = [tensor.cuda() for tensor in batch]
            (obs_traj, pred_traj_gt, obs_traj_rel, pred_traj_gt_rel,
             non_linear_ped, loss_mask, seq_start_end,
             ped_list, frame_num) = batch

            ade, fde = [], []
            # total_traj += pred_traj_gt.size(1)

            pred_traj_fake_rel = generator(
                obs_traj, obs_traj_rel, seq_start_end
            )
            pred_traj_fake = relative_to_abs(
                pred_traj_fake_rel, obs_traj[-1]
            )
            # ade.append(displacement_error(
            #     pred_traj_fake, pred_traj_gt, mode='raw'
            # ))
            # fde.append(final_displacement_error(
            #     pred_traj_fake[-1], pred_traj_gt[-1], mode='raw'
            # ))

            # Save to txt file
            frame_num = frame_num.reshape(-1, args.obs_len).detach().cpu().numpy()
            batch_num = frame_num.shape[0]
            ped_list_batch = ped_list.reshape(batch_num,
                                              -1).detach().cpu().numpy()

            obs_traj_new = obs_traj.reshape(
                args.obs_len, batch_num, -1, 2).permute(1, 0, 2, 3)
            pred_traj_fake_new = pred_traj_fake.reshape(
                args.pred_len, batch_num, -1, 2).permute(1, 0, 2, 3)
            traj_mix = torch.cat(
                (obs_traj_new, pred_traj_fake_new), 1).detach().cpu().numpy()

            for i, frame_data in enumerate(traj_mix):
                f_name = f'{str(frame_num[i][0])}-{str(frame_num[i][-1] + args.pred_len)}.txt'
                write_path = os.path.join(variant_path, f_name)
                f = open(write_path, 'w')
                for j, frame in enumerate(frame_data):
                    if j >= frame_num[i].shape[0]:
                        frame_id = str(frame_num[i][-1] + j - frame_num[i].shape[0] + 1)
                    else:
                        frame_id = str(frame_num[i][j])
                    for k, traj in enumerate(frame):
                        traj_str = frame_id + '\t' + str(
                            ped_list_batch[i, k]) + '\t' + str(traj[0]) + '\t' + str(traj[1]) + '\n'
                        # print(traj_str)
                        f.write(traj_str)
                f.close()
        

        #     ade_sum = evaluate_helper(ade, seq_start_end)
        #     fde_sum = evaluate_helper(fde, seq_start_end)

        #     ade_outer.append(ade_sum)
        #     fde_outer.append(fde_sum)
        # ade = sum(ade_outer) / (total_traj * args.pred_len)
        # fde = sum(fde_outer) / (total_traj)

        return ade, fde


def main(args):
    if os.path.isdir(args.model_path):
        filenames = os.listdir(args.model_path)
        filenames.sort()
        paths = [
            os.path.join(args.model_path, file_) for file_ in filenames
        ]
    else:
        paths = [args.model_path]

    for path in paths:
        checkpoint = torch.load(path)
        generator = get_generator(checkpoint, args)
        _args = AttrDict(checkpoint['args'])
        _args.dataset_name = args.sc_type
        path = get_dset_path(_args.dataset_name, args.dset_type, carla=True)
        _args.skip = 1
        _args.pred_len = args.pred_len
        _, loader = data_loader(
            _args, path, phase='infer', infer_data=args.infer_data)
        ade, fde = evaluate(_args, loader, generator,
                            args.num_samples, args.infer_data, args.result_dir)
        # print('Dataset: {}, Pred Len: {}, ADE: {:.2f}, FDE: {:.2f}'.format(
        #     _args.dataset_name, _args.pred_len, ade, fde))


if __name__ == '__main__':
    args = parser.parse_args()
    main(args)
