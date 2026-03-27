import argparse
import torch
from torch import nn
import LoadData as ld
import models
import adan
import util
import numpy as np
import matplotlib.pyplot as plt

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


def compute_feature_mse(pred_tensor, true_tensor, scale=100.0):
    """
    计算每个温度通道的 MSE。
    输入形状: [T, B, F]
    返回形状: [F]
    """
    squared_error = (scale * pred_tensor - scale * true_tensor) ** 2
    return squared_error.mean(dim=(0, 1))


# ==================== 绘图复用函数 ====================
def plot_results(pred_tensor, true_tensor, window_size, features, title, save_path):
    """
    将预测张量和真实张量拼接，并绘制为连续的时间序列图保存。
    """
    total_batches = pred_tensor.shape[1]
    pred_list = []
    true_list = []

    for i in range(0, total_batches, window_size):
        pred_list.append(pred_tensor[:, i, :].cpu().numpy())
        true_list.append(true_tensor[:, i, :].cpu().numpy())

    pred_np = np.concatenate(pred_list, axis=0) * 100
    true_np = np.concatenate(true_list, axis=0) * 100
    time_axis = np.arange(pred_np.shape[0]) * (4.0 / 3600.0)

    feature_names = ['Permanent Magnet (PM)', 'Stator Yoke (SY)', 'Stator Tooth (ST)', 'Stator Winding (SW)']

    fig, axs = plt.subplots(2, 2, figsize=(15, 10), dpi=100)
    fig.suptitle(title, fontsize=18, fontweight='bold')

    for i in range(features):
        row = i // 2
        col = i % 2
        ax = axs[row, col]
        ax.plot(time_axis, true_np[:, i], label='Measurement', color='steelblue', linewidth=2.0)
        ax.plot(time_axis, pred_np[:, i], label='Estimation', color='indianred', linestyle='--', linewidth=1.5)
        ax.set_title(feature_names[i] if i < len(feature_names) else f'Feature {i + 1}', fontsize=14)
        ax.set_xlabel('Time (Hours)', fontsize=12)
        ax.set_ylabel('Temperature (°C)', fontsize=12)
        ax.grid(True, linestyle=':', alpha=0.7)
        ax.legend(loc='best', fontsize=11)

    plt.tight_layout()
    plt.subplots_adjust(top=0.92)
    plt.savefig(save_path, bbox_inches='tight', dpi=300)
    print(f"Plot saved successfully to: {save_path}")
    plt.close(fig)


def plot_errors(pred_tensor, true_tensor, window_size, features, title, save_path):
    """
    计算并绘制预测误差随时间的变化
    """
    total_batches = pred_tensor.shape[1]
    pred_list = []
    true_list = []

    for i in range(0, total_batches, window_size):
        pred_list.append(pred_tensor[:, i, :].cpu().numpy())
        true_list.append(true_tensor[:, i, :].cpu().numpy())

    pred_np = np.concatenate(pred_list, axis=0) * 100
    true_np = np.concatenate(true_list, axis=0) * 100
    
    # 计算误差 (预测值 - 真实值)
    error_np = pred_np - true_np
    time_axis = np.arange(error_np.shape[0]) * (4.0 / 3600.0)

    feature_names = ['Permanent Magnet (PM)', 'Stator Yoke (SY)', 'Stator Tooth (ST)', 'Stator Winding (SW)']

    fig, axs = plt.subplots(2, 2, figsize=(15, 10), dpi=100)
    fig.suptitle(title, fontsize=18, fontweight='bold')

    for i in range(features):
        row = i // 2
        col = i % 2
        ax = axs[row, col]
        ax.plot(time_axis, error_np[:, i], label='Error (Estimation - Measurement)', color='darkorange', linewidth=1.5)
        # 添加0刻度基准线，方便查看正负误差
        ax.axhline(y=0, color='black', linestyle='--', linewidth=1.2, alpha=0.8) 
        ax.set_title(feature_names[i] if i < len(feature_names) else f'Feature {i + 1}', fontsize=14)
        ax.set_xlabel('Time (Hours)', fontsize=12)
        ax.set_ylabel('Error (°C)', fontsize=12)
        ax.grid(True, linestyle=':', alpha=0.7)
        ax.legend(loc='best', fontsize=11)

    plt.tight_layout()
    plt.subplots_adjust(top=0.92)
    plt.savefig(save_path, bbox_inches='tight', dpi=300)
    print(f"Error plot saved successfully to: {save_path}")
    plt.close(fig)

# ============================================================

def Train(arguments):
    util.seed_torch(arguments.seed)

    # DataSet
    train, valid, test = ld.loadData(windows=arguments.estimate_window + arguments.predict_window,
                                     samples=arguments.train_samples, down=arguments.down_rate)
    train, valid, test = util.to_tensor(train).to(device), util.to_tensor(valid).to(device), \
        util.to_tensor(test).to(device)
    print("Dataset Shapes:", train.shape, valid.shape, test.shape)

    train_true = train[:, arguments.estimate_window:, arguments.control:].permute(1, 0, 2)
    valid_true = valid[:, arguments.estimate_window:, arguments.control:].permute(1, 0, 2)
    test_true = test[:, arguments.estimate_window:, arguments.control:].permute(1, 0, 2)

    train_loader = ld.loader(train, arguments.estimate_window, arguments.batch, arguments.control,
                             num_workers=arguments.num_workers)

    # Init the structured linear neural dynamics model
    dynamic = models.Dynamic(
        arguments.control, arguments.estimate_window, arguments.hidden_size, arguments.features,
        arguments.n_layers, arguments.model,
        perron_lambda_min=arguments.perron_lambda_min, perron_lambda_max=arguments.perron_lambda_max
    ).to(device)
    print(f'Total parameters of {arguments.model} is {util.get_parameters(dynamic)}')

    # Train
    SMAE = nn.SmoothL1Loss()
    MSE = nn.MSELoss()
    if arguments.train:
        optimizer = adan.Adan(
            dynamic.parameters(),
            lr=arguments.lr,
            weight_decay=arguments.weight_decay,
            max_grad_norm=arguments.max_grad_norm
        )
        early_stop = util.EarlyStopping(
            patience=arguments.patience, cold=3,
            path=f'checkpoints/{arguments.model}.pth',
            use_mse=arguments.save_by_mse
        )
        for epoch in range(arguments.epochs):
            dynamic.train()
            epoch_r, epoch_p, n_batches = 0.0, 0.0, 0
            for step, (oc, data, label) in enumerate(train_loader):
                label = label.permute(1, 0, 2)
                output, hidden_states = dynamic(data.permute(1, 0, 2), oc.permute(1, 0, 2))
                r_loss = SMAE(label, output)
                p_loss = SMAE(torch.abs(hidden_states[:-1, :, :]), torch.abs(hidden_states[1:, :, :]))
                loss = r_loss + arguments.p_loss_weight * p_loss
                optimizer.zero_grad()
                loss.backward()
                if arguments.max_grad_norm > 0:
                    torch.nn.utils.clip_grad_norm_(dynamic.parameters(), arguments.max_grad_norm)
                optimizer.step()
                epoch_r += util.to_numpy(r_loss)
                epoch_p += util.to_numpy(p_loss)
                n_batches += 1

            # Valid
            dynamic.eval()
            with torch.no_grad():
                val_o, _ = dynamic(valid[:, 0:arguments.estimate_window, arguments.control:].permute(1, 0, 2),
                                   valid[:, arguments.estimate_window:, 0:arguments.control].permute(1, 0, 2))
                val_loss = SMAE(100 * val_o, 100 * valid_true)
                valid_MSE = MSE(100 * val_o, 100 * valid_true)
            early_stop(util.to_numpy(val_loss), dynamic, optimizer, val_mse=util.to_numpy(valid_MSE))
            dynamic.train()

            if epoch % arguments.print_rate == 0:
                avg_r = epoch_r / max(n_batches, 1)
                avg_p = epoch_p / max(n_batches, 1)
                print('epoch:%-5d train_loss: %-12.3e smooth_loss: %-12.3e valid_MSE: %-9.3f lr: %.1e' %
                      (epoch, avg_r, avg_p, util.to_numpy(valid_MSE), optimizer.param_groups[0]['lr']))
            if early_stop.early_stop:
                break

    # Test & Visualization
    if arguments.predict:
        dynamic.load_state_dict(torch.load(f'./checkpoints/{arguments.model}.pth'))
        dynamic.eval()

        feature_names = ['Permanent Magnet (PM)', 'Stator Yoke (SY)', 'Stator Tooth (ST)', 'Stator Winding (SW)']

        with torch.no_grad():
            test_pred, _ = dynamic(test[:, 0:arguments.estimate_window, arguments.control:].permute(1, 0, 2),
                                   test[:, arguments.estimate_window:, 0:arguments.control].permute(1, 0, 2))
            test_loss = MSE(100 * test_pred, 100 * test_true)
            test_feature_mse = compute_feature_mse(test_pred, test_true)
            print('\nTest Loss (MSE): %.5f' % test_loss)
            print('Test MSE by Feature:')
            for i, mse in enumerate(test_feature_mse):
                name = feature_names[i] if i < len(feature_names) else f'Feature {i + 1}'
                print(f'  {name}: {mse.item():.5f}')

            train_pred, _ = dynamic(train[:, 0:arguments.estimate_window, arguments.control:].permute(1, 0, 2),
                                    train[:, arguments.estimate_window:, 0:arguments.control].permute(1, 0, 2))
            train_loss = MSE(100 * train_pred, 100 * train_true)
            train_feature_mse = compute_feature_mse(train_pred, train_true)
            print('Train Loss (MSE): %.5f\n' % train_loss)
            print('Train MSE by Feature:')
            for i, mse in enumerate(train_feature_mse):
                name = feature_names[i] if i < len(feature_names) else f'Feature {i + 1}'
                print(f'  {name}: {mse.item():.5f}')
            print('')

        print("Drawing and saving continuous temperature plots...")
        plot_results(train_pred, train_true, arguments.predict_window, arguments.features,
                     title=f'Continuous Temperature Estimation - TRAIN SET ({arguments.model.upper()} Model)',
                     save_path=f'./checkpoints/{arguments.model}_TRAIN_continuous_plot.png')
        plot_results(test_pred, test_true, arguments.predict_window, arguments.features,
                     title=f'Continuous Temperature Estimation - TEST SET ({arguments.model.upper()} Model)',
                     save_path=f'./checkpoints/{arguments.model}_TEST_continuous_plot.png')
                     
        print("Drawing and saving continuous error plots...")
        plot_errors(train_pred, train_true, arguments.predict_window, arguments.features,
                     title=f'Estimation Error - TRAIN SET ({arguments.model.upper()} Model)',
                     save_path=f'./checkpoints/{arguments.model}_TRAIN_error_plot.png')
        plot_errors(test_pred, test_true, arguments.predict_window, arguments.features,
                     title=f'Estimation Error - TEST SET ({arguments.model.upper()} Model)',
                     save_path=f'./checkpoints/{arguments.model}_TEST_error_plot.png')
                     
        print("All plots have been saved successfully!")


parser = argparse.ArgumentParser(description='Train configs (optimized)')
parser.add_argument('--seed', type=int, default=2026, help='Random seed')
parser.add_argument('--down_rate', type=int, default=8, help='Down sample rate')
parser.add_argument('--estimate_window', type=int, default=16, help='Seq_len for estimate the h0')
parser.add_argument('--predict_window', type=int, default=128, help='Seq_len for predict')
parser.add_argument('--train_samples', type=int, default=30000, help='Nums of the train samples')
parser.add_argument('--control', type=int, default=10, help='The control input dimension')
parser.add_argument('--print_rate', type=int, default=1, help='The rate of printing')
parser.add_argument('--features', type=int, default=4, help='The output dimension')
parser.add_argument('--hidden_size', type=int, default=48, help='Hidden_size of each hidden layer')
parser.add_argument('--batch', type=int, default=1024, help='Batch size for training process')
parser.add_argument('--epochs', type=int, default=5000, help='Max epoch for training process')
parser.add_argument('--n_layers', type=int, default=3, help='Nums of layers for MLP')
parser.add_argument('--model', type=str, default='perron', choices=['perron', 'linear'])
parser.add_argument('--train', type=bool, default=True, help='Train or just load the checkpoint for testing')
parser.add_argument('--predict', type=bool, default=True, help='Whether to evaluate on the test set?')
# 优化相关
parser.add_argument('--lr', type=float, default=1e-3, help='Learning rate')
parser.add_argument('--weight_decay', type=float, default=0.0, help='L2 weight decay for Adan (0=与原版一致)')
parser.add_argument('--max_grad_norm', type=float, default=0.0, help='Gradient clip norm (0=关闭，与原版一致)')
parser.add_argument('--p_loss_weight', type=float, default=0.1, help='Weight of physics-informed smoothness loss')
parser.add_argument('--patience', type=int, default=100, help='Early stopping patience (epochs)')
parser.add_argument('--save_by_mse', type=bool, default=False, help='False=按验证Smooth L1保存最佳(与原版一致)，True=按验证MSE')
parser.add_argument('--num_workers', type=int, default=0, help='DataLoader num_workers (0 for Windows stability)')
parser.add_argument('--perron_lambda_min', type=float, default=0.1, help='Perron state matrix min eigenvalue')
parser.add_argument('--perron_lambda_max', type=float, default=0.999, help='Perron state matrix max eigenvalue')
args = parser.parse_args()

Train(arguments=args)