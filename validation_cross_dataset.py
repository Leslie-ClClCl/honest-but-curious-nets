# 用于验证跨数据集时，原论文中方法的泛化性
# Author: Shining Sweety
# Create date: 2022/01/10 (YY/MM/DD)
# Modification date: 2022/01/10 (YY/MM/DD)
# Modification log:

import os
import numpy as np
import torch
from torch.utils.data import DataLoader, TensorDataset
from torchsummary import summary

from hbcnets.data import utk_face
from setting import args_parser
from hbcnets import datasets, models, utils, trainers, constants
exp_name = 'exp_for_valid'


def prepare_labels(train_labels, test_labels, valid_labels=None, dataset_name=None):
    if dataset_name == "celeba":
        train_labels[:, 0] = np.where(train_labels[:, 8] == 1, 0, (np.where(train_labels[:, 9] == 1, 1, 2)))
        train_labels = train_labels[:, [constants.HONEST, constants.CURIOUS]]
        valid_labels[:, 0] = np.where(valid_labels[:, 8] == 1, 0, (np.where(valid_labels[:, 9] == 1, 1, 2)))
        valid_labels = valid_labels[:, [constants.HONEST, constants.CURIOUS]]
        test_labels[:, 0] = np.where(test_labels[:, 8] == 1, 0, (np.where(test_labels[:, 9] == 1, 1, 2)))
        test_labels = test_labels[:, [constants.HONEST, constants.CURIOUS]]
        return train_labels, valid_labels, test_labels
        #####
    elif dataset_name == "utk_face":
        ####  Labels: age, gender, race
        if (constants.HONEST == 0 and constants.K_Y == 2) or (constants.CURIOUS == 0 and constants.K_S == 2):
            train_labels[:, 0] = np.where(train_labels[:, 0] > 30, 1, 0)
            test_labels[:, 0] = np.where(test_labels[:, 0] > 30, 1, 0)
        elif (constants.HONEST == 0 and constants.K_Y == 3) or (constants.CURIOUS == 0 and constants.K_S == 3):
            train_labels[:, 0] = np.where(train_labels[:, 0] < 21, 2,
                                          (np.where(train_labels[:, 0] < 36, 0, 1)))
            test_labels[:, 0] = np.where(test_labels[:, 0] < 21, 2,
                                         (np.where(test_labels[:, 0] < 36, 0, 1)))
        elif (constants.HONEST == 0 and constants.K_Y == 4) or (constants.CURIOUS == 0 and constants.K_S == 4):
            train_labels[:, 0] = np.where(train_labels[:, 0] < 22, 0,
                                          (np.where(train_labels[:, 0] < 30, 1,
                                                    (np.where(train_labels[:, 0] < 46, 2, 3)))))
            test_labels[:, 0] = np.where(test_labels[:, 0] < 22, 0,
                                         (np.where(test_labels[:, 0] < 30, 1,
                                                   (np.where(test_labels[:, 0] < 46, 2, 3)))))
        elif (constants.HONEST == 0 and constants.K_Y == 5) or (constants.CURIOUS == 0 and constants.K_S == 5):
            train_labels[:, 0] = np.where(train_labels[:, 0] < 20, 0,
                                          (np.where(train_labels[:, 0] < 27, 1,
                                                    (np.where(train_labels[:, 0] < 35, 2,
                                                              (np.where(train_labels[:, 0] < 50, 3, 4)))))))
            test_labels[:, 0] = np.where(test_labels[:, 0] < 20, 0,
                                         (np.where(test_labels[:, 0] < 27, 1,
                                                   (np.where(test_labels[:, 0] < 35, 2,
                                                             (np.where(test_labels[:, 0] < 50, 3, 4)))))))
        else:
            if not (constants.HONEST == 1 or constants.CURIOUS == 1):
                raise ValueError("K_Y and K_S must be 2,3,4, or 5.")

        if (constants.HONEST == 2 and constants.K_Y == 2) or (constants.CURIOUS == 2 and constants.K_S == 2):
            train_labels[:, 2] = np.where(train_labels[:, 2] == 0, 0, 1)
            test_labels[:, 2] = np.where(test_labels[:, 2] == 0, 0, 1)
        elif (constants.HONEST == 2 and constants.K_Y == 3) or (constants.CURIOUS == 2 and constants.K_S == 3):
            train_labels[:, 2] = np.where(train_labels[:, 2] == 0, 0,
                                          (np.where(train_labels[:, 2] == 2, 2, 1)))
            test_labels[:, 2] = np.where(test_labels[:, 2] == 0, 0,
                                         (np.where(test_labels[:, 2] == 2, 2, 1)))
        elif (constants.HONEST == 2 and constants.K_Y == 4) or (constants.CURIOUS == 2 and constants.K_S == 4):
            train_labels[:, 2] = np.where(train_labels[:, 2] == 0, 0,
                                          (np.where(train_labels[:, 2] == 1, 1,
                                                    (np.where(train_labels[:, 2] == 3, 3, 2)))))
            test_labels[:, 2] = np.where(test_labels[:, 2] == 0, 0,
                                         (np.where(test_labels[:, 2] == 1, 1,
                                                   (np.where(test_labels[:, 2] == 3, 3, 2)))))
        elif (constants.HONEST == 2 and constants.K_Y == 5) or (constants.CURIOUS == 2 and constants.K_S == 5):
            pass
        else:
            if not (constants.HONEST == 1 or constants.CURIOUS == 1):
                raise ValueError("K_Y must be 2,3,4, or 5.")

        return train_labels, test_labels


def get_dataset_cross() -> None:
    npz_cel_path = './data/temp/CelebA_npy_64/celeba.npz'
    npz_utk_path = './data/temp/UTKFace_npy_64/utk_face.npz'
    # 加载 utk_face 数据集
    npz_utk = np.load(npz_utk_path)
    Train_X_utk = np.array(npz_utk['train_images'])
    Train_y_utk = np.array(npz_utk['train_labels'])
    Test_X_utk = np.array(npz_utk['test_images'])
    Test_y_utk = np.array(npz_utk['test_labels'])

    # 加载 celeba 数据集
    npz_cel = np.load(npz_cel_path)
    Train_X_cel = np.array(npz_cel['train_images'])
    Train_y_cel = np.array(npz_cel['train_labels'])
    Valid_X_cel = np.array(npz_cel['valid_images'])
    Valid_y_cel = np.array(npz_cel['valid_labels'])
    Test_X_cel = np.array(npz_cel['test_images'])
    Test_y_cel = np.array(npz_cel['test_labels'])

    # 标签的预处理
    Train_y_utk, Test_y_utk = prepare_labels(Train_y_utk, Test_y_utk, dataset_name='utk_face')
    Train_y_cel, Valid_y_cel, Test_y_cel = prepare_labels(Train_y_cel, Valid_y_cel, Test_y_cel, dataset_name='celeba')

    # 挑选出需要的标签
    Train_y_utk = Train_y_utk[:, [constants.HONEST, constants.CURIOUS]].astype(int)
    Test_y_utk = Test_y_utk[:, [constants.HONEST, constants.CURIOUS]].astype(int)
    Test_y_cel = Test_y_cel[:, [constants.HONEST, constants.CURIOUS]].astype(int)
    return Train_X_utk, Train_y_utk, Test_X_utk, Test_y_utk, Test_X_cel, Test_y_cel


if __name__ == '__main__':
    # 参数读取，参数在 ../setting.py 文件中
    args = args_parser()

    # 为程序分配显卡
    if args.gpu is not None:
        torch.cuda.set_device(int(args.gpu))

    # 读取数据集
    # 从 UTK_FACE 中读取 Age、Gender 属性分别作为 Honest 和 Curious 属性，用于训练阶段
    # 从 CelebA   中读取 Male 属性作为 Curious 属性，用于测试阶段
    # CelebA 中 Male属性是第 20 个类
    Train_X_utk, Train_y_utk, Test_X_utk, Test_y_utk, Test_X_cel, Test_y_cel = get_dataset_cross()
    train_dataset = (Train_X_utk, Train_y_utk)
    print('Data has been prepared!')

    # 进行 Honest 推断的模型
    model = models.Classifier(num_classes=constants.K_Y, with_softmax=constants.SOFTMAX)
    model.to(args.device)
    summary(model, input_size=(3, constants.IMAGE_SIZE, constants.IMAGE_SIZE), device=args.device)

    # 如果攻击的形式是 parameterized 的
    # 进行 Curious 推断的模型
    param_G = models.Parameterized(num_inputs=constants.K_Y, num_classes=constants.K_S)
    param_G.to(args.device)
    summary(param_G, input_size=(constants.K_Y,), device=args.device)

    # 结果的保存路径、模型的训练
    save_dir = args.root_dir + "/results_par/" + constants.DATASET + "/" + exp_name + "/"
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    model, param_G = trainers.train_model_par(args, model, param_G, train_dataset, save_dir)

    # 在测试集上测试准确率
    # 加载训练中表现最好的模型
    model.load_state_dict(torch.load(save_dir + "best_model.pt", map_location=torch.device(args.device)))
    param_G.load_state_dict(torch.load(save_dir + "best_param_G.pt", map_location=torch.device(args.device)))
    # 先在 utk_face 上进行测试
    test_loader_utk = DataLoader(TensorDataset(torch.Tensor(Test_X_utk), torch.Tensor(Test_y_utk).long()),
                                  batch_size=len(Train_X_utk) // 50, shuffle=False, drop_last=False)
    eval_acc_1, eval_acc_2, cf_mat_1, cf_mat_2 = utils.evaluate_acc_par(args, model, param_G, test_loader_utk,
                                                                        cf_mat=True, roc=True)
    print('On UTK_Face:')
    print(eval_acc_1)
    print(eval_acc_2)
    # 然后在 CelebA 上进行测试
    test_loader_cel = DataLoader(TensorDataset(torch.Tensor(Test_X_cel), torch.Tensor(Test_y_cel).long()),
                                 batch_size=len(Test_X_cel) // 50, shuffle=False, drop_last=False)
    eval_acc_1, eval_acc_2, cf_mat_1, cf_mat_2 = utils.evaluate_acc_par(args, model, param_G, test_loader_cel,
                                                                        cf_mat=True, roc=True)
    print('On CelebA:')
    print(eval_acc_1)
    print(eval_acc_2)
