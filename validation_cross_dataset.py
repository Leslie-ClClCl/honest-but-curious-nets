# 用于验证跨数据集时，原论文中方法的泛化性
# Author: Shining Sweety
# Create date: 2022/01/10 (YY/MM/DD)
# Modification date: 2022/01/11 (YY/MM/DD)
# Modification log:

import os
import numpy as np
import torch
from torch.utils.data import DataLoader, TensorDataset
from torchsummary import summary

from setting import args_parser
from hbcnets import datasets, models, utils, trainers, constants


def prepare_labels(train_labels, test_labels, valid_labels=None, dataset_name=None):
    if dataset_name == "celeba":
        curious = 20  # Male & Female
        train_labels[:, 0] = np.where(train_labels[:, 8] == 1, 0, (np.where(train_labels[:, 9] == 1, 1, 2)))
        train_labels[:, 20] = np.where(train_labels[:, 20] == 1, 0, 1)
        valid_labels[:, 0] = np.where(valid_labels[:, 8] == 1, 0, (np.where(valid_labels[:, 9] == 1, 1, 2)))
        valid_labels[:, 20] = np.where(valid_labels[:, 20] == 1, 0, 1)
        test_labels[:, 0] = np.where(test_labels[:, 8] == 1, 0, (np.where(test_labels[:, 9] == 1, 1, 2)))
        test_labels[:, 20] = np.where(test_labels[:, 20] == 1, 0, 1)
        train_labels = train_labels[:, [constants.HONEST, curious]]
        valid_labels = valid_labels[:, [constants.HONEST, curious]]
        test_labels = test_labels[:, [constants.HONEST, curious]]
        return train_labels, test_labels, valid_labels
        #####
    elif dataset_name == "utk_face":
        # Labels: age, gender, race
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
    """
    载入 utk face 和 celeba 数据集
    进行标签的预处理
    筛选出所需的标签
    """
    npz_cel_path = 'hbcnets/data/temp/CelebA_npy_64/celeba.npz'
    npz_utk_path = 'hbcnets/data/temp/UTKFace_npy_64/utk_face.npz'
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
    Train_y_cel, Test_y_cel, Valid_y_cel = prepare_labels(Train_y_cel, Test_y_cel, Valid_y_cel, dataset_name='celeba')

    # 挑选出需要的标签
    Train_y_utk = Train_y_utk[:, [constants.HONEST, constants.CURIOUS]].astype(int)
    Test_y_utk = Test_y_utk[:, [constants.HONEST, constants.CURIOUS]].astype(int)
    return Train_X_utk, Train_y_utk, Test_X_utk, Test_y_utk, Train_X_cel, Train_y_cel


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
    # 唯一标识一次实验的设置，方便重新读取
    exp_name = str(constants.HONEST) + "_" + str(constants.CURIOUS) + "_" + str(constants.K_Y) + \
               "_" + str(constants.K_S) + "_" + str(int(constants.BETA_X)) + "_" + str(int(constants.BETA_Y)) + \
               "_" + str(constants.SOFTMAX) + "/" + str(constants.IMAGE_SIZE) + "_" + str(constants.RANDOM_SEED)

    # 进行 Honest 推断的模型
    model = models.Classifier(num_classes=constants.K_Y, with_softmax=constants.SOFTMAX)
    model.to(args.device)
    summary(model, input_size=(3, constants.IMAGE_SIZE, constants.IMAGE_SIZE), device=args.device)

    if args.attack == "parameterized":
        # 结果的保存路径、模型的训练
        save_dir = args.root_dir + "/results_par/" + constants.DATASET + "/" + exp_name + "/"
        if not os.path.exists(save_dir):
            os.makedirs(save_dir)

        # 在测试集上测试准确率
        # 加载训练中表现最好的模型
        model.load_state_dict(torch.load(save_dir + "best_model.pt", map_location=torch.device(args.device)))
        param_G = models.Parameterized(num_inputs=constants.K_Y, num_classes=constants.K_S)
        param_G.to(args.device)
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
    elif args.attack == "regularized":
        save_dir = args.root_dir + "/results_reg/" + constants.DATASET + "/" + exp_name + "/"
        if not os.path.exists(save_dir):
            os.makedirs(save_dir)

        # Valid
        # 挑选一个更好的分割点
        best_REGTAU = 0.
        best_acc_valid = 0.
        model.load_state_dict(torch.load(save_dir + "best_model.pt", map_location=torch.device(args.device)))
        for REGTAU in np.arange(.04, .91, .02):
            valid_data_loader = trainers.get_data_loader(args, train_dataset, train=False)
            eval_acc_1, eval_acc_2 = utils.evaluate_acc_reg(args, model, valid_data_loader, beTau=REGTAU)
            if eval_acc_2 > best_acc_valid:
                best_REGTAU = REGTAU
                best_acc_valid = eval_acc_2
        print("\n$$$ Tau {} Valid Acc G , {:.2f}".format(best_REGTAU, best_acc_valid))
        # Test
        # 在测试集上测试准确率、AUC
        test_data_loader = DataLoader(TensorDataset(torch.Tensor(Test_X_cel), torch.Tensor(Test_y_cel).long()),
                                      batch_size=len(Test_X_cel) // 50, shuffle=False, drop_last=False)
        eval_acc_1, eval_acc_2, cf_mat_1, cf_mat_2 = utils.evaluate_acc_reg(args, model, test_data_loader, cf_mat=True,
                                                                            roc=True,
                                                                            beTau=best_REGTAU)
        print("\n$$$ Test Accuracy of the BEST model 1 {:.2f}".format(eval_acc_1))
        print("     Confusion Matrix 1:\n", (cf_mat_1 * 100).round(2))
        print("\n$$$ Test Accuracy of the BEST model 2 {:.2f}".format(eval_acc_2))
        print("     Confusion Matrix 2:\n", (cf_mat_2 * 100).round(2))