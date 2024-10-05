import os
import numpy as np
import matplotlib.pyplot as plt
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torch
import collections
import itertools
from configparser import ConfigParser
from pathlib import Path
from PIL import Image
from models import Models
from dataset import data_generate, probs_analysis
from losses import BCEWrap, CEWrap, MarkovBCEFocal, BCEFocal, AdjBCEFocal, BCEPowerMinus, BCEPower, BCEBalance, CEBalance, MSEWrap, MSEBCEFocal, BCEMix
from utils import History, Accumulator, gen_model_configs, count, extract_model_para_list, \
    to_para_dict_list, paras_to_str, normalize, AccumulatorTensor


device = torch.device("cuda:0" if torch.cuda.is_available() else 'cpu')
cfg = ConfigParser()
cfg.read('config.cfg')
model_configs = dict(cfg.items('MODELS'))
para_configs = dict(cfg.items('PARAS'))
dataset_configs = dict(cfg.items('DATASET'))
paras_dict_list = to_para_dict_list(para_configs)
globals().update(dict(cfg.items('TRAIN')))



if os.path.exists(cloud_root):
    model_path = cloud_model_path
    print(f'Model files will be saved in cloud model path: {cloud_model_path}.')
else:
    print('Cloud model path does not exists!')
os.makedirs(model_path, exist_ok=True)



def _train_once(epoch_last, epochs, num_of_letters, model, optimizer, history, dl_train, dl_test, pkt_file, paras):
    
    if not hasattr(paras, "LOSS"):
        loss_func = [CEWrap()] * 4
    else:
        loss_paras = paras.LOSS.split('-')
        if loss_paras[0] == 'BCE':
            loss_func = [BCEWrap()] * 4
        elif loss_paras[0]  == 'CE':
            loss_func = [CEWrap()] * 4
        elif loss_paras[0]  == 'MSE':
            loss_func = [MSEWrap(alpha=float(loss_paras[1]))] * 4
        elif loss_paras[0] == 'BCEMix':
            loss_func = [BCEMix()] * 4
        elif loss_paras[0] == 'BCEPowerMinus':
            loss_func = [BCEPowerMinus(alpha=float(loss_paras[1]), gamma=float(loss_paras[2]))] * 4
        elif loss_paras[0] == 'BCEFocal':
            loss_func = [BCEFocal(alpha=float(loss_paras[1]), gamma=float(loss_paras[2]))] * 4
        elif loss_paras[0] == "MarkovBCEFocal":
            loss_func = [MarkovBCEFocal(alpha=float(loss_paras[1]), gamma=float(loss_paras[2]), markovstat=predicted_transition_matrix)] * 4

    loss_fn1, loss_fn2, loss_fn3, loss_fn4 = loss_func

    for epoch in range(epoch_last + 1, epochs + 1):
        # train
        model.train()
        train_loss_accu = Accumulator(num_of_letters)
        train_correct_accu = Accumulator(num_of_letters)
        train_errormap_accu = AccumulatorTensor(num_of_letters)

        # torch.autograd.set_detect_anomaly(True)
        for x_train, y_train in dl_train:
            x_train, y_train = x_train.to(device), y_train.to(device)
            optimizer.zero_grad()
            if not hasattr(paras, "WEIGHTS"):
                # pred1, pred2, pred3, pred4 = model(x_train)
                pred1, pred2, pred3, pred4, alpha_norm = model(x_train)
            else:
                pred1, pred2, pred3, pred4, weights = model(x_train)
                if 'Markov' in paras.WEIGHTS:
                    pred1 = pred1
                    prior2 = torch.mm(pred1, transition_matrix)
                    pred2 = pred2 + weights[:,0].unsqueeze(1).repeat(1, prior2.shape[1]) * prior2
                    pred2 = pred2 / pred2.sum(axis=-1, keepdims=True)
                    prior3 = torch.mm(prior2, transition_matrix)
                    pred3 = pred3 + weights[:,1].unsqueeze(1).repeat(1, prior3.shape[1]) * prior3
                    pred3 = pred3 / pred3.sum(axis=-1, keepdims=True)
                    prior4 = torch.mm(prior3, transition_matrix)
                    pred4 = pred4 + weights[:,2].unsqueeze(1).repeat(1, prior4.shape[1])  * prior4
                    pred4 = pred4 / pred4.sum(axis=-1, keepdims=True)
                else:
                    pred1 = pred2 = pred3 = pred4 = None

            if not hasattr(paras, "LOSS") or loss_paras[0] == 'CE':
                label1, label2, label3, label4 = map(lambda x: torch.argmax(x, dim=-1), [y_train[:,0,:], y_train[:,1,:], y_train[:,2,:], y_train[:,3,:]])
            elif loss_paras[0]  == 'BCE':
                label1, label2, label3, label4 = y_train[:,0,:], y_train[:,1,:], y_train[:,2,:], y_train[:,3,:]

            loss1 = loss_fn1(pred1, label1) 
            loss2 = loss_fn2(pred2, label2) 
            loss3 = loss_fn3(pred3, label3) 
            loss4 = loss_fn4(pred4, label4) 
            loss = (loss1 + loss2 + loss3 + loss4) / 4
            
            # with torch.autograd.detect_anomaly():
            loss.backward()
            optimizer.step()
            with torch.no_grad():
                c1,m1,c2,m2,c3,m3,c4,m4 = count([pred1, pred2, pred3, pred4], [label1, label2, label3, label4])
                train_loss_accu.add(loss1.item(), loss2.item(), loss3.item(), loss4.item())
                train_correct_accu.add(c1, c2, c3, c4)
                train_errormap_accu.add(m1, m2, m3, m4)

        history['train_loss'].add(train_loss_accu.to_dict(reduction_factor=len(dl_train)))
        history['train_accuracy'].add(train_correct_accu.to_dict(reduction_factor=len(dl_train.dataset)))
        history['train_errormap'].add(train_errormap_accu.to_dict())
        alpha_norm_detach = round(alpha_norm.detach().cpu().item(), 5) if alpha_norm else -1
        history['metrics'].add({'alpha': alpha_norm_detach})

        # test
        model.eval()
        test_loss_accu = Accumulator(num_of_letters)
        test_correct_accu = Accumulator(num_of_letters)
        test_errormap_accu = AccumulatorTensor(num_of_letters)

        for x_test, y_test in dl_test:
            with torch.no_grad():
                x_test, y_test = x_test.to(device), y_test.to(device)

                if not hasattr(paras, "WEIGHTS"):
                    # pred1, pred2, pred3, pred4 = model(x_test)
                    pred1, pred2, pred3, pred4, alpha_norm = model(x_test, state='val')
                else:
                    pred1, pred2, pred3, pred4, weights = model(x_test, state='val')
                    if 'Markov' in paras.WEIGHTS:
                        pred1 = pred1
                        pred2 = pred2 + weights[:,0].unsqueeze(1).repeat(1, pred1.shape[1]) * torch.mm(pred1, transition_matrix)
                        pred2 = pred2 / pred2.sum(axis=-1, keepdims=True)
                        pred3 = pred3 + weights[:,1].unsqueeze(1).repeat(1, pred2.shape[1]) * torch.mm(pred2, transition_matrix)
                        pred3 = pred3 / pred3.sum(axis=-1, keepdims=True)
                        pred4 = pred4 + weights[:,2].unsqueeze(1).repeat(1, pred3.shape[1]) * torch.mm(pred3, transition_matrix)
                        pred4 = pred4 / pred4.sum(axis=-1, keepdims=True)
                    else:
                        pred1 = pred2 = pred3 = pred4 = None
                
                if not hasattr(paras, "LOSS") or loss_paras[0] == 'CE':
                    label1, label2, label3, label4 = map(lambda x: torch.argmax(x, dim=-1), [y_test[:,0,:], y_test[:,1,:], y_test[:,2,:], y_test[:,3,:]])
                elif loss_paras[0]  == 'BCE':
                    label1, label2, label3, label4 = y_test[:,0,:], y_test[:,1,:], y_test[:,2,:], y_test[:,3,:]

                loss1 = loss_fn1(pred1, label1) 
                loss2 = loss_fn2(pred2, label2) 
                loss3 = loss_fn3(pred3, label3) 
                loss4 = loss_fn4(pred4, label4) 

                c1,m1,c2,m2,c3,m3,c4,m4 = count([pred1, pred2, pred3, pred4], [label1, label2, label3, label4])
                test_loss_accu.add(loss1.item(), loss2.item(), loss3.item(), loss4.item())
                test_correct_accu.add(c1, c2, c3, c4)
                test_errormap_accu.add(m1, m2, m3, m4)

        history['test_loss'].add(test_loss_accu.to_dict(reduction_factor=len(dl_test)))
        history['test_accuracy'].add(test_correct_accu.to_dict(reduction_factor=len(dl_test.dataset)))
        history['test_errormap'].add(test_errormap_accu.to_dict())

        print(f'epoch {epoch}:', 
                history['train_loss'].state_dict(avg=True)['average'][epoch-1],
                history['test_loss'].state_dict(avg=True)['average'][epoch-1],
                history['train_accuracy'].state_dict(avg=True)['average'][epoch-1],
                history['test_accuracy'].state_dict(avg=True)['average'][epoch-1],
                alpha_norm_detach
              )

        if epoch % int(epochs_save) == 0:
            checkpoint = {
                "epoch": epoch,
                "model_state": model.state_dict(),
                "optim_state": optimizer.state_dict(),
                'train_loss': history['train_loss'].state_dict(),
                'test_loss': history['test_loss'].state_dict(),
                'train_accuracy': history['train_accuracy'].state_dict(),
                'test_accuracy': history['test_accuracy'].state_dict(),
                'train_errormap': history['train_errormap'].state_dict(),
                'test_errormap': history['test_errormap'].state_dict(),
                'metrics': history['metrics'].state_dict()
            }
            torch.save(checkpoint, pkt_file)


def train(epochs, num_of_letters, model_path, model_configs, paras_dict_list):
    data_source = f"{dataset_configs['data_path']}"
    data_source = data_source.replace("\\", "/")
    if os.path.exists(cloud_root):
        data_source = os.path.join(cloud_root, os.path.split(data_source)[-1])  # 这里有坑，对于云平台，前面多加一个data
    data_path_list = [os.path.join(data_source, d) for d in os.listdir(data_source) if os.path.isdir(os.path.join(data_source, d))]

    if not data_path_list or os.path.join(data_source, 'train') in data_path_list:
        data_path_list = [data_source]
    elif any(['base_aug' in path.lower() for path in data_path_list]):
        assert len(data_path_list) > 2 and any(['val' in path.lower() for path in data_path_list])
        base = data_path_list[['base_aug' in path.lower() for path in data_path_list].index(True)]
        val = data_path_list[['val' in path.lower() for path in data_path_list].index(True)]
        data_path_list_all = [[base, val]] + list(filter(lambda x: (base in x) and (val in x), list(itertools.combinations(data_path_list, 3))))
        if len(data_path_list) > 3:
            data_path_list_all.append(data_path_list)
        data_path_list = data_path_list_all

    for data_path in data_path_list:
        if type(data_path) == str:
            dataset_name = os.path.basename(data_path)
        else:
            dataset_name = ['Base'] + sorted([os.path.basename(p) for p in data_path if 'base_aug' not in p.lower() and 'val' not in p.lower()])
            dataset_name = '_'.join(dataset_name)

        captcha_dl_train, captcha_dl_test = data_generate(
            int(batch_size), 
            int(num_of_letters), 
            float(dataset_split), 
            data_path, 
            charsets, 
            int(img_rows), 
            int(img_cols), 
            img_format,
            colormode='rgb'
        ) 
        model_args = gen_model_configs(model_configs)
        for model_name, model_config in model_args.items():
            paras_list = []
            model_para_dict_list = extract_model_para_list(model_config, paras_dict_list)
            para_tuple = collections.namedtuple('P', model_para_dict_list.keys())
            if not model_para_dict_list:
                paras_list.append(None)
            else:
                for p in itertools.product(*[plist for plist in model_para_dict_list.values()]):
                    paras_list.append(para_tuple(*p))
            for paras in paras_list:
                # model = Model(PARAS=paras, **model_config).to(device)
                model_base_name = model_name.split('_')[0]
                print(model_base_name)
                model = Models[model_base_name.upper()](PARAS=paras, **model_config).to(device) 
                optimizer = optim.Adam(model.parameters(), lr=0.0001)
                # loss_fn = loss_fns[paras.DEEPCAPTCHA]
                # loss_fn.loss_type_name = paras.DEEPCAPTCHA

                history_dict = {
                    'train_loss': History([i for i in range(num_of_letters)]),
                    'test_loss': History([i for i in range(num_of_letters)]),
                    'train_accuracy': History([i for i in range(num_of_letters)]),
                    'test_accuracy': History([i for i in range(num_of_letters)]),
                    'train_errormap': History([i for i in range(num_of_letters)]),
                    'test_errormap': History([i for i in range(num_of_letters)]),
                    'metrics':History(['alpha'])
                }
                epoch_last = 0
                pkt_file = os.path.join(model_path, f'{model_name}@{paras_to_str(paras)}#{dataset_name}.pth')
                if os.path.exists(pkt_file):
                    checkpoint = torch.load(pkt_file, map_location=torch.device(device))
                    epoch_last = checkpoint['epoch']
                    model.load_state_dict(checkpoint['model_state'])
                    try:
                        optimizer.load_state_dict(checkpoint['optim_state'])
                    except:
                        optimizer = optim.Adam(model.parameters(), lr=0.0001)
                    history_dict['train_loss'].load_state_dict(checkpoint['train_loss'])
                    history_dict['test_loss'].load_state_dict(checkpoint['test_loss'])
                    history_dict['train_accuracy'].load_state_dict(checkpoint['train_accuracy'])
                    history_dict['test_accuracy'].load_state_dict(checkpoint['test_accuracy'])
                    history_dict['train_errormap'].load_state_dict(checkpoint['train_errormap'])
                    history_dict['test_errormap'].load_state_dict(checkpoint['test_errormap'])
                    history_dict['metrics'].load_state_dict(checkpoint['metrics'])
                    print(f"###### {model_name}@{paras_to_str(paras)}#{dataset_name} load successful at epoch={epoch_last}. ######")
                print("Begin training:", f'{model_name}@{paras_to_str(paras)}#{dataset_name}')
                _train_once(epoch_last, epochs, num_of_letters, model, optimizer, history_dict, captcha_dl_train, captcha_dl_test, pkt_file, paras)
                print(f'{model_name}@{paras_to_str(paras)}#{dataset_name}', " completed!")


if __name__ == '__main__':
    train(int(epochs), int(num_of_letters), model_path, model_configs, paras_dict_list)



    
