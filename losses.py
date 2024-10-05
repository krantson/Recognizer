import torch.nn as nn
import torch

def BCEBitAvg(first_order_sta, normalize=True):
    if normalize:
        weight = (first_order_sta.sum(axis=0) / first_order_sta.shape[0]) * first_order_sta.shape[-1]
    else:
        weight = (first_order_sta.sum(axis=0) / first_order_sta.shape[0])
    return nn.BCELoss(weight=weight)

def BCEWrap():
    loss_f = nn.BCELoss()
    def f(predict, y_train, *in_arg):
        return loss_f(predict, y_train)
    return f

def CEWrap():
    loss_f = nn.CrossEntropyLoss()
    def f(predict, y_train, *in_arg):
        return loss_f(predict, y_train)
    return f

def BCEMix():
    loss_f = nn.BCELoss()
    def f(predict, y_train, *in_arg):
        y_index = torch.argmax(y_train, dim=-1)
        predict_value = predict[range(y_index.shape[0]), y_index]
        return loss_f(predict, y_train) + (1 - predict_value.sum() / len(predict_value))
    return f


def BCEPower(alpha=2):
    def f(predict, y_train, *in_arg):
        eps = torch.finfo(predict.dtype).eps
        y_index = torch.argmax(y_train, dim=-1)
        predict_value = predict[range(y_index.shape[0]), y_index]
        gamma = torch.pow(predict_value.shape[0]/(predict_value.sum(dim=-1)+eps), 1/alpha)
        return -(torch.log(torch.pow(predict, gamma)+eps)*y_train + torch.log(torch.pow(1-predict, gamma)+eps)*(1-y_train)).sum()/predict.numel()
    return f

def BCEPowerMinus(alpha=2, gamma=3):
    def f(predict, y_train, *in_arg):
        eps = torch.finfo(predict.dtype).eps
        y_index = torch.argmax(y_train, dim=-1)
        predict_value = predict[range(y_index.shape[0]), y_index]
        minus = (predict_value * alpha).unsqueeze(1).repeat(1, predict.shape[1])
        return -(y_train*torch.log(torch.pow(predict+eps, gamma-minus+eps)+eps) + (1-y_train)*torch.log(torch.pow(1-predict+eps, gamma-minus+eps)+eps)).sum()/predict.numel()
    return f


def BCEFocal(alpha=0.5, gamma=0, *out_arg):
    # eps = torch.tensor(1e-45)
    def f(predict, y_train, *in_arg):
        eps = torch.finfo(predict.dtype).eps
        weight = predict
        return -(torch.log(predict+eps)*y_train*torch.pow(1-weight+eps, gamma)*alpha + torch.log(1-predict+eps)*(1-y_train)*torch.pow(weight+eps, gamma)*(1-alpha)).sum()*2/predict.numel()
    return f


def MSEWrap(alpha=1):
    loss_f = nn.MSELoss()
    def f(predict, y_train, *in_arg):
        return loss_f(predict, y_train) * alpha
    return f

def MSEBCEFocal(alpha=0.5, gamma=0, *out_arg):
    loss_f = nn.MSELoss()
    def f(predict, y_train, *in_arg):
        eps = torch.finfo(predict.dtype).eps
        weight = predict
        return (alpha)*-(torch.log(predict+eps)*y_train*torch.pow(1-weight+eps, gamma) + torch.log(1-predict+eps)*(1-y_train)*torch.pow(weight+eps, gamma)).sum()/predict.numel() + (1-alpha)*loss_f(predict, y_train)
    return f

def BCEBalance(threshold=0.5, gamma=1, *out_arg):
    # eps = torch.tensor(1e-45)
    def f(predict, y_train, *in_arg):
        eps = torch.finfo(predict.dtype).eps
        y_index = torch.argmax(y_train, dim=-1)
        y_predict_mask = (predict[range(len(y_index)), y_index] < threshold).to(predict.dtype)
        y_predict_mask = y_predict_mask.unsqueeze(-1).repeat(1, y_train.shape[1])

        y_predict_mask_sum = y_predict_mask.sum(dim=-1, keepdims=True)
        zero_mask = ((y_predict_mask_sum == 0).repeat(1, y_train.shape[1]))
        y_predict_mask[zero_mask] = (1-predict[range(len(y_index)), y_index])[:,None].repeat(1, y_train.shape[1])[zero_mask]

        # if torch.all(y_predict_mask == 0):
        #     y_predict_mask = torch.full_like(predict, 0.1)
        return -((torch.log(predict+eps)*y_train*torch.pow(1-predict+eps, gamma) + torch.log(1-predict+eps)*(1-y_train)*torch.pow(predict+eps, gamma))*y_predict_mask).sum()/predict.numel()
    return f


def CEBalance(gamma=5, *out_arg):
    # eps = torch.tensor(1e-45)
    def f(predict, y_train, *in_arg):
        eps = torch.finfo(predict.dtype).eps
        y_index = torch.argmax(y_train, dim=-1)
        predict_value = predict[range(predict.shape[0]), y_index]
        max_two_value = torch.topk(predict, 2, dim=-1)[0].sum(dim=-1)
        weight = torch.exp(gamma*(max_two_value - 2*predict_value)) / (max_two_value + eps)
        return  -((((torch.log(predict+eps)*y_train) + torch.log(1-predict+eps)*(1-y_train)).sum(dim=-1))*weight).sum()/predict.numel()
    return f

def MarkovBCEFocal(alpha, gamma, markovstat):
    def f(predict, y_train, y_train_previous=None):
        eps = torch.finfo(predict.dtype).eps
        if y_train_previous is not None:
            y_previous_index = torch.argmax(y_train_previous, dim=-1)
            weight = markovstat[y_previous_index]*26
            return -((torch.log(predict+eps)*y_train*torch.pow(1-predict+eps, gamma) + torch.log(1-predict+eps)*(1-y_train)*torch.pow(predict+eps, gamma))*weight).sum()/predict.numel()
        else:
            return -(torch.log(predict+eps)*y_train*torch.pow(1-predict+eps, gamma)*alpha + torch.log(1-predict+eps)*(1-y_train)*torch.pow(predict+eps, gamma)*(1-alpha)).sum()*2/predict.numel()
    return f


def AdjBCEFocal(gamma, *out_arg):
    def f(predict, y_train, weights):
        eps = torch.finfo(predict.dtype).eps
        weights = weights * 26
        return -((torch.log(predict+eps)*y_train*torch.pow(1-predict+eps, gamma) + torch.log(1-predict+eps)*(1-y_train)*torch.pow(predict+eps, gamma))*weights).sum()/predict.numel()
    return f


def BCEBitAvgExp(first_order_sta, normalize=True, coefficient=4):
    avg_sta = first_order_sta.sum(axis=0) / first_order_sta.shape[0]
    def f(predict, y_train):
        eps = torch.finfo(predict.dtype).eps
        weight = torch.exp(coefficient*(1 - avg_sta))
        weight = weight / weight.sum(axis=-1, keepdims=True)
        weight = weight * y_train.shape[-1] if normalize else weight
        return -((torch.log(predict+eps)*y_train + torch.log(1-predict+eps)*(1-y_train))*weight).sum()/predict.numel()
    return f


def BCEBit(statistc_char, normalize=True):
    #  statistc_char: (26,)
    def f(predict, y_train):
        eps = torch.finfo(predict.dtype).eps
        weight = statistc_char * y_train.shape[-1] if normalize else statistc_char
        return -((torch.log(predict+eps)*y_train + torch.log(1-predict+eps)*(1-y_train))*weight).sum()/predict.numel()
    return f


def BCEBitExp(statistc_char, normalize=True, coefficient=4):
    #  statistc_char: (26,)
    def f(predict, y_train):
        eps = torch.finfo(predict.dtype).eps
        weight = torch.exp(coefficient*(1 - statistc_char))
        weight = weight / weight.sum(axis=-1, keepdims=True)
        weight = weight* y_train.shape[-1] if normalize else weight
        return -((torch.log(predict+eps)*y_train + torch.log(1-predict+eps)*(1-y_train))*weight).sum()/predict.numel()
    return f



# def BCEMarkov(statistc_char, transition_matrix, normalize=True, coefficient=4, using_score=False, regularization=False):
#     def f(predict, y_train, predict_previous=None, score=None):
#         eps = torch.finfo(predict.dtype).eps
#         predict_arg = torch.argmax(predict, dim=-1) # (64,)
#         if using_score:
#             weight = score
#         elif predict_previous is None:
#             weight = statistc_char[predict_arg]
#         else:
#             predict_previous_arg = torch.argmax(predict_previous, dim=-1) # (64,)
#             weight = transition_matrix[predict_previous_arg, predict_arg]
        
#         weight = weight / (torch.max(weight) + eps)
#         weight = torch.exp(-coefficient * weight)

#         if regularization:
#             return -(torch.log(predict+eps)*y_train + torch.log(1-predict+eps)*(1-y_train)).sum() + weight.sum()
#         else:
#             weight = weight / weight.sum(axis=-1, keepdims=True)
#             weight = weight * len(weight) if normalize else weight
#             return -((torch.log(predict+eps)*y_train + torch.log(1-predict+eps)*(1-y_train)).sum(axis=-1)*weight).sum()/predict.numel()
#     return f


def BCEMarkov(transition_matrix):
    loss_f = nn.BCELoss()
    def f(predict, y_train, *in_arg):
        predict = predict / predict.sum(axis=-1, keepdims=True)
        return loss_f(predict, y_train)
    return f



def BCEMarkovDecay(statistc_char, transition_matrix, normalize=True, coefficient=-9, epochs=40):
    def f(predict, y_train, predict_previous=None, epoch=30):
        eps = torch.finfo(predict.dtype).eps
        if predict_previous is None:
            weight = torch.exp(coefficient*(1 - statistc_char))
        else:
            predict_previous_arg = torch.argmax(predict_previous, dim=-1) # (64,)
            # coefficient_decay = coefficient * (1-2*(epoch/epochs))
            coefficient_decay = coefficient * torch.sign(torch.tensor(epochs/2 - epoch))
            weight = torch.exp(coefficient_decay*(1 - transition_matrix[predict_previous_arg]))
        weight = weight / weight.sum(axis=-1, keepdims=True)
        weight = weight * y_train.shape[-1] if normalize else weight
        return -((torch.log(predict+eps)*y_train + torch.log(1-predict+eps)*(1-y_train))*weight).sum()/predict.numel()
    return f


def BCEMarkovChar(statistc_char, transition_matrix, normalize=True, coefficient=4):
    def f(predict, y_train, predict_previous=None):
        eps = torch.finfo(predict.dtype).eps
        y_train_arg = torch.argmax(y_train, dim=-1, keepdims=True)
        if predict_previous is None:
            weight = torch.exp(coefficient*(1 - statistc_char[y_train_arg]))
        else:
            predict_previous_arg = torch.argmax(predict_previous, dim=-1, keepdims=True) # (64,)
            weight = torch.exp(coefficient*(1 - transition_matrix[predict_previous_arg, y_train_arg]))
        weight = weight / weight.sum(axis=-1, keepdims=True)
        weight = weight * y_train.shape[-1] if normalize else weight
        return -((torch.log(predict+eps)*y_train + torch.log(1-predict+eps)*(1-y_train))*weight).sum()/predict.numel()
    return f


def BCEChar(statistc_char, normalize=True):
    #  statistc_char: (26,)
    def f(predict, y_train):
        eps = torch.finfo(predict.dtype).eps
        # predict and y_train: (Batch, classes_26)
        y_train_arg = torch.argmax(y_train, dim=-1, keepdims=True)  #(64,1)
        weight = statistc_char[y_train_arg] * y_train.shape[-1] if normalize else statistc_char[y_train_arg]
        return -((torch.log(predict+eps)*y_train + torch.log(1-predict+eps)*(1-y_train))*weight).sum()/predict.numel()
    return f


def MarkovScore(transition_matrix, *pred_result):
    joint_score = 1
    for i in range(1, len(pred_result)):
        predict_previous_arg = torch.argmax(pred_result[i-1], dim=-1) # (64,)
        predict_arg = torch.argmax(pred_result[i], dim=-1) # (64,)
        joint_score *= transition_matrix[predict_previous_arg, predict_arg]
    return joint_score









