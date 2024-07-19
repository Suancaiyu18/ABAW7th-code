import numpy as np
import torch
import torch.nn.functional as F
from .loss import loss_function

dtype = torch.cuda.FloatTensor() if torch.cuda.is_available() else torch.FloatTensor()
dtypel = torch.cuda.LongTensor() if torch.cuda.is_available() else torch.LongTensor()

def train(train_loader, model, optimizer, dinov2_model):
    model.train()
    au_loss_record, va_loss_record, expr_loss_record, loss_record = 0, 0, 0 , 0
    for i, (image , label) in enumerate(train_loader):
        label = torch.stack(label)
        optimizer.zero_grad()
        image = image.type_as(dtype)

        with torch.no_grad():
            train_feat = dinov2_model.forward_features(image)
            train_output = train_feat['x_norm_patchtokens']

        output = model(train_output)
        au_loss, va_loss, expr_loss = loss_function(output, label)
        loss = au_loss + va_loss + expr_loss

        loss.backward()
        optimizer.step()

        au_loss_record += au_loss.item()
        va_loss_record += va_loss.item()
        expr_loss_record += expr_loss.item()
        loss_record += loss.item()

    return loss_record / len(train_loader), au_loss_record / len(train_loader), \
            va_loss_record / len(train_loader), expr_loss_record / len(train_loader)

def evaluation(val_loader, model, dinov2_model):
    model.eval()
    output_list = []
    with torch.no_grad():
        for image, label, video_name in val_loader:
            video_name = np.expand_dims(np.array(video_name), axis=1)
            image = image.type_as(dtype).squeeze(1)

            with torch.no_grad():
                train_feat = dinov2_model.forward_features(image)
                train_output = train_feat['x_norm_patchtokens']

            output = model(train_output)
            au_results = output['AU'].detach().cpu()
            va_results = output['VA'].detach().cpu()
            expr_results = output['EXPR'].detach().cpu()
            au, expr, va = results_process(au_results, va_results, expr_results)
            all_results = np.concatenate([video_name, va, np.expand_dims(expr, axis=1), au], axis=1)
            output_list.append(all_results)

    return output_list

def results_process(au_results, va_results, expr_results):
    au = (torch.sigmoid(au_results) > 0.5).type(torch.LongTensor).numpy()
    expr = F.softmax(expr_results, dim=-1).argmax(-1).type(torch.LongTensor).numpy()
    va = va_results.numpy()
    return au, expr, va
