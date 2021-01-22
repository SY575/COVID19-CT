import os
import torch.utils.data
from torch.nn import DataParallel
from datetime import datetime
from torch.optim.lr_scheduler import MultiStepLR
from config import BATCH_SIZE, PROPOSAL_NUM, SAVE_FREQ, LR, WD, resume, save_dir
from core import model, dataset
from core.utils import init_log, progress_bar
import numpy as np
from sklearn.metrics import roc_auc_score
import pickle

os.environ['CUDA_VISIBLE_DEVICES'] = '7'
start_epoch = 1
save_dir = os.path.join(save_dir, datetime.now().strftime('%Y%m%d_%H%M%S'))
if os.path.exists(save_dir):
    raise NameError('model dir exists!')
os.makedirs(save_dir)
logging = init_log(save_dir)
_print = logging.info

# read dataset

train_path = '../input/train/'
val_path = '../input/val/'
test_path = '../input/test/'
trainset = dataset.SARS(root=train_path, is_train=True)
valset = dataset.SARS(root=val_path, is_train=False)
testset = dataset.SARS(root=test_path, is_train=False)
trainloader = torch.utils.data.DataLoader(trainset, batch_size=BATCH_SIZE,
                                          shuffle=True, num_workers=8, drop_last=False)
testloader = torch.utils.data.DataLoader(testset, batch_size=BATCH_SIZE,
                                         shuffle=False, num_workers=8, drop_last=False)
valloader = torch.utils.data.DataLoader(valset, batch_size=BATCH_SIZE,
                                         shuffle=False, num_workers=8, drop_last=False)

n_class = 2
# define model
net = model.attention_net(topN=PROPOSAL_NUM, n_class=n_class)
if resume:
    ckpt = torch.load(resume)
    net.load_state_dict(ckpt['net_state_dict'])
    start_epoch = ckpt['epoch'] + 1
creterion = torch.nn.CrossEntropyLoss()

# define optimizers
raw_parameters = list(net.pretrained_model.parameters())
part_parameters = list(net.proposal_net.parameters())
concat_parameters = list(net.concat_net.parameters())
partcls_parameters = list(net.partcls_net.parameters())

raw_optimizer = torch.optim.SGD(raw_parameters, lr=LR, momentum=0.9, weight_decay=WD)
concat_optimizer = torch.optim.SGD(concat_parameters, lr=LR, momentum=0.9, weight_decay=WD)
part_optimizer = torch.optim.SGD(part_parameters, lr=LR, momentum=0.9, weight_decay=WD)
partcls_optimizer = torch.optim.SGD(partcls_parameters, lr=LR, momentum=0.9, weight_decay=WD)

schedulers = [MultiStepLR(raw_optimizer, milestones=[60, 100], gamma=0.1),
              MultiStepLR(concat_optimizer, milestones=[60, 100], gamma=0.1),
              MultiStepLR(part_optimizer, milestones=[60, 100], gamma=0.1),
              MultiStepLR(partcls_optimizer, milestones=[60, 100], gamma=0.1)]

if resume:
    ckpt = torch.load(resume)
    net.pretrained_model.load_state_dict({layer.replace('pretrained_model.', ''):ckpt['net_state_dict'][layer] 
    for layer in ckpt['net_state_dict'] if 'pretrained_model' in layer})
    
    start_epoch = ckpt['epoch'] + 1
net = net.cuda()
net = DataParallel(net)

skip_epoch = 0

for epoch in range(start_epoch, 500):
    if epoch > skip_epoch:
        add = True
    else:
        add = False
    for scheduler in schedulers:
        scheduler.step()

    # begin training
    _print('--' * 50)
    net.train()
    train_correct = 0
    total = 0
    for i, data in enumerate(trainloader):
        img, label, img_raw = data[0].cuda(), data[1].cuda(), data[2]
        batch_size = img.size(0)
        raw_optimizer.zero_grad()
        part_optimizer.zero_grad()
        concat_optimizer.zero_grad()
        partcls_optimizer.zero_grad()
        raw_logits, concat_logits, part_logits, _, top_n_prob = net(img, img_raw, add)
        part_loss = model.list_loss(part_logits.view(batch_size * PROPOSAL_NUM, -1),
                                    label.unsqueeze(1).repeat(1, PROPOSAL_NUM).view(-1)).view(batch_size, PROPOSAL_NUM)
        raw_loss = creterion(raw_logits, label)
        concat_loss = creterion(concat_logits, label)
        rank_loss = model.ranking_loss(top_n_prob, part_loss)
        partcls_loss = creterion(part_logits.view(batch_size * PROPOSAL_NUM, -1),
                                 label.unsqueeze(1).repeat(1, PROPOSAL_NUM).view(-1))

        total_loss = raw_loss + rank_loss + concat_loss + partcls_loss
        total_loss.backward()
        raw_optimizer.step()
        part_optimizer.step()
        concat_optimizer.step()
        partcls_optimizer.step()
        progress_bar(i, len(trainloader), 'train')
        
        _, concat_predict = torch.max(concat_logits, 1)
        total += batch_size
        train_correct += torch.sum(concat_predict.data == label.data)
        
    print(float(train_correct) / total)
    pickle.dump(net, open('./model.pkl', 'wb'))
    if epoch % SAVE_FREQ == 0 :#and epoch > 20:
        train_loss = 0
        train_correct = 0
        total = 0
        net.eval()
        auc_label_lst = []
        auc_pred_lst = []
        people_lst = []
        file_name_lst = []
        for i, data in enumerate(valloader):
            with torch.no_grad():
                img, label, img_raw = data[0].cuda(), data[1].cuda(), data[2]
                batch_size = img.size(0)
                _, concat_logits, _, _, _, = net(img, img_raw, add)
                # calculate loss
                concat_loss = creterion(concat_logits, label)
                # calculate accuracy
                _, concat_predict = torch.max(concat_logits, 1)
                auc_label_lst += list(label.data.cpu().numpy())
                pred = torch.nn.Softmax(1)(concat_logits)
                auc_pred_lst.append(pred.data.cpu().numpy())
                people_lst.append(data[3])
                file_name_lst.append(data[4])
                
                total += batch_size
                train_correct += torch.sum(concat_predict.data == label.data)
                train_loss += concat_loss.item() * batch_size
                progress_bar(i, len(valloader), 'eval train set')
        train_acc = float(train_correct) / total
        train_loss = train_loss / total

        _print(
            'epoch:{} - train loss: {:.3f} and train acc: {:.3f} total sample: {}'.format(
                epoch,
                train_loss,
                train_acc,
                total))
        
        print(f'auc: {roc_auc_score(auc_label_lst, np.concatenate(auc_pred_lst, 0)[:, 1]):.4f}')
        np.save('./train_pred.npy', np.concatenate(auc_pred_lst, 0))
        np.save('./train_label.npy', np.array(auc_label_lst))
        np.save('./train_people.npy', np.concatenate(people_lst, 0))
        np.save('./train_file_name.npy', np.concatenate(file_name_lst, 0))
	# evaluate on test set
        test_loss = 0
        test_correct = 0
        total = 0
        auc_label_lst = []
        auc_pred_lst = []
        people_lst = []
        img_vis_lst = []
        file_name_lst = []
        anchor_lst = []
        for i, data in enumerate(testloader):
# =============================================================================
#             if i < 1:
#                 continue
# =============================================================================
            with torch.no_grad():
                img, label, img_raw = data[0].cuda(), data[1].cuda(), data[2]
                batch_size = img.size(0)
                _, concat_logits, _, _, _ = net(img, img_raw, add, False)
                # calculate loss
                concat_loss = creterion(concat_logits, label)
                # calculate accuracy
                _, concat_predict = torch.max(concat_logits, 1)
                auc_label_lst += list(label.data.cpu().numpy())
                pred = torch.nn.Softmax(1)(concat_logits)
                auc_pred_lst.append(pred.data.cpu().numpy())
                people_lst.append(data[3])
                file_name_lst += list(data[4])
# =============================================================================
#                 img_vis_lst.append(img_vis)
#                 anchor_lst.append(anchor)
# =============================================================================

                total += batch_size
                test_correct += torch.sum(concat_predict.data == label.data)
                test_loss += concat_loss.item() * batch_size
                progress_bar(i, len(testloader), 'eval test set')
        test_acc = float(test_correct) / total
        test_loss = test_loss / total
        _print(
            'epoch:{} - test loss: {:.3f} and test acc: {:.3f} total sample: {}'.format(
                epoch,
                test_loss,
                test_acc,
                total))
        
        
        print(f'auc: {roc_auc_score(auc_label_lst, np.concatenate(auc_pred_lst, 0)[:, 1]):.4f}')
        np.save('./test_pred.npy', np.concatenate(auc_pred_lst, 0))
        np.save('./test_label.npy', np.array(auc_label_lst))
        np.save('./test_people.npy', np.concatenate(people_lst, 0))
        np.save('./test_file_name.npy', np.array(file_name_lst))
        
# =============================================================================
#         np.save('./test_anchor_lst.npy', np.concatenate(anchor_lst, 0))
#         np.save('./test_vis.npy', np.concatenate(img_vis_lst, 0))
#         assert 0
# =============================================================================
	# save model
        net_state_dict = net.module.state_dict()
        if not os.path.exists(save_dir):
            os.mkdir(save_dir)
        torch.save({
            'epoch': epoch,
            'train_loss': train_loss,
            'train_acc': train_acc,
            'test_loss': test_loss,
            'test_acc': test_acc,
            'net_state_dict': net_state_dict},
            os.path.join(save_dir, '%03d.ckpt' % epoch))
# =============================================================================
#         assert 0
# =============================================================================
print('finishing training')
