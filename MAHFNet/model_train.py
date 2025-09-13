import torch
import torch.nn as nn
from MAHFNet import MAHFNet  # BMT=BIT
from block import *
import torch.optim as optim
import numpy as np
import clip
from opts import AverageMeter
import random
from sklearn.metrics import f1_score 
import logging
from datetime import datetime
from Dataset_MVSA_single import get_dataset as MVSA_S
from Dataset_TumEmo import get_dataset as TumEmo
from Dataset_HFM import get_dataset as HFM
def setup_logging():
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(f"training_{datetime.now().strftime('%Y%m%d_%H%M')}.log"),
            logging.StreamHandler()
        ]
    )
def setup_seed(seed):
    """固定所有随机种子以保证可复现性"""
    # Python 内置随机数生成器
    random.seed(seed)
    
    # NumPy 随机数生成器
    np.random.seed(seed)
    
    # PyTorch 相关设置
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)  # 多GPU时使用

    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

device = 'cuda' if torch.cuda.is_available() else 'cpu'  # cuda

dataset = 'HFM' # ''' TumEmo or MVSA_S

if dataset == 'TumEmo':
    batch_size = 64 # 32
    cls_num = 7
    epoch = 20  #
    train_data,valid_data,test_data = TumEmo(batch_size=batch_size)

if dataset == 'MVSA_S':
    batch_size = 64
    cls_num = 3
    epoch = 40  
    train_data,valid_data,test_data = MVSA_S(batch_size=batch_size)

if dataset == 'HFM':
    batch_size = 64 # 32 原本32
    cls_num = 2
    epoch = 40  #
    train_data,valid_data, test_data = HFM(batch_size=batch_size)



clip_pth = 'ViT-B/16'
if clip_pth in ['ViT-B/16', 'ViT-B/32']:
    dim = 512
elif clip_pth in ['ViT-L/14']:
    dim = 768

model, _ = clip.load(clip_pth)
clip_model = model.cuda()
for param in clip_model.parameters():
    param.requires_grad_(False)

net = MAHFNet(dim=dim).cuda()

text_tokenizer = lambda texts: clip.tokenize(texts, truncate=True).cuda()
XE_loss = nn.CrossEntropyLoss()
optimizer = optim.Adam(net.parameters(), lr=0.0001)
lr_scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=1, gamma=0.8)

def train_(epoch, total_epoch):
    # 每次输入barch_idx个数据
    net.train()
    tasks_top1 = AverageMeter()
    tasks_losses = AverageMeter()
    for batch_idx, data in enumerate(train_data):
        image,  text, translation, emo_label, = data[0], data[1], data[2],data[3]

        image,  emo_label = image.to(device), emo_label.to(device)

        clip_model.eval()
        text_tokens = text_tokenizer (text)
        translation_tokens = text_tokenizer (translation)
        text_f = clip_model.get_text_feature(text_tokens)
        text_f = torch.as_tensor(text_f, dtype=torch.float32)
        text_tran = clip_model.get_text_feature(translation_tokens)
        text_tran = torch.as_tensor(text_tran, dtype=torch.float32)
        
        image_f = clip_model.get_image_feature(image)
        image_f = torch.as_tensor(image_f, dtype=torch.float32)

        output,moe = net(image_f, text_f, text_tran)
        loss = XE_loss(output, emo_label.long())+moe
        emo_res = output.max(1)[1]  
        cor = emo_res.eq(emo_label).sum().item()
        tasks_top1.update(cor * 100 / (emo_label.size(0) + 0.0), emo_label.size(0))
        tasks_losses.update(loss.item(), emo_label.size(0))
        if batch_idx % 100 == 0:
            logging.info(
                "train:Epoch [%d/%d], Step [%d/%d], Loss: %.4f, Acc Val: %.4f, Acc Avg: %.4f",
                epoch + 1, total_epoch, batch_idx + 1, len(train_data),
                tasks_losses.val, tasks_top1.val, tasks_top1.avg
            )
        optimizer.zero_grad()
        loss.backward()
        # update
        optimizer.step()

    logging.info(
        "train:Epoch [%d/%d], Loss Avg: %.4f, Acc Avg: %.4f",
        epoch + 1, total_epoch, tasks_losses.avg, tasks_top1.avg
    )
    return tasks_top1.avg


def valid_(epoch, total_epoch):
    net.eval()
    global best_acc
    tasks_top1 = AverageMeter()
    tasks_losses = AverageMeter()
    prediction = []
    truth = []
    
    with torch.no_grad():
        for batch_idx, data in enumerate(valid_data):
            image,  text, translation, emo_label, = data[0], data[1], data[2],data[3]

            image,  emo_label = image.to(device), emo_label.to(device)

            clip_model.eval()
            text_tokens = text_tokenizer (text)
            translation_tokens = text_tokenizer (translation)
            text_f = clip_model.get_text_feature(text_tokens)
            text_f = torch.as_tensor(text_f, dtype=torch.float32)
            text_tran = clip_model.get_text_feature(translation_tokens)
            text_tran = torch.as_tensor(text_tran, dtype=torch.float32)
            
            image_f = clip_model.get_image_feature(image)
            image_f = torch.as_tensor(image_f, dtype=torch.float32)
            
            # 模型推理
            output,moe = net(image_f, text_f, text_tran)
            loss = XE_loss(output, emo_label.long()) + moe
            
            # 结果收集
            emo_res = output.argmax(dim=1)
            prediction.append(emo_res.cpu().numpy())
            truth.append(emo_label.cpu().numpy())

            # 指标计算
            cor = emo_res.eq(emo_label).sum().item()
            tasks_top1.update(cor * 100 / emo_label.size(0), emo_label.size(0))
            tasks_losses.update(loss.item(), emo_label.size(0))

            # 进度输出
            if batch_idx % 100 == 0:
                logging.info(
                    "train:Epoch [%d/%d], Step [%d/%d], Loss: %.4f, Acc Val: %.4f, Acc Avg: %.4f",
                    epoch + 1, total_epoch, batch_idx + 1, len(valid_data),
                    tasks_losses.val, tasks_top1.val, tasks_top1.avg
            )

    # 合并所有结果
    prediction = np.concatenate(prediction)
    truth = np.concatenate(truth)
    
    # 计算最终指标
    final_acc = tasks_top1.avg
    final_f1 = f1_score(truth, prediction, average='weighted') * 100  # macro平均
    
    # 输出最终结果
    logging.info("\nvalid Result Epoch [%d/%d]:", epoch+1, total_epoch)
    logging.info("Loss: %.4f | Acc: %.2f%% | F1: %.2f%%", 
                tasks_losses.avg, final_acc, final_f1)
    logging.info("-" * 60)  # 添加分隔线

    # 模型保存逻辑
    if final_acc > best_acc:
        best_acc = final_acc
        # 保存模型
        torch.save({
            'net': net.state_dict(),
            'acc': final_acc,
            'f1': final_f1,
            'epoch': epoch
        }, f'./{dataset}_test={memory}/best_model.pth')
        print(f"Model saved with Acc {final_acc:.2f}% and F1 {final_f1:.2f}%")

def test_():
    checkpoint = torch.load(f'./{dataset}_test={memory}/best_model.pth')
    net.load_state_dict(checkpoint['net'])
    net.eval()
    net.eval()
    tasks_top1 = AverageMeter()
    tasks_losses = AverageMeter()
    prediction = []
    truth = []
    
    with torch.no_grad():
        for batch_idx, data in enumerate(test_data):
            image,  text, translation, emo_label, = data[0], data[1], data[2],data[3]

            image,  emo_label = image.to(device), emo_label.to(device)

            clip_model.eval()
            text_tokens = text_tokenizer (text)
            translation_tokens = text_tokenizer (translation)
            text_f = clip_model.get_text_feature(text_tokens)
            text_f = torch.as_tensor(text_f, dtype=torch.float32)
            text_tran = clip_model.get_text_feature(translation_tokens)
            text_tran = torch.as_tensor(text_tran, dtype=torch.float32)
            
            image_f = clip_model.get_image_feature(image)
            image_f = torch.as_tensor(image_f, dtype=torch.float32)
            
            # 模型推理
            output,moe = net(image_f, text_f, text_tran)
            loss = XE_loss(output, emo_label.long()) + moe
            
            # 收集结果
            emo_res = output.argmax(dim=1)
            prediction.append(emo_res.cpu().numpy())
            truth.append(emo_label.cpu().numpy())
            
            # 计算指标
            cor = emo_res.eq(emo_label).sum().item()
            tasks_top1.update(cor * 100 / emo_label.size(0), emo_label.size(0))
            tasks_losses.update(loss.item(), emo_label.size(0))
            
            # 打印进度

    
    # 合并结果
    prediction = np.concatenate(prediction)
    truth = np.concatenate(truth)
    
    np.savez(
    f'./{dataset}_test={memory}/FINAL_TEST.npz',
    pred=prediction,
    labels=truth
)

    # 计算最终指标
    final_acc = tasks_top1.avg
    final_f1 = f1_score(truth, prediction, average='weighted') * 100
    logging.info('test result')
    logging.info("Loss: %.4f | Acc: %.2f%% | F1: %.2f%%", 
                tasks_losses.avg, final_acc, final_f1)
    print(f"\n{'=' * 30} Test Results {'=' * 30}")
    print(f"* Accuracy: {final_acc:.2f}%")
    print(f"* F1 Score: {final_f1:.4f}")
    print('=' * 76)
    
setup_seed(50)
setup_logging()
for epoch_ in range(epoch):
    train_acc = train_(epoch_, epoch)
    valid_(epoch_, epoch, train_acc)
    lr_scheduler.step()
test_()