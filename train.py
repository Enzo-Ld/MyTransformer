# import torch
# from torch import nn, optim
# from model import Transformer
# from dataset import *
# import tqdm, os

# if __name__ == "__main__":
#     device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
#     tokenizer = AutoTokenizer.from_pretrained("./tokenizer")
#     tokenizer.add_special_tokens({"bos_token": "<s>"})
#
#     # 模型超参数
#     src_vocab_size = tokenizer.vocab_size + len(tokenizer.special_tokens_map)
#     dst_vocab_size = tokenizer.vocab_size + len(tokenizer.special_tokens_map)
#     pad_idx = tokenizer.pad_token_id
#     n_layers = 3
#     heads = 4
#     d_model = 256
#     d_ff = 512
#     dropout = 0.1
#     max_seq_len = 512
#     batch_size = 2
#     epochs = 100

#     # 初始化模型
#     model = Transformer(src_vocab_size, dst_vocab_size, pad_idx, n_layers, heads, d_model, d_ff, dropout, max_seq_len).to(device)

#     # 数据加载
#     train_dataset = EnglishChineseDataset(tokenizer, "./data/train.txt", max_seq_len)
#     test_dataset = EnglishChineseDataset(tokenizer, "./data/test.txt", max_seq_len)
#     train_loader = DataLoader(train_dataset, batch_size, shuffle=True)
#     test_loader = DataLoader(test_dataset, batch_size, shuffle=False)

#     # 优化器和损失函数
#     optimizer = optim.Adam(model.parameters(), lr=1e-4)
#     loss_fun = nn.CrossEntropyLoss(ignore_index=pad_idx)

#     # 初始化最佳模型指标
#     best_val_loss = float('inf')  # 初始设为无穷大
#     best_val_acc = 0.0            # 初始设为0
#     best_model_path = "best_model.pt"

#     with tqdm.tqdm(total=epochs) as t:
#         for epoch in range(epochs):
#             model.train()
#             train_loss_sum = 0.0
#             train_acc_sum = 0.0
#             train_samples = 0

#             # 训练阶段
#             for index, (en_in, de_in, de_label) in enumerate(train_loader):
#                 en_in, de_in, de_label = en_in.to(device), de_in.to(device), de_label.to(device)
#                 outputs = model(en_in, de_in)
#                 preds = torch.argmax(outputs, -1)
#                 label_mask = de_label != pad_idx

#                 correct = preds == de_label
#                 acc = torch.sum(label_mask * correct) / torch.sum(label_mask)

#                 outputs = outputs.view(-1, dst_vocab_size)
#                 d_label = de_label.view(-1)
#                 loss = loss_fun(outputs, d_label)

#                 optimizer.zero_grad()
#                 loss.backward()
#                 torch.nn.utils.clip_grad_norm_(model.parameters(), 1)
#                 optimizer.step()

#                 train_loss_sum += loss.item() * len(en_in)
#                 train_acc_sum += acc.item() * len(en_in)
#                 train_samples += len(en_in)

#                 if index % 100 == 0:
#                     print(f"Epoch: {epoch}, Batch: {index}/{len(train_loader)}, Train Loss: {loss.item():.4f}, Train Acc: {acc:.4f}")

#             # 计算训练集平均 loss 和 acc
#             avg_train_loss = train_loss_sum / train_samples
#             avg_train_acc = train_acc_sum / train_samples

#             # 验证阶段
#             model.eval()
#             val_loss_sum = 0.0
#             val_acc_sum = 0.0
#             val_samples = 0

#             with torch.no_grad():
#                 for index, (en_in, de_in, de_label) in enumerate(test_loader):
#                     en_in, de_in, de_label = en_in.to(device), de_in.to(device), de_label.to(device)
#                     outputs = model(en_in, de_in)
#                     preds = torch.argmax(outputs, -1)
#                     label_mask = de_label != pad_idx

#                     correct = preds == de_label
#                     acc = torch.sum(label_mask * correct) / torch.sum(label_mask)

#                     outputs = outputs.view(-1, dst_vocab_size)
#                     d_label = de_label.view(-1)
#                     loss = loss_fun(outputs, d_label)

#                     val_loss_sum += loss.item() * len(en_in)
#                     val_acc_sum += acc.item() * len(en_in)
#                     val_samples += len(en_in)

#             # 计算验证集平均 loss 和 acc
#             avg_val_loss = val_loss_sum / val_samples
#             avg_val_acc = val_acc_sum / val_samples

#             print(f"Epoch: {epoch}, Train Loss: {avg_train_loss:.4f}, Train Acc: {avg_train_acc:.4f}, Val Loss: {avg_val_loss:.4f}, Val Acc: {avg_val_acc:.4f}")

#             # 检查是否是最佳模型
#             if avg_val_loss < best_val_loss or avg_val_acc > best_val_acc:
#                 best_val_loss = min(avg_val_loss, best_val_loss)
#                 best_val_acc = max(avg_val_acc, best_val_acc)
#                 torch.save(model.state_dict(), best_model_path)
#                 print(f"New best model saved! Val Loss: {avg_val_loss:.4f}, Val Acc: {avg_val_acc:.4f}")

#             t.update(1)

#     print(f"Training finished! Best Val Loss: {best_val_loss:.4f}, Best Val Acc: {best_val_acc:.4f}")





import torch
from torch import nn,optim
from model import Transformer
from dataset import *
import tqdm,os

if __name__=="__main__":
    device=torch.device("cuda" if torch.cuda.is_available() else "cpu")
    tokenizer=AutoTokenizer.from_pretrained("./tokenizer")
    tokenizer.add_special_tokens({"bos_token":"<s>"})

    src_vocab_size=tokenizer.vocab_size+len(tokenizer.special_tokens_map)
    dst_vocab_size=tokenizer.vocab_size+len(tokenizer.special_tokens_map)


    pad_idx=tokenizer.pad_token_id
    n_layers=3
    heads=4
    d_model=128
    d_ff=512
    dropout=0.1
    max_seq_len=512
    batch_size=4
    epoches=50



    model=Transformer(src_vocab_size,dst_vocab_size,pad_idx,n_layers,heads,d_model,d_ff,dropout,max_seq_len).to(device)

    train_dataset=EnglishChineseDataset(tokenizer,"./data/train.txt",max_seq_len)
    test_dataset=EnglishChineseDataset(tokenizer,"./data/test.txt",max_seq_len)

    train_loader=DataLoader(train_dataset,batch_size,shuffle=True)
    test_loader=DataLoader(test_dataset,batch_size,shuffle=False)

    optimizer=optim.Adam(model.parameters(),lr=1e-4)
    loss_fun=nn.CrossEntropyLoss(ignore_index=pad_idx)

    with tqdm.tqdm(total=epoches) as t:
        for epoch in range(epoches): 
            model.train()
            for index,(en_in,de_in,de_label) in enumerate(train_loader):
                en_in,de_in,de_label=en_in.to(device),de_in.to(device),de_label.to(device)
                outputs=model(en_in,de_in)
                preds=torch.argmax(outputs,-1)
                label_mask=de_label!=pad_idx

                correct=preds==de_label
                acc=torch.sum(label_mask*correct)/torch.sum(label_mask)

                #batch,seq_len,dst_vocal_size
                outputs=outputs.view(-1,dst_vocab_size)
                d_label=de_label.view(-1)
                train_loss=loss_fun(outputs,d_label)

                optimizer.zero_grad()
                train_loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(),1)
                optimizer.step()

                if index%100==0:
                    print(f"epoch:{epoch}/{epoches},iter:{index}/{len(train_loader)},index:{index},train_loss:{train_loss.item()},acc:{acc}")




            torch.save(model.state_dict(),"model.pt")
            print("successfully save model")
            model.eval()
            with torch.no_grad():
                for index,(en_in,de_in,de_label) in enumerate(test_loader):
                    en_in,de_in,de_label=en_in.to(device),de_in.to(device),de_label.to(device)
                    outputs=model(en_in,de_in)
                    preds=torch.argmax(outputs,-1)
                    label_mask=de_label!=pad_idx

                    correct=preds==de_label
                    acc=torch.sum(label_mask*correct)/torch.sum(label_mask)

                    #batch,seq_len,dst_vocal_size
                    outputs=outputs.view(-1,dst_vocab_size)
                    d_label=de_label.view(-1)
                    test_loss=loss_fun(outputs,d_label)

                    if index%100==0:
                        print(f"iter:{index}/{len(test_loader)},index:{index},test_loss:{test_loss.item()},acc:{acc}")
