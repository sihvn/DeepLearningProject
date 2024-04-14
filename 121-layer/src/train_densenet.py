import torch
import torch.nn as nn

from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.utils.tensorboard import SummaryWriter

import torch
torch.manual_seed(42)

## function to calculate the F1 score
def f1_score(tp, fp, fn):
    return 2 * (tp) / (2 * tp + fp + fn)

def train(model,
          train_loader,
          train_dataset_length,
          val_loader,
          num_class,
          device,
          model_name,
          lr
          ):
    # Define the loss function and optimizer
    criterion = nn.BCELoss(reduction='mean')
    optimizer = torch.optim.Adam(model.parameters(), lr = lr)
    scheduler = ReduceLROnPlateau(optimizer, factor = 0.1, patience = 5, mode = 'min')

    # Create a TensorBoard writer
    # model_name = "official_dnet_binary_by_img_count_lr_1e-4"
    model_name = model_name
    writer = SummaryWriter(log_dir=f".//runs//{model_name}_train")
    val_writer = SummaryWriter(log_dir=f".//runs//{model_name}_val")

    # Train the model
    n_epochs = 5
    bs = train_loader.batch_size
    conf_threshold = 1/num_class
    lossMIN = 100000
    for epoch in range(n_epochs):

        ## train
        model.train()
        for i, (images, labels) in enumerate(train_loader):
            images = images.to(device)
            labels = labels.unsqueeze(1).to(torch.float).to(device)
            ## NEW for v2
            # followup = followup.to(device)
            ## NEW for v2 end
            # labels = labels.to(device).unsqueeze(1).float()

            # Forward pass
            ## NEW for v2
            # outputs = model((images,followup))
            ## NEW for v2 end
            outputs = model(images)
            loss = criterion(outputs, labels)

            # Backprop
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            scheduler.step(loss)

            if (i + 1) % 20 == 1:
                # calculate statistics
                tp_array = [0]
                fp_array = [0]
                fn_array = [0]
                pred_labels = (outputs > conf_threshold)
                tp_array += sum(torch.logical_and(pred_labels, labels))
                fp_array += sum(torch.logical_and(torch.logical_xor(pred_labels, labels).long(), pred_labels))
                fn_array += sum(torch.logical_and(torch.logical_xor(pred_labels, labels).long(), labels))
                
                writer.add_scalar('Loss/img_count', loss, epoch * train_dataset_length + i * bs)
                writer.add_scalar('TP_Sum/img_count', sum(tp_array), epoch * train_dataset_length + i * bs)
                writer.add_scalar('FP_Sum/img_count', sum(fp_array), epoch * train_dataset_length + i * bs)
                writer.add_scalar('FN_Sum/img_count', sum(fn_array), epoch * train_dataset_length + i * bs)
                writer.add_scalar('F1_Score/img_count', f1_score(sum(tp_array), sum(fp_array), sum(fn_array)), epoch * train_dataset_length + i * bs)
                print("Epoch [{}/{}], Step [{}/{}], Loss: {:.4f}, tp_sum: {:.4f}, fp_sum: {:.4f}, fn_sum: {:.4f}, batch_f1_score: {:.4f}".format(epoch + 1, \
                                                                        n_epochs, \
                                                                        i + 1, \
                                                                        len(train_loader), \
                                                                        loss,\
                                                                        sum(tp_array), \
                                                                        sum(fp_array),\
                                                                        sum(fn_array),\
                                                                        f1_score(sum(tp_array), sum(fp_array), sum(fn_array))))
            # print("outputs\n", outputs)
            # print("pred_labels\n", pred_labels)
            # print("actual labels\n", labels)

            if loss < lossMIN:
                    lossMIN = loss    
                    torch.save({'epoch': epoch + 1, 'state_dict': model.state_dict(), 'best_loss': lossMIN, 'optimizer' : optimizer.state_dict()}, r'./dnet_models/m-' + model_name + '.pth.tar')

        ## val
        model.eval()
        ## calculation on the validation side of things
        tp_array = [0]
        fp_array = [0]
        fn_array = [0]

        for i, (images, labels) in enumerate(val_loader):
            images = images.to(device)
            labels = labels.to(device).unsqueeze(1).float()

            # Forward pass
            outputs = model(images)
            loss = criterion(outputs, labels)

            pred_labels = (outputs > conf_threshold)
            tp_array += sum(torch.logical_and(pred_labels, labels))
            fp_array += sum(torch.logical_and(torch.logical_xor(pred_labels, labels).long(), pred_labels))
            fn_array += sum(torch.logical_and(torch.logical_xor(pred_labels, labels).long(), labels))

        ## write to tensorboard    
        val_writer.add_scalar('Loss/img_count', loss, train_dataset_length * (epoch+1))
        val_writer.add_scalar('TP_Sum/img_count', sum(tp_array), train_dataset_length * (epoch+1))
        val_writer.add_scalar('FP_Sum/img_count', sum(fp_array), train_dataset_length * (epoch+1))
        val_writer.add_scalar('FN_Sum/img_count', sum(fn_array), train_dataset_length * (epoch+1))
        val_writer.add_scalar('F1_Score/img_count', f1_score(sum(tp_array), sum(fp_array), sum(fn_array)), train_dataset_length * (epoch+1))

