import torch
import numpy as np

def temporal_ce_loss(output, target):

    seq_len = output.shape[1]
    target = target.view(target.shape[0], 1).contiguous()
    target = target.repeat(1, seq_len)

    # Using cross_entropy loss
    output=torch.log(output)

    # flatten all the labels and mask
    target = target.view(-1)

    # flatten all predictions
    no_classes=output.shape[-1]
    output = output.view(-1,no_classes)

#     print(mask.shape, output.shape,target.shape)
#     print( output[:, target].shape)
    # count how many frames we have
    # nb_frames = seq_len

    # pick the values for the label and zero out the rest with the mask
#     output = output[:, target] * mask
    y=target.view(-1,1).long()
#     print(y.shape,torch.gather(output, 1, y).shape)
    output=torch.squeeze(torch.gather(output, 1, y))
    # output = torch.gather(output, 1, y)* mask

    # compute cross entropy loss which ignores all elem where mask =0
    ce_loss = -torch.sum(output) / output.shape[0]

    return ce_loss


def vote_video_classification_result(output, y):
    f = output.detach().cpu().numpy()
    b = y.detach().cpu().numpy()
    num_samples = f.shape[0]
    num_classes = f.shape[-1]
    ix = np.zeros((num_samples,), dtype='int')
    ix_top3 = np.zeros((num_samples, 3), dtype='int')

 # for each example, we only consider argmax of the seq len
    votes = np.zeros((f.shape[-1],), dtype='int')
    for i, eg in enumerate(f):
        predictions = np.argmax(eg, axis=-1)
#         print(predictions.shape)
        for cls in range(f.shape[-1]):
            count = (predictions == cls).sum(axis=-1)
            votes[cls] = count
        ix[i] = np.argmax(votes)
        ix_top3[i] = torch.topk(torch.from_numpy(votes),3)[1].numpy()

    return torch.from_numpy(ix), torch.from_numpy(ix_top3)