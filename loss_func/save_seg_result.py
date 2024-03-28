import pickle
from SegLossOdyssey.losses_pytorch.dice_loss import DC_and_topk_loss, DC_and_CE_loss

with open(r'C:\Git\NeuralNetwork\Pancreas_with_Monai_Lightning\loss_func\pickle_model_output\label2.pickle', 'rb') as file:
    # Load the data from the file
    label2 = pickle.load(file)

with open(r'C:\Git\NeuralNetwork\Pancreas_with_Monai_Lightning\loss_func\pickle_model_output\logit2.pickle', 'rb') as file:
    # Load the data from the file
    logit2 = pickle.load(file)

DiceTopKLoss = DC_and_topk_loss(aggregate='sum',
                                soft_dice_kwargs={
                                    'batch_dice': False,
                                    'do_bg': False,
                                    'smooth': 1e-5},
                                ce_kwargs={'k': 10}
                                )

DiceCELoss = DC_and_CE_loss(aggregate='sum',
                            soft_dice_kwargs={
                                'batch_dice': False,
                                'do_bg': False,
                                'smooth': 1e-5},
                            ce_kwargs={'reduction': 'mean'}
                            )

loss1 = DiceTopKLoss(logit2, label2)
loss2 = DiceCELoss(logit2, label2)



print(f"loss1:{loss1}")
print(f"loss2:{loss2}")
