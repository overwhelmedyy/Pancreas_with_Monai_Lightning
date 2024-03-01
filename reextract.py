import re
import statistics
text = '''Dice Metric : 0.6931076645851135
Best valid Loss : 0.6931076645851135
EPOCH[VALID]1/1: 100%|██████████| 50/50 [02:41<00:00,  3.24s/it]
Dice Metric : 0.7452266812324524
Best valid Loss : 0.7452266812324524
EPOCH[VALID]1/1: 100%|██████████| 50/50 [02:39<00:00,  3.19s/it]
Dice Metric : 0.7756366729736328
Best valid Loss : 0.7756366729736328
EPOCH[VALID]1/1: 100%|██████████| 50/50 [02:40<00:00,  3.21s/it]
Dice Metric : 0.7352340817451477
Best valid Loss : 0.7352340817451477
EPOCH[VALID]1/1: 100%|██████████| 50/50 [02:39<00:00,  3.20s/it]
Dice Metric : 0.7509235739707947
Best valid Loss : 0.7509235739707947
EPOCH[VALID]1/1: 100%|██████████| 50/50 [02:40<00:00,  3.21s/it]
Dice Metric : 0.7778388857841492
Best valid Loss : 0.7778388857841492
EPOCH[VALID]1/1: 100%|██████████| 50/50 [02:40<00:00,  3.20s/it]
Dice Metric : 0.7643113732337952
Best valid Loss : 0.7643113732337952
EPOCH[VALID]1/1: 100%|██████████| 50/50 [02:40<00:00,  3.21s/it]
Dice Metric : 0.7797756791114807
Best valid Loss : 0.7797756791114807
EPOCH[VALID]1/1: 100%|██████████| 50/50 [02:36<00:00,  3.14s/it]
Dice Metric : 0.740034282207489
Best valid Loss : 0.740034282207489
EPOCH[VALID]1/1: 100%|██████████| 50/50 [02:38<00:00,  3.17s/it]
Dice Metric : 0.748847484588623
Best valid Loss : 0.748847484588623
EPOCH[VALID]1/1: 100%|██████████| 50/50 [02:39<00:00,  3.20s/it]
Dice Metric : 0.7699909210205078
Best valid Loss : 0.7699909210205078
EPOCH[VALID]1/1: 100%|██████████| 50/50 [02:39<00:00,  3.19s/it]
Dice Metric : 0.7308322787284851
Best valid Loss : 0.7308322787284851
EPOCH[VALID]1/1: 100%|██████████| 50/50 [02:41<00:00,  3.23s/it]
Dice Metric : 0.7695111036300659
Best valid Loss : 0.7695111036300659
EPOCH[VALID]1/1: 100%|██████████| 50/50 [02:39<00:00,  3.20s/it]
Dice Metric : 0.7479560375213623
Best valid Loss : 0.7479560375213623
EPOCH[VALID]1/1: 100%|██████████| 50/50 [02:40<00:00,  3.21s/it]
Dice Metric : 0.7208433747291565
Best valid Loss : 0.7208433747291565
EPOCH[VALID]1/1: 100%|██████████| 50/50 [02:41<00:00,  3.22s/it]
Dice Metric : 0.7748923897743225
Best valid Loss : 0.7748923897743225
EPOCH[VALID]1/1: 100%|██████████| 50/50 [02:39<00:00,  3.20s/it]
Dice Metric : 0.7700524926185608
Best valid Loss : 0.7700524926185608
EPOCH[VALID]1/1: 100%|██████████| 50/50 [02:39<00:00,  3.20s/it]
Dice Metric : 0.7686748504638672
Best valid Loss : 0.7686748504638672
EPOCH[VALID]1/1: 100%|██████████| 50/50 [02:39<00:00,  3.19s/it]
Dice Metric : 0.756960391998291
Best valid Loss : 0.756960391998291
EPOCH[VALID]1/1: 100%|██████████| 50/50 [02:40<00:00,  3.21s/it]
Dice Metric : 0.726111650466919
Best valid Loss : 0.726111650466919
EPOCH[VALID]1/1: 100%|██████████| 50/50 [02:41<00:00,  3.23s/it]
Dice Metric : 0.7648216485977173
Best valid Loss : 0.7648216485977173
EPOCH[VALID]1/1: 100%|██████████| 50/50 [02:39<00:00,  3.20s/it]
Dice Metric : 0.7656980156898499
Best valid Loss : 0.7656980156898499
EPOCH[VALID]1/1: 100%|██████████| 50/50 [02:39<00:00,  3.20s/it]
Dice Metric : 0.7549254298210144
Best valid Loss : 0.7549254298210144
EPOCH[VALID]1/1: 100%|██████████| 50/50 [02:38<00:00,  3.17s/it]
Dice Metric : 0.7913827300071716
Best valid Loss : 0.7913827300071716
EPOCH[VALID]1/1: 100%|██████████| 50/50 [02:39<00:00,  3.19s/it]
Dice Metric : 0.734478235244751
Best valid Loss : 0.734478235244751
EPOCH[VALID]1/1: 100%|██████████| 50/50 [02:39<00:00,  3.20s/it]
Dice Metric : 0.7514108419418335
Best valid Loss : 0.7514108419418335
EPOCH[VALID]1/1: 100%|██████████| 50/50 [02:37<00:00,  3.15s/it]
Dice Metric : 0.7145030498504639
Best valid Loss : 0.7145030498504639
EPOCH[VALID]1/1: 100%|██████████| 50/50 [02:39<00:00,  3.20s/it]
Dice Metric : 0.7466952800750732
Best valid Loss : 0.7466952800750732
EPOCH[VALID]1/1: 100%|██████████| 50/50 [02:38<00:00,  3.16s/it]
Dice Metric : 0.765889048576355
Best valid Loss : 0.765889048576355
EPOCH[VALID]1/1: 100%|██████████| 50/50 [02:40<00:00,  3.21s/it]
Dice Metric : 0.7169206738471985
Best valid Loss : 0.7169206738471985
EPOCH[VALID]1/1: 100%|██████████| 50/50 [02:38<00:00,  3.18s/it]
Dice Metric : 0.7674408555030823
Best valid Loss : 0.7674408555030823
EPOCH[VALID]1/1: 100%|██████████| 50/50 [02:37<00:00,  3.16s/it]
Dice Metric : 0.7875125408172607
Best valid Loss : 0.7875125408172607
EPOCH[VALID]1/1: 100%|██████████| 50/50 [02:43<00:00,  3.27s/it]
Dice Metric : 0.7561856508255005
Best valid Loss : 0.7561856508255005'''

dice_metric_values = re.findall(r"Dice Metric : (\d+\.\d+)", text)
dice_metric_values = [float(x) for x in dice_metric_values]
mean_dice_values = statistics.mean(dice_metric_values)
var_dice_metric = statistics.variance(dice_metric_values)

print(mean_dice_values,var_dice_metric)