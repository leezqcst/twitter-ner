import numpy as np

def macro_f1(flat_targets, flat_preds):
    scores = {}
    for target, pred in zip(flat_targets, flat_preds):
        if target not in scores:
            scores[target] = {'tp':0,
                              'fp':0,
                              'fn':0}
        if pred not in scores:
            scores[pred] = {'tp':0,
                            'fp':0,
                            'fn':0}
        if target == pred:
            scores[pred]['tp'] += 1
        
        else:
            scores[pred]['fp'] += 1
            scores[target]['fn'] += 1
        stats = {'scores':{}}
        for target, score in scores.items():
            precision = score['tp'] / float(score['tp'] + score['fp'] +1e-15)
            recall = score['tp'] / float(score['tp'] + score['fn'] + 1e-15)
            f1 = 2*precision*recall / (precision + recall + 1e-15)
            support = score['tp'] + score['fn']
            stats['scores'][target] = {'precision':precision,
                             'recall':recall,
                             'f1':f1,
                             'support':support}
        stats['macro_precision'] = np.mean([s['precision'] for s in stats['scores'].values()])
        stats['macro_recall'] = np.mean([s['recall'] for s in stats['scores'].values()]) 
        stats['macro_f1'] = np.mean([s['f1'] for s in stats['scores'].values()])
    return stats