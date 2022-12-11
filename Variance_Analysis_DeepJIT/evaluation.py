from model import DeepJIT
from utils import mini_batches_test
from sklearn.metrics import roc_auc_score    
import torch 
from tqdm import tqdm

def evaluation_model(data, params):
    pad_msg, pad_code, labels, dict_msg, dict_code = data
    batches = mini_batches_test(X_msg=pad_msg, X_code=pad_code, Y=labels)

    params.vocab_msg, params.vocab_code = len(dict_msg), len(dict_code)
    if len(labels.shape) == 1:
        params.class_num = 1
    else:
        params.class_num = labels.shape[1]

    # set up parameters
    params.cuda = (not params.no_cuda) and torch.cuda.is_available()
    del params.no_cuda
    params.filter_sizes = [int(k) for k in params.filter_sizes.split(',')]

    model = DeepJIT(args=params)
    if torch.cuda.is_available():
        model = model.cuda()
    model.load_state_dict(torch.load(params.load_model))

    model.eval()  # eval mode (batchnorm uses moving mean/variance instead of mini-batch mean/variance)
    with torch.no_grad():
        all_predict, all_label = list(), list()
        for i, (batch) in enumerate(tqdm(batches)):
            pad_msg, pad_code, label = batch
            if torch.cuda.is_available():                
                pad_msg, pad_code, labels = torch.tensor(pad_msg).cuda(), torch.tensor(
                    pad_code).cuda(), torch.cuda.FloatTensor(label)
            else:                
                pad_msg, pad_code, label = torch.tensor(pad_msg).long(), torch.tensor(pad_code).long(), torch.tensor(
                    labels).float()
            if torch.cuda.is_available():
                predict = model.forward(pad_msg, pad_code)
                predict = predict.cpu().detach().numpy().tolist()
            else:
                predict = model.forward(pad_msg, pad_code)
                predict = predict.detach().numpy().tolist()
            all_predict += predict
            all_label += labels.tolist()

    auc_score = roc_auc_score(y_true=all_label,  y_score=all_predict)
    print('Test data -- AUC score:', auc_score)

    # Calculating per-class accuracy with what is available:  I have all_labels and all_predictions given by the authors
    # Counting the number of correct commits and the number of faulty commits; Labels == 1.0 or 0.0; commit faulty if label == 1.0
    total_clean = 0
    total_faulty = 0
    for i in all_label:
        if i == 1.0:
            total_faulty = total_faulty + 1
        else:
            total_clean = total_clean + 1
    index = 0
    correctly_predicted_faulty = 0
    correctly_predicted_clean = 0
    while index < all_label.__len__():
        if all_label.__getitem__(index) == 1.0:
            if all_predict.__getitem__(index) >= 0.5:
                correctly_predicted_faulty = correctly_predicted_faulty + 1
        elif all_label.__getitem__(index) == 0.0:
            if all_predict.__getitem__(index) < 0.5:
                correctly_predicted_clean = correctly_predicted_clean + 1
        index += 1

    print('Number of clean commits in data: ' + str(total_clean))
    print('Number of faulty commits in data: ' + str(total_faulty))
    print('Number of commits correctly classified as clean: ' + str(correctly_predicted_clean))
    print('Number of commits correctly classified as false: ' + str(correctly_predicted_faulty))
    per_class_faulty = correctly_predicted_faulty/total_faulty
    per_class_correct = correctly_predicted_clean/total_clean
    # per-class accuracy of clean commits: Number of commits classified as correct/ number of total correct commits
    print('Per-class accuracy for correct commits: ' + str(per_class_correct))
    # per-class accuracy faulty commits: Number of commits classified as faulty / number of total faulty commits
    print('Per-class accuracy for faulty commits: ' + str(per_class_faulty))
    # True positives (TP) are modules correctly classified as faulty modules.
    print('TP: ' + str(correctly_predicted_faulty))
    # False positives (FP) refer to fault free modules incorrectly labeled as faulty (sometimes called false alarms)
    # aka 'clean modules' - 'modules correctly predicted as clean' = the number of clean modules remaining were incorrectly labeled as faulty
    FP = total_clean - correctly_predicted_clean
    print('FP: ' + str(FP))
    # True negatives (TN) correspond to correctly classified fault-free modules
    print('TN: ' + str(correctly_predicted_clean))
    # Finally, false negatives (FN) refer to faulty modules incorrectly classified as fault-free
    # aka "faulty modules" - "modules correctly predicted as faulty" = the number of faulty modules remaining were incorrectly labeled as clean
    FN = total_faulty - correctly_predicted_faulty
    print('FN: ' + str(FN))

